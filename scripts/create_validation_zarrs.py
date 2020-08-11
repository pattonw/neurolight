import gunpowder as gp
from gunpowder.batch_request import BatchRequest
import numpy as np
import neurolight as nl
from neurolight.transforms.swc_to_graph import parse_swc
from neurolight.gunpowder.nodes.snapshot_source import SnapshotSource
from neurolight.pipelines import DEFAULT_CONFIG
import networkx as nx

from neurolight.gunpowder.nodes.maxima import Skeletonize
from neurolight.gunpowder.nodes.emst import EMST
from neurolight.gunpowder.nodes.evaluate import Evaluate, MergeGraphs
from neurolight.gunpowder.nodes.clahe import scipyCLAHE

import mlpack as mlp
import torch

import copy
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)


def get_roi_from_swc(swc_cube, transform_file, voxel_size):
    graph = parse_swc(
        filename=swc_cube,
        transform=transform_file,
        offset=np.array([0, 0, 0]),
        resolution=np.array(voxel_size),
        transpose=[2, 1, 0],
    )
    mins = np.array([float("inf")] * 3)
    maxs = np.array([-float("inf")] * 3)
    for attrs in graph.nodes.values():
        u_loc = attrs["location"]
        min_stack = np.stack([mins, np.array(u_loc)])
        max_stack = np.stack([maxs, np.array(u_loc)])
        mins = np.min(min_stack, axis=0)
        maxs = np.max(max_stack, axis=0)

    return gp.Roi(gp.Coordinate(tuple(mins)), gp.Coordinate(tuple(maxs - mins)))


def get_validation_dir(benchmark_datasets_path, block):
    # map 1-5 -> 1, 6-10 -> 6, 11-15 -> 11, etc.
    def f(x):
        return ((x - 1) // 5) * 5 + 1

    # map 1-5 -> 5, 6-10 -> 10, 11-15 -> 15, etc.
    def g(x):
        return ((x + 4) // 5) * 5

    validation_dir = (
        benchmark_datasets_path
        / f"10-01_validation_{f(block)}-{g(block)}"
        / f"10-01_validation_{block}"
    )

    return validation_dir


def get_emb_model(config):
    model_config = copy.deepcopy(DEFAULT_CONFIG)
    if "EMB_MODEL_CONFIG" in config:
        model_config.update(json.load(open(config["EMB_MODEL_CONFIG"])))

    model = nl.networks.pytorch.EmbeddingUnet(model_config)

    device = config.get("DEVICE", "cuda")
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    if "EMB_MODEL_CHECKPOINT" in config:
        checkpoint_file = config["EMB_MODEL_CHECKPOINT"]
        print(f"Loading checkpoint: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise Exception()

    return model


def add_emb_pred(config, pipeline, raw, block, model):

    emb_pred = gp.ArrayKey(f"EMB_PRED_{block}")

    pipeline = (
        pipeline
        + nl.gunpowder.nodes.helpers.UnSqueeze(raw)
        + gp.torch.Predict(model, inputs={"raw": raw}, outputs={0: emb_pred})
        + nl.gunpowder.nodes.helpers.Squeeze(raw)
        + nl.gunpowder.nodes.helpers.Squeeze(emb_pred)
    )

    return pipeline, emb_pred


def get_fg_model(config):
    model_config = copy.deepcopy(DEFAULT_CONFIG)
    model_config.update(json.load(open(config["FG_MODEL_CONFIG"])))

    model = nl.networks.pytorch.ForegroundUnet(model_config)

    device = config.get("DEVICE", "cuda")
    checkpoint_file = config["FG_MODEL_CHECKPOINT"]
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict()

    return model


def add_fg_pred(config, pipeline, raw, block, model):
    checkpoint_file = config["FG_MODEL_CHECKPOINT"]

    device = config.get("DEVICE", "cuda")

    fg_pred = gp.ArrayKey(f"FG_PRED_{block}")

    pipeline = (
        pipeline
        + nl.gunpowder.nodes.helpers.UnSqueeze(raw)
        + gp.torch.Predict(
            model,
            inputs={"raw": raw},
            outputs={0: fg_pred},
            checkpoint=checkpoint_file,
            device=device,
        )
        + nl.gunpowder.nodes.helpers.Squeeze(raw)
        + nl.gunpowder.nodes.helpers.Squeeze(fg_pred)
    )

    return pipeline, fg_pred


def add_scan(pipeline, data_shapes):
    ref_request = gp.BatchRequest()
    for key, shape in data_shapes.items():
        ref_request.add(key, shape)
    pipeline = pipeline + gp.Scan(reference=ref_request)
    return pipeline


def validation_pipeline(config):
    """
    Per block
    {
        Raw -> predict -> scan
        gt -> rasterize        -> merge -> candidates -> trees
    } -> merge -> comatch + evaluate
    """
    blocks = config["BLOCKS"]
    benchmark_datasets_path = Path(config["BENCHMARK_DATA_PATH"])
    sample = config["VALIDATION_SAMPLES"][0]
    sample_dir = Path(config["SAMPLES_PATH"])
    raw_n5 = config["RAW_N5"]
    transform_template = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"

    neuron_width = int(config["NEURON_RADIUS"])
    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    micron_scale = max(voxel_size)
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    distance_attr = config["DISTANCE_ATTR"]

    validation_pipelines = []
    specs = {}

    for block in blocks:
        validation_dir = get_validation_dir(benchmark_datasets_path, block)
        trees = []
        cube = None
        for gt_file in validation_dir.iterdir():
            if gt_file.name[0:4] == "tree" and gt_file.name[-4:] == ".swc":
                trees.append(gt_file)
            if gt_file.name[0:4] == "cube" and gt_file.name[-4:] == ".swc":
                cube = gt_file
        assert cube.exists()

        cube_roi = get_roi_from_swc(
            cube,
            Path(transform_template.format(sample=sample)),
            np.array([300, 300, 1000]),
        )

        raw = gp.ArrayKey(f"RAW_{block}")
        raw_clahed = gp.ArrayKey(f"RAW_CLAHED_{block}")
        ground_truth = gp.GraphKey(f"GROUND_TRUTH_{block}")
        labels = gp.ArrayKey(f"LABELS_{block}")

        raw_source = (
            gp.ZarrSource(
                filename=str(Path(sample_dir, sample, raw_n5).absolute()),
                datasets={raw: "volume-rechunked", raw_clahed: "volume-rechunked"},
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                    raw_clahed: gp.ArraySpec(
                        interpolatable=True, voxel_size=voxel_size
                    ),
                },
            )
            + gp.Normalize(raw, dtype=np.float32)
            + gp.Normalize(raw_clahed, dtype=np.float32)
            + scipyCLAHE([raw_clahed], [20, 64, 64])
        )
        swc_source = nl.gunpowder.nodes.MouselightSwcFileSource(
            validation_dir,
            [ground_truth],
            transform_file=transform_template.format(sample=sample),
            ignore_human_nodes=False,
            scale=voxel_size,
            transpose=[2, 1, 0],
            points_spec=[
                gp.PointsSpec(
                    roi=gp.Roi(
                        gp.Coordinate([None, None, None]),
                        gp.Coordinate([None, None, None]),
                    )
                )
            ],
        )

        additional_request = BatchRequest()
        input_roi = cube_roi.grow(
            (input_size - output_size) // 2, (input_size - output_size) // 2
        )

        cube_roi_shifted = gp.Roi(
            (0,) * len(cube_roi.get_shape()), cube_roi.get_shape()
        )
        input_roi = cube_roi_shifted.grow(
            (input_size - output_size) // 2, (input_size - output_size) // 2
        )

        block_spec = specs.setdefault(block, {})
        block_spec[raw] = gp.ArraySpec(input_roi)
        additional_request[raw] = gp.ArraySpec(roi=input_roi)
        block_spec[raw_clahed] = gp.ArraySpec(input_roi)
        additional_request[raw_clahed] = gp.ArraySpec(roi=input_roi)
        block_spec[ground_truth] = gp.GraphSpec(cube_roi_shifted)
        additional_request[ground_truth] = gp.GraphSpec(roi=cube_roi_shifted)
        block_spec[labels] = gp.ArraySpec(cube_roi_shifted)
        additional_request[labels] = gp.ArraySpec(roi=cube_roi_shifted)

        pipeline = (
            (swc_source, raw_source)
            + gp.nodes.MergeProvider()
            + gp.SpecifiedLocation(locations=[cube_roi.get_center()])
            + gp.Crop(raw, roi=input_roi)
            + gp.Crop(raw_clahed, roi=input_roi)
            + gp.Crop(ground_truth, roi=cube_roi_shifted)
            + nl.gunpowder.RasterizeSkeleton(
                ground_truth,
                labels,
                connected_component_labeling=True,
                array_spec=gp.ArraySpec(
                    voxel_size=voxel_size,
                    dtype=np.int64,
                    roi=gp.Roi(
                        gp.Coordinate([None, None, None]),
                        gp.Coordinate([None, None, None]),
                    ),
                ),
            )
            + nl.gunpowder.GrowLabels(labels, radii=[neuron_width * micron_scale])
            + gp.Crop(labels, roi=cube_roi_shifted)
            + gp.Snapshot(
                {
                    raw: f"volumes/{block}/raw",
                    raw_clahed: f"volumes/{block}/raw_clahe",
                    ground_truth: f"points/{block}/ground_truth",
                    labels: f"volumes/{block}/labels",
                },
                additional_request=additional_request,
                output_dir="validations",
                output_filename="validations.hdf",
            )
        )

        validation_pipelines.append(pipeline)

    validation_pipeline = (
        tuple(pipeline for pipeline in validation_pipelines)
        + gp.MergeProvider()
        + gp.PrintProfilingStats()
    )
    return validation_pipeline, specs

if __name__ == "__main__":
    DEFAULT_CONFIG["BLOCKS"] = list(range(1, 26))
    pipeline, specs = validation_pipeline(DEFAULT_CONFIG)
    request = gp.BatchRequest()
    for block, block_specs in specs.items():
        for array, array_spec in block_specs.items():
            request[array] = array_spec
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
