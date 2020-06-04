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

        checkpoint = torch.load(checkpoint_file, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict()

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
    candidate_threshold = config["NMS_THRESHOLD"]
    candidate_spacing = min(config["NMS_WINDOW_SIZE"]) * micron_scale
    coordinate_scale = config["COORDINATE_SCALE"] * np.array(voxel_size) / micron_scale

    emb_model = get_emb_model(config)
    fg_model = get_fg_model(config)

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
        ground_truth = gp.GraphKey(f"GROUND_TRUTH_{block}")
        labels = gp.ArrayKey(f"LABELS_{block}")
        candidates = gp.ArrayKey(f"CANDIDATES_{block}")
        mst = gp.GraphKey(f"MST_{block}")

        raw_source = gp.ZarrSource(
            filename=str(Path(sample_dir, sample, raw_n5).absolute()),
            datasets={raw: "volume-rechunked"},
            array_specs={raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size)},
        ) + gp.Normalize(raw, dtype=np.float32)
        emb_source, emb = add_emb_pred(config, raw_source, raw, block, emb_model)
        pred_source, fg = add_fg_pred(config, emb_source, raw, block, fg_model)
        pred_source = add_scan(
            pred_source, {raw: input_size, emb: output_size, fg: output_size}
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
        block_spec = specs.setdefault(block, {})
        block_spec["raw"] = (raw, gp.ArraySpec(input_roi))
        additional_request[raw] = gp.ArraySpec(roi=input_roi)
        block_spec["ground_truth"] = (ground_truth, gp.GraphSpec(cube_roi))
        additional_request[ground_truth] = gp.GraphSpec(roi=cube_roi)
        block_spec["labels"] = (labels, gp.ArraySpec(cube_roi))
        additional_request[labels] = gp.ArraySpec(roi=cube_roi)
        block_spec["fg_pred"] = (fg, gp.ArraySpec(cube_roi))
        additional_request[fg] = gp.ArraySpec(roi=cube_roi)
        block_spec["emb_pred"] = (emb, gp.ArraySpec(cube_roi))
        additional_request[emb] = gp.ArraySpec(roi=cube_roi)
        block_spec["candidates"] = (candidates, gp.ArraySpec(cube_roi))
        additional_request[candidates] = gp.ArraySpec(roi=cube_roi)
        block_spec["mst_pred"] = (mst, gp.GraphSpec(cube_roi))
        additional_request[mst] = gp.GraphSpec(roi=cube_roi)

        pipeline = (
            (swc_source, pred_source)
            + gp.nodes.MergeProvider()
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
            + Skeletonize(fg, candidates, candidate_spacing, candidate_threshold)
            + EMST(
                emb,
                candidates,
                mst,
                distance_attr=distance_attr,
                coordinate_scale=coordinate_scale,
            )
            + gp.Snapshot(
                {
                    raw: f"volumes/{raw}",
                    ground_truth: f"points/{ground_truth}",
                    labels: f"volumes/{labels}",
                    fg: f"volumes/{fg}",
                    emb: f"volumes/{emb}",
                    candidates: f"volumes/{candidates}",
                    mst: f"points/{mst}",
                },
                additional_request=additional_request,
                output_dir="snapshots",
                output_filename="{id}.hdf",
            )
        )

        validation_pipelines.append(pipeline)

    full_gt = gp.GraphKey("FULL_GT")
    full_mst = gp.GraphKey("FULL_MST")
    score = gp.ArrayKey("SCORE")

    validation_pipeline = (
        tuple(pipeline for pipeline in validation_pipelines)
        + gp.MergeProvider()
        + MergeGraphs(specs, full_gt, full_mst)
        + Evaluate(full_gt, full_mst, score, edge_threshold_attr=distance_attr)
        + gp.PrintProfilingStats()
    )
    return validation_pipeline, score
