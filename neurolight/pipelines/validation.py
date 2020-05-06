import gunpowder as gp
import numpy as np
import neurolight as nl
from neurolight.transforms.swc_to_graph import parse_swc
from neurolight.gunpowder.nodes.snapshot_source import SnapshotSource
from neurolight.pipelines import DEFAULT_CONFIG
import networkx as nx

import mlpack as mlp

import copy
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)

# config_file = Path("./config.ini")
# config = parse_config(config_file)
#
# benchmark_datasets_path = config["benchmark_dir"]


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

    assert all(
        abs(maxs - mins - np.array([100000] * 3)) < np.array([10000, 10000, 10000])
    )
    return gp.Roi(gp.Coordinate(tuple(mins)), gp.Coordinate(tuple(maxs - mins)))


def validation_data_sources(config, blocks=list(range(1, 26)), hdf5=False):

    if hdf5:
        return validation_data_sources_from_snapshots(config, blocks=blocks)
    else:
        return validation_data_sources_recomputed(config, blocks=blocks)


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


def get_validation_cube_roi(benchmark_datasets_path, transform_file, block):

    validation_dir = get_validation_dir(benchmark_datasets_path, block)
    cube_swc = validation_dir / f"cube_100um_validation_{block}.swc"
    assert cube_swc.exists()

    cube_roi = get_roi_from_swc(
        cube_swc, Path(transform_file), np.array([300, 300, 1000])
    )

    return cube_roi


def get_cube_roi(config, block):

    benchmark_datasets_path = Path(config["BENCHMARK_DATA_PATH"])
    sample = config["VALIDATION_SAMPLES"][0]
    transform_template = "/nrs/mouselight/SAMPLES/2018-10-01/transform.txt"

    return get_validation_cube_roi(benchmark_datasets_path, transform_template, block)


def get_raw_snapshot_source(config, blocks):
    validation_blocks = Path(config["VALIDATION_BLOCKS"])

    raw = gp.ArrayKey("RAW")

    block_pipelines = []
    for block in blocks:

        pipeline = SnapshotSource(
            validation_blocks / f"block_{block}.hdf", {raw: "volumes/raw"}
        )

        block_pipelines.append(pipeline)
    return block_pipelines, (raw,)


def get_labels_snapshot_source(config, blocks):
    validation_blocks = Path(config["VALIDATION_BLOCKS"])

    labels = gp.ArrayKey("LABELS")
    gt = gp.GraphKey("GT")

    block_pipelines = []
    for block in blocks:

        pipeline = SnapshotSource(
            validation_blocks / f"block_{block}.hdf",
            {labels: "volumes/labels", gt: "points/gt"},
            directed={gt: True},
        )

        block_pipelines.append(pipeline)
    return block_pipelines, (labels, gt)


def get_requests(config, blocks, raw, emb_pred, labels, gt):
    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape
    diff = input_size - output_size

    cube_rois = [get_cube_roi(config, block) for block in blocks]

    requests = []
    for cube_roi in cube_rois:
        context_roi = cube_roi.grow(diff // 2, diff // 2)
        request = gp.BatchRequest()
        request[raw] = gp.ArraySpec(roi=context_roi)
        request[emb_pred] = gp.ArraySpec(roi=cube_roi)
        request[labels] = gp.ArraySpec(roi=cube_roi)
        request[gt] = gp.GraphSpec(roi=cube_roi)
        requests.append(request)
    return requests


def get_embedding_pipelines(config, blocks):
    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    raw_pipelines, (raw,) = get_raw_snapshot_source(config, blocks)
    labels_pipelines, (labels, gt) = get_labels_snapshot_source(config, blocks)
    emb_pipelines, (emb_pred,) = add_emb_preds(config, raw_pipelines, raw)
    emb_pipelines = add_scans(emb_pipelines, {raw: input_size, emb_pred: output_size})
    pipelines = merge_pipelines(emb_pipelines, labels_pipelines)

    requests = get_requests(config, blocks, raw, emb_pred, labels, gt)

    return pipelines, requests, (raw, emb_pred, labels, gt)


def get_foreground_pipelines(config, blocks):
    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    raw_pipelines, (raw,) = get_raw_snapshot_source(config, blocks)
    labels_pipelines, (labels, gt) = get_labels_snapshot_source(config, blocks)
    fg_pipelines, (fg_pred,) = add_fg_preds(config, raw_pipelines, raw)
    fg_pipelines = add_scans(fg_pipelines, {raw: input_size, fg_pred: output_size})
    pipelines = merge_pipelines(fg_pipelines, labels_pipelines)

    requests = get_requests(config, blocks, raw, fg_pred, labels, gt)

    return pipelines, requests, (raw, fg_pred, labels, gt)


def merge_pipelines(pipelines_a, pipelines_b):
    merged_pipelines = []
    for a, b in zip(pipelines_a, pipelines_b):
        merged_pipelines.append((a, b) + gp.MergeProvider())
    return merged_pipelines


def validation_data_sources_from_snapshots(config, blocks):
    validation_blocks = Path(config["VALIDATION_BLOCKS"])

    raw = gp.ArrayKey("RAW")
    ground_truth = gp.GraphKey("GROUND_TRUTH")
    labels = gp.ArrayKey("LABELS")

    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    block_pipelines = []
    for block in blocks:

        pipelines = (
            SnapshotSource(
                validation_blocks / f"block_{block}.hdf",
                {labels: "volumes/labels", ground_truth: "points/gt"},
                directed={ground_truth: True},
            ),
            SnapshotSource(
                validation_blocks / f"block_{block}.hdf", {raw: "volumes/raw"}
            ),
        )

        cube_roi = get_cube_roi(config, block)

        request = gp.BatchRequest()
        input_roi = cube_roi.grow(
            (input_size - output_size) // 2, (input_size - output_size) // 2
        )
        request[raw] = gp.ArraySpec(input_roi)
        request[ground_truth] = gp.GraphSpec(cube_roi)
        request[labels] = gp.ArraySpec(cube_roi)

        block_pipelines.append((pipelines, request))
    return block_pipelines, (raw, labels, ground_truth)


def validation_data_sources_recomputed(config, blocks):
    benchmark_datasets_path = Path(config["BENCHMARK_DATA_PATH"])
    sample = config["VALIDATION_SAMPLES"][0]
    sample_dir = Path(config["SAMPLES_PATH"])
    raw_n5 = config["RAW_N5"]
    transform_template = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"

    neuron_width = int(config["NEURON_WIDTH"])
    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    validation_dirs = {}
    for group in benchmark_datasets_path.iterdir():
        if "validation" in group.name and group.is_dir():
            for validation_dir in group.iterdir():
                validation_num = int(validation_dir.name.split("_")[-1])
                if validation_num in blocks:
                    validation_dirs[validation_num] = validation_dir

    validation_dirs = [validation_dirs[block] for block in blocks]

    raw = gp.ArrayKey("RAW")
    ground_truth = gp.GraphKey("GROUND_TRUTH")
    labels = gp.ArrayKey("LABELS")

    validation_pipelines = []
    for validation_dir in validation_dirs:
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

        pipeline = (
            (
                gp.ZarrSource(
                    filename=str(Path(sample_dir, sample, raw_n5).absolute()),
                    datasets={raw: "volume"},
                    array_specs={
                        raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size)
                    },
                ),
                nl.gunpowder.nodes.MouselightSwcFileSource(
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
                ),
            )
            + gp.nodes.MergeProvider()
            + gp.Normalize(raw, dtype=np.float32)
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
            + nl.gunpowder.GrowLabels(labels, radii=[neuron_width * 1000])
        )

        request = gp.BatchRequest()
        input_roi = cube_roi.grow(
            (input_size - output_size) // 2, (input_size - output_size) // 2
        )
        print(f"input_roi has shape: {input_roi.get_shape()}")
        print(f"cube_roi has shape: {cube_roi.get_shape()}")
        request[raw] = gp.ArraySpec(input_roi)
        request[ground_truth] = gp.GraphSpec(cube_roi)
        request[labels] = gp.ArraySpec(cube_roi)

        validation_pipelines.append((pipeline, request))
    return validation_pipelines, (raw, labels, ground_truth)


def add_fg_preds(config, pipelines, raw):
    model_config = copy.deepcopy(DEFAULT_CONFIG)
    model_config.update(json.load(open(config["FG_MODEL_CONFIG"])))
    checkpoint_file = config["FG_MODEL_CHECKPOINT"]

    device = config.get("DEVICE", "cuda")

    fg_pred = gp.ArrayKey("FG_PRED")

    fg_pipelines = []
    for pipeline in pipelines:
        fg_pipelines.append(
            pipeline
            + nl.gunpowder.nodes.helpers.UnSqueeze(raw)
            + gp.torch.Predict(
                nl.networks.pytorch.ForegroundUnet(model_config),
                inputs={"raw": raw},
                outputs={0: fg_pred},
                checkpoint=checkpoint_file,
                device=device,
            )
            + nl.gunpowder.nodes.helpers.Squeeze(raw)
            + nl.gunpowder.nodes.helpers.Squeeze(fg_pred)
        )

    return fg_pipelines, (fg_pred,)


def add_emb_preds(config, pipelines, raw):
    model_config = copy.deepcopy(DEFAULT_CONFIG)
    model_config.update(json.load(open(config["EMB_MODEL_CONFIG"])))
    device = config.get("DEVICE", "cuda")

    checkpoint_file = config["EMB_MODEL_CHECKPOINT"]

    emb_pred = gp.ArrayKey("EMB_PRED")

    emb_pipelines = []
    for pipeline in pipelines:
        emb_pipelines.append(
            pipeline
            + nl.gunpowder.nodes.helpers.UnSqueeze(raw)
            + gp.torch.Predict(
                nl.networks.pytorch.EmbeddingUnet(model_config),
                inputs={"raw": raw},
                outputs={0: emb_pred},
                checkpoint=checkpoint_file,
                device=device,
            )
            + nl.gunpowder.nodes.helpers.Squeeze(raw)
            + nl.gunpowder.nodes.helpers.Squeeze(emb_pred)
        )

    return emb_pipelines, (emb_pred,)


def add_nms(pipeline, config, foreground):
    model_config = copy.deepcopy(DEFAULT_CONFIG)
    model_config.update(json.load(open(config["EMB_MODEL_CONFIG"])))

    # Data properties
    voxel_size = gp.Coordinate(model_config["VOXEL_SIZE"])
    micron_scale = voxel_size[0]

    # Config options
    window_size = gp.Coordinate(model_config["NMS_WINDOW_SIZE"]) * micron_scale
    threshold = model_config["NMS_THRESHOLD"]

    # New array Key
    maxima = gp.ArrayKey("MAXIMA")

    pipeline = (
        pipeline
        + nl.gunpowder.nodes.helpers.UnSqueeze(foreground)
        + nl.gunpowder.nodes.NonMaxSuppression(
            foreground, maxima, window_size, threshold
        )
        + nl.gunpowder.nodes.helpers.Squeeze(foreground)
        + nl.gunpowder.nodes.helpers.Squeeze(maxima)
    )

    return pipeline, maxima


def add_scans(pipelines, data_shapes):
    scanned_pipelines = []
    for pipeline in pipelines:
        ref_request = gp.BatchRequest()
        for key, shape in data_shapes.items():
            ref_request.add(key, shape)
        scanned_pipelines.append(pipeline + gp.Scan(reference=ref_request))
    return scanned_pipelines


def get_embedding_mst(embedding, coordinate_scale, voxel_size, offset, candidates):

    _, depth, height, width = embedding.shape
    coordinates = np.meshgrid(
        np.arange(0, (depth - 0.5) * coordinate_scale[0], coordinate_scale[0]),
        np.arange(0, (height - 0.5) * coordinate_scale[1], coordinate_scale[1]),
        np.arange(0, (width - 0.5) * coordinate_scale[2], coordinate_scale[2]),
        indexing="ij",
    )
    for i in range(len(coordinates)):
        coordinates[i] = coordinates[i].astype(np.float32)

    embedding = np.concatenate([embedding, coordinates], 0)
    embedding = np.transpose(embedding, axes=[1, 2, 3, 0])
    embedding = embedding.reshape(depth * width * height, -1)
    candidates = candidates.reshape(depth * width * height)
    embedding = embedding[candidates == 1, :]

    emst = mlp.emst(embedding)["output"]

    mst = nx.DiGraph()
    for u, v, distance in emst:
        u = int(u)
        pos_u = (embedding[u][-3:] / coordinate_scale) * voxel_size
        v = int(v)
        pos_v = (embedding[v][-3:] / coordinate_scale) * voxel_size
        mst.add_node(u, location=pos_u + offset)
        mst.add_node(v, location=pos_v + offset)
        mst.add_edge(u, v, d=distance)
    for node, attrs in mst.nodes.items():
        assert "location" in attrs
    return mst
