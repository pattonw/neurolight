import gunpowder as gp
from gunpowder.batch_request import BatchRequest
import numpy as np
import neurolight as nl
from neurolight.transforms.swc_to_graph import parse_swc
from neurolight.gunpowder.nodes.snapshot_source import SnapshotSource
from neurolight.pipelines import DEFAULT_CONFIG

from neurolight.gunpowder.nodes.maxima import Skeletonize

from neurolight.gunpowder.nodes.minimax import MiniMax, MiniMaxEmbeddings
from neurolight.gunpowder.nodes.emst import EMST
from neurolight.gunpowder.nodes.evaluate import Evaluate, MergeGraphs
from neurolight.gunpowder.nodes.clahe import scipyCLAHE
from neurolight.gunpowder.nodes.score_edges import ScoreEdges
from neurolight.gunpowder.nodes.threshold_edges import ThresholdEdges
from neurolight.gunpowder.nodes.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.nodes.emst_components import ComponentWiseEMST

import torch

import copy
from pathlib import Path
import logging
import json


class MergeScores(gp.BatchFilter):
    def __init__(self, output, inputs):
        self.output = output
        self.inputs = inputs

    def setup(self):
        self.provides(self.output, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        deps = gp.BatchRequest()
        for block, block_specs in self.inputs.items():
            for key, spec in block_specs.items():
                if "SCORE" in str(key):
                    deps[key] = spec

        return deps

    def process(self, batch, request):
        final_scores = {}
        for key, array in batch.items():
            if "SCORE" in str(key):
                block = int(str(key).split("_")[1])
                final_scores[block] = array.data
        final_scores = [
            final_scores[block] for block in range(1, 26) if block in final_scores
        ]
        outputs = gp.Batch()
        outputs[self.output] = gp.Array(
            np.array(final_scores), gp.ArraySpec(nonspatial=True)
        )
        return outputs


def emb_validation_pipeline(
    config,
    snapshot_file,
    candidates_path,
    raw_path,
    gt_path,
    candidates_mst_path=None,
    candidates_mst_dense_path=None,
    path_stat="max",
):
    checkpoint = config["EMB_EVAL_CHECKPOINT"]
    blocks = config["BLOCKS"]
    benchmark_datasets_path = Path(config["BENCHMARK_DATA_PATH"])
    sample = config["VALIDATION_SAMPLES"][0]
    transform_template = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"

    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    micron_scale = max(voxel_size)
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    distance_attr = config["DISTANCE_ATTR"]
    coordinate_scale = config["COORDINATE_SCALE"] * np.array(voxel_size) / micron_scale
    num_thresholds = config["NUM_EVAL_THRESHOLDS"]
    threshold_range = config["EVAL_THRESHOLD_RANGE"]

    edge_threshold_0 = config["EVAL_EDGE_THRESHOLD_0"]
    component_threshold_0 = config["COMPONENT_THRESHOLD_0"]
    component_threshold_1 = config["COMPONENT_THRESHOLD_1"]

    clip_limit = config["CLAHE_CLIP_LIMIT"]
    normalize = config["CLAHE_NORMALIZE"]

    validation_pipelines = []
    specs = {}

    emb_model = get_emb_model(config)
    emb_model.eval()

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
            np.array(voxel_size[::-1]),
        )

        candidates_1 = gp.ArrayKey(f"CANDIDATES_1_{block}")

        raw = gp.ArrayKey(f"RAW_{block}")
        mst_0 = gp.GraphKey(f"MST_0_{block}")
        mst_dense_0 = gp.GraphKey(f"MST_DENSE_0_{block}")
        mst_1 = gp.GraphKey(f"MST_1_{block}")
        mst_dense_1 = gp.GraphKey(f"MST_DENSE_1_{block}")
        mst_2 = gp.GraphKey(f"MST_2_{block}")
        mst_dense_2 = gp.GraphKey(f"MST_DENSE_2_{block}")
        gt = gp.GraphKey(f"GT_{block}")
        score = gp.ArrayKey(f"SCORE_{block}")
        details = gp.GraphKey(f"DETAILS_{block}")
        optimal_mst = gp.GraphKey(f"OPTIMAL_MST_{block}")

        # Volume Source
        raw_source = SnapshotSource(
            snapshot_file,
            datasets={
                raw: raw_path.format(block=block),
                candidates_1: candidates_path.format(block=block),
            },
        )

        # Graph Source
        graph_datasets = {gt: gt_path.format(block=block)}
        graph_directionality = {gt: False}
        edge_attrs = {}
        if candidates_mst_path is not None:
            graph_datasets[mst_0] = candidates_mst_path.format(block=block)
            graph_directionality[mst_0] = False
            edge_attrs[mst_0] = [distance_attr]
        if candidates_mst_dense_path is not None:
            graph_datasets[mst_dense_0] = candidates_mst_dense_path.format(block=block)
            graph_directionality[mst_dense_0] = False
            edge_attrs[mst_dense_0] = [distance_attr]
        gt_source = SnapshotSource(
            snapshot_file,
            datasets=graph_datasets,
            directed=graph_directionality,
            edge_attrs=edge_attrs,
        )

        if config["EVAL_CLAHE"]:
            raw_source = raw_source + scipyCLAHE(
                [raw],
                gp.Coordinate([20, 64, 64]) * voxel_size,
                clip_limit=clip_limit,
                normalize=normalize,
            )
        else:
            pass

        emb_source, emb, neighborhood = add_emb_pred(
            config, raw_source, raw, block, emb_model
        )

        reference_sizes = {raw: input_size, emb: output_size, candidates_1: output_size}
        if neighborhood is not None:
            reference_sizes[neighborhood] = output_size

        emb_source = add_scan(emb_source, reference_sizes)

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
        block_spec[candidates_1] = gp.ArraySpec(cube_roi_shifted)
        block_spec[emb] = gp.ArraySpec(cube_roi_shifted)
        if neighborhood is not None:
            block_spec[neighborhood] = gp.ArraySpec(cube_roi_shifted)
        block_spec[gt] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[mst_0] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[mst_dense_0] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[mst_1] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[mst_dense_1] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[mst_2] = gp.GraphSpec(cube_roi_shifted, directed=False)
        # block_spec[mst_dense_2] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[score] = gp.ArraySpec(nonspatial=True)
        block_spec[optimal_mst] = gp.GraphSpec(cube_roi_shifted, directed=False)

        additional_request = BatchRequest()
        additional_request[raw] = gp.ArraySpec(input_roi)
        additional_request[candidates_1] = gp.ArraySpec(cube_roi_shifted)
        additional_request[emb] = gp.ArraySpec(cube_roi_shifted)
        if neighborhood is not None:
            additional_request[neighborhood] = gp.ArraySpec(cube_roi_shifted)
        additional_request[gt] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[mst_0] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[mst_dense_0] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[mst_1] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[mst_dense_1] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[mst_2] = gp.GraphSpec(cube_roi_shifted, directed=False)
        # additional_request[mst_dense_2] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[details] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[optimal_mst] = gp.GraphSpec(cube_roi_shifted, directed=False)

        pipeline = (emb_source, gt_source) + gp.MergeProvider()

        if candidates_mst_path is not None and candidates_mst_dense_path is not None:
            # mst_0 provided, just need to calculate distances.
            pass
        elif config["EVAL_MINIMAX_EMBEDDING_DIST"]:
            # No mst_0 provided, must first calculate mst_0 and dense mst_0
            pipeline += MiniMaxEmbeddings(
                emb,
                candidates_1,
                decimated=mst_0,
                dense=mst_dense_0,
                distance_attr=distance_attr,
            )

        else:
            # mst/mst_dense not provided. Simply use euclidean distance on candidates
            pipeline += EMST(
                emb,
                candidates_1,
                mst_0,
                distance_attr=distance_attr,
                coordinate_scale=coordinate_scale,
            )
            pipeline += EMST(
                emb,
                candidates_1,
                mst_dense_0,
                distance_attr=distance_attr,
                coordinate_scale=coordinate_scale,
            )

        pipeline += ThresholdEdges(
            (mst_0, mst_1),
            edge_threshold_0,
            component_threshold_0,
            msts_dense=(mst_dense_0, mst_dense_1),
            distance_attr=distance_attr,
        )

        pipeline += ComponentWiseEMST(
            emb,
            mst_1,
            mst_2,
            distance_attr=distance_attr,
            coordinate_scale=coordinate_scale,
        )

        # pipeline += ScoreEdges(
        #     mst, mst_dense, emb, distance_attr=distance_attr, path_stat=path_stat
        # )

        pipeline += Evaluate(
            gt,
            mst_2,
            score,
            roi=cube_roi_shifted,
            details=details,
            edge_threshold_attr=distance_attr,
            num_thresholds=num_thresholds,
            threshold_range=threshold_range,
            small_component_threshold=component_threshold_1,
            # connectivity=mst_1,
            output_graph=optimal_mst,
        )

        if config["EVAL_SNAPSHOT"]:
            snapshot_datasets = {
                raw: f"volumes/raw",
                emb: f"volumes/embeddings",
                candidates_1: f"volumes/candidates_1",
                mst_0: f"points/mst_0",
                mst_dense_0: f"points/mst_dense_0",
                mst_1: f"points/mst_1",
                mst_dense_1: f"points/mst_dense_1",
                # mst_2: f"points/mst_2",
                gt: f"points/gt",
                details: f"points/details",
                optimal_mst: f"points/optimal_mst",
            }
            if neighborhood is not None:
                snapshot_datasets[neighborhood] = f"volumes/neighborhood"
            pipeline += gp.Snapshot(
                snapshot_datasets,
                output_dir=config["EVAL_SNAPSHOT_DIR"],
                output_filename=config["EVAL_SNAPSHOT_NAME"].format(
                    checkpoint=checkpoint,
                    block=block,
                    coordinate_scale=",".join([str(x) for x in coordinate_scale]),
                ),
                edge_attrs={
                    mst_0: [distance_attr],
                    mst_dense_0: [distance_attr],
                    mst_1: [distance_attr],
                    mst_dense_1: [distance_attr],
                    # mst_2: [distance_attr],
                    # optimal_mst: [distance_attr], # it is unclear how to add distances if using connectivity graph
                    # mst_dense_2: [distance_attr],
                    details: ["details", "label_pair"],
                },
                node_attrs={details: ["details", "label_pair"]},
                additional_request=additional_request,
            )

        validation_pipelines.append(pipeline)

    final_score = gp.ArrayKey("SCORE")

    validation_pipeline = (
        tuple(pipeline for pipeline in validation_pipelines)
        + gp.MergeProvider()
        + MergeScores(final_score, specs)
        + gp.PrintProfilingStats()
    )
    return validation_pipeline, final_score


def fg_validation_pipeline(config, snapshot_file, raw_path, gt_path):
    checkpoint = config["FG_EVAL_CHECKPOINT"]
    blocks = config["BLOCKS"]
    benchmark_datasets_path = Path(config["BENCHMARK_DATA_PATH"])
    sample = config["VALIDATION_SAMPLES"][0]
    transform_template = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"

    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    candidate_spacing = config["CANDIDATE_SPACING"]
    candidate_threshold = config["CANDIDATE_THRESHOLD"]

    distance_attr = config["DISTANCE_ATTR"]
    num_thresholds = config["NUM_EVAL_THRESHOLDS"]
    threshold_range = config["EVAL_THRESHOLD_RANGE"]

    component_threshold = config["COMPONENT_THRESHOLD_1"]

    clip_limit = config["CLAHE_CLIP_LIMIT"]
    normalize = config["CLAHE_NORMALIZE"]

    validation_pipelines = []
    specs = {}

    fg_model = get_fg_model(config)
    fg_model.eval()

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
            np.array(voxel_size[::-1]),
        )

        candidates = gp.ArrayKey(f"CANDIDATES_{block}")
        raw = gp.ArrayKey(f"RAW_{block}")
        mst = gp.GraphKey(f"MST_{block}")
        gt = gp.GraphKey(f"GT_{block}")
        score = gp.ArrayKey(f"SCORE_{block}")
        details = gp.GraphKey(f"DETAILS_{block}")

        raw_source = SnapshotSource(
            snapshot_file, datasets={raw: raw_path.format(block=block)}
        )
        gt_source = SnapshotSource(
            snapshot_file,
            datasets={gt: gt_path.format(block=block)},
            directed={gt: False},
        )

        if config["EVAL_CLAHE"]:
            raw_source = raw_source + scipyCLAHE(
                [raw],
                gp.Coordinate([20, 64, 64]) * voxel_size,
                clip_limit=clip_limit,
                normalize=normalize,
            )
        else:
            pass

        fg_source, fg = add_fg_pred(config, raw_source, raw, block, fg_model)

        fg_source = add_scan(fg_source, {raw: input_size, fg: output_size})

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
        block_spec[candidates] = gp.ArraySpec(cube_roi_shifted)
        block_spec[fg] = gp.ArraySpec(cube_roi_shifted)
        block_spec[gt] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[mst] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[score] = gp.ArraySpec(nonspatial=True)

        additional_request = BatchRequest()
        additional_request[raw] = gp.ArraySpec(input_roi)
        additional_request[candidates] = gp.ArraySpec(cube_roi_shifted)
        additional_request[fg] = gp.ArraySpec(cube_roi_shifted)
        additional_request[gt] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[mst] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[details] = gp.GraphSpec(cube_roi_shifted, directed=False)

        pipeline = (
            (fg_source, gt_source)
            + gp.MergeProvider()
            + Skeletonize(fg, candidates, candidate_spacing, candidate_threshold)
            + MiniMax(fg, candidates, mst, distance_attr=distance_attr)
        )

        pipeline += Evaluate(
            gt,
            mst,
            score,
            roi=cube_roi_shifted,
            details=details,
            edge_threshold_attr=distance_attr,
            num_thresholds=num_thresholds,
            threshold_range=threshold_range,
            small_component_threshold=component_threshold,
        )

        if config["EVAL_SNAPSHOT"]:
            pipeline += gp.Snapshot(
                {
                    raw: f"volumes/raw",
                    fg: f"volumes/foreground",
                    candidates: f"volumes/candidates",
                    mst: f"points/mst",
                    gt: f"points/gt",
                    details: f"points/details",
                },
                output_dir=config["EVAL_SNAPSHOT_DIR"],
                output_filename=config["EVAL_SNAPSHOT_NAME"].format(
                    checkpoint=checkpoint, block=block
                ),
                edge_attrs={mst: [distance_attr], details: ["details", "label_pair"]},
                node_attrs={details: ["details", "label_pair"]},
                additional_request=additional_request,
            )

        validation_pipelines.append(pipeline)

    final_score = gp.ArrayKey("SCORE")

    validation_pipeline = (
        tuple(pipeline for pipeline in validation_pipelines)
        + gp.MergeProvider()
        + MergeScores(final_score, specs)
        + gp.PrintProfilingStats()
    )
    return validation_pipeline, final_score


def pre_computed_fg_validation_pipeline(
    config, snapshot_file, raw_path, gt_path, fg_path
):
    blocks = config["BLOCKS"]
    benchmark_datasets_path = Path(config["BENCHMARK_DATA_PATH"])
    sample = config["VALIDATION_SAMPLES"][0]
    transform_template = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"

    voxel_size = gp.Coordinate(config["VOXEL_SIZE"])
    input_shape = gp.Coordinate(config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(config["OUTPUT_SHAPE"])
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    candidate_spacing = config["CANDIDATE_SPACING"]
    candidate_threshold = config["CANDIDATE_THRESHOLD"]

    distance_attr = config["DISTANCE_ATTR"]
    num_thresholds = config["NUM_EVAL_THRESHOLDS"]
    threshold_range = config["EVAL_THRESHOLD_RANGE"]

    component_threshold = config["COMPONENT_THRESHOLD_1"]

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
            np.array(voxel_size[::-1]),
        )

        candidates = gp.ArrayKey(f"CANDIDATES_{block}")
        raw = gp.ArrayKey(f"RAW_{block}")
        mst = gp.GraphKey(f"MST_{block}")
        gt = gp.GraphKey(f"GT_{block}")
        fg = gp.ArrayKey(f"FG_{block}")
        score = gp.ArrayKey(f"SCORE_{block}")
        details = gp.GraphKey(f"DETAILS_{block}")

        raw_source = SnapshotSource(
            snapshot_file,
            datasets={
                raw: raw_path.format(block=block),
                fg: fg_path.format(block=block),
            },
        )
        gt_source = SnapshotSource(
            snapshot_file,
            datasets={gt: gt_path.format(block=block)},
            directed={gt: False},
        )

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
        block_spec[candidates] = gp.ArraySpec(cube_roi_shifted)
        block_spec[fg] = gp.ArraySpec(cube_roi_shifted)
        block_spec[gt] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[mst] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[score] = gp.ArraySpec(nonspatial=True)

        additional_request = BatchRequest()
        additional_request[raw] = gp.ArraySpec(input_roi)
        additional_request[candidates] = gp.ArraySpec(cube_roi_shifted)
        additional_request[fg] = gp.ArraySpec(cube_roi_shifted)
        additional_request[gt] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[mst] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[details] = gp.GraphSpec(cube_roi_shifted, directed=False)

        pipeline = (
            (raw_source, gt_source)
            + gp.MergeProvider()
            + Skeletonize(fg, candidates, candidate_spacing, candidate_threshold)
            + MiniMax(fg, candidates, mst, distance_attr=distance_attr)
        )

        pipeline += Evaluate(
            gt,
            mst,
            score,
            roi=cube_roi_shifted,
            details=details,
            edge_threshold_attr=distance_attr,
            num_thresholds=num_thresholds,
            threshold_range=threshold_range,
            small_component_threshold=component_threshold,
        )

        if config["EVAL_SNAPSHOT"]:
            pipeline += gp.Snapshot(
                {
                    raw: f"volumes/raw",
                    fg: f"volumes/foreground",
                    candidates: f"volumes/candidates",
                    mst: f"points/mst",
                    gt: f"points/gt",
                    details: f"points/details",
                },
                output_dir="eval_results",
                output_filename=config["EVAL_SNAPSHOT_NAME"].format(block=block),
                edge_attrs={mst: [distance_attr], details: ["details", "label_pair"]},
                node_attrs={details: ["details", "label_pair"]},
                additional_request=additional_request,
            )

        validation_pipelines.append(pipeline)

    final_score = gp.ArrayKey("SCORE")

    validation_pipeline = (
        tuple(pipeline for pipeline in validation_pipelines)
        + gp.MergeProvider()
        + MergeScores(final_score, specs)
        + gp.PrintProfilingStats()
    )
    return validation_pipeline, final_score


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
    if "EMB_EVAL_MODEL_CONFIG" in config:
        model_config.update(json.load(open(config["EMB_EVAL_MODEL_CONFIG"])))

    model = nl.networks.pytorch.EmbeddingUnet(model_config)

    device = config.get("DEVICE", "cuda")
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    if "EMB_EVAL_MODEL_CHECKPOINT" in config:
        checkpoint_file = config["EMB_EVAL_MODEL_CHECKPOINT"].format(
            setup=config["EMB_EVAL_SETUP"],
            checkpoint=config["EMB_EVAL_CHECKPOINT"],
            emb_net_name=config["EMBEDDING_NET_NAME"],
        )
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
    if config["AUX_TASK"]:
        neighborhood = gp.ArrayKey(f"NEIGHBORHOOD_{block}")
    else:
        neighborhood = None

    if neighborhood is None:
        predict_node = gp.torch.Predict(
            model, inputs={"raw": raw}, outputs={0: emb_pred}
        )
    else:
        predict_node = gp.torch.Predict(
            model, inputs={"raw": raw}, outputs={0: emb_pred, 1: neighborhood}
        )

    pipeline = (
        pipeline
        + nl.gunpowder.nodes.helpers.UnSqueeze(raw)
        + predict_node
        + nl.gunpowder.nodes.helpers.Squeeze(raw)
        + nl.gunpowder.nodes.helpers.Squeeze(emb_pred)
    )
    if neighborhood is not None:
        pipeline += nl.gunpowder.nodes.helpers.Squeeze(neighborhood)

    return pipeline, emb_pred, neighborhood


def get_fg_model(config):
    model_config = copy.deepcopy(DEFAULT_CONFIG)
    model_config.update(json.load(open(config["FG_EVAL_MODEL_CONFIG"])))

    model = nl.networks.pytorch.ForegroundUnet(model_config)

    device = config.get("DEVICE", "cuda")
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    if "FG_EVAL_MODEL_CHECKPOINT" in config:
        checkpoint_file = config["FG_EVAL_MODEL_CHECKPOINT"].format(
            setup=config["FG_EVAL_SETUP"],
            checkpoint=config["FG_EVAL_CHECKPOINT"],
            fg_net_name=config["FOREGROUND_NET_NAME"],
        )

        checkpoint = torch.load(checkpoint_file, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise Exception()

    return model


def add_fg_pred(config, pipeline, raw, block, model):
    device = config.get("DEVICE", "cuda")

    fg_pred = gp.ArrayKey(f"FG_PRED_{block}")

    pipeline = (
        pipeline
        + nl.gunpowder.nodes.helpers.UnSqueeze(raw)
        + gp.torch.Predict(
            model, inputs={"raw": raw}, outputs={0: fg_pred}
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

        raw_source = (
            gp.ZarrSource(
                filename=str(Path(sample_dir, sample, raw_n5).absolute()),
                datasets={raw: "volume-rechunked"},
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size)
                },
            )
            + gp.Normalize(raw, dtype=np.float32)
            + mCLAHE([raw], [20, 64, 64])
        )
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
                edge_attrs={mst: [distance_attr]},
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
