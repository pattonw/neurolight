import gunpowder as gp
from gunpowder.batch_request import BatchRequest
import numpy as np
import neurolight as nl
from neurolight.transforms.swc_to_graph import parse_swc
from neurolight.gunpowder.nodes.snapshot_source import SnapshotSource
from neurolight.pipelines import DEFAULT_CONFIG
from neurolight.pipelines.pipeline_pieces import add_label_processing

from neurolight.gunpowder.nodes.maxima import Skeletonize
from neurolight.gunpowder.nodes.skeletonize_candidates import (
    Skeletonize as Skeletonize_V2,
    SinSample,
)

from neurolight.gunpowder.nodes.minimax import MiniMax
from neurolight.gunpowder.nodes.emst import EMST
from neurolight.gunpowder.nodes.evaluate import Evaluate, MSEEvaluate, MergeGraphs
from neurolight.gunpowder.nodes.clahe import scipyCLAHE
from neurolight.gunpowder.nodes.threshold_edges import ThresholdEdges
from neurolight.gunpowder.nodes.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.nodes.emst_components import ComponentWiseEMST

import torch
from omegaconf import OmegaConf

import copy
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


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
    raw_path,
    gt_path,
    candidates_path=None,
    candidates_mst_path=None,
    candidates_mst_dense_path=None,
    fg_pred_path=None,
):
    checkpoint = config.emb_model.checkpoint
    blocks = config.eval.blocks
    benchmark_datasets_path = Path(config.data.benchmark_data_path)
    sample = config.eval.sample
    transform_template = config.data.transform_template

    voxel_size = gp.Coordinate(config.data.voxel_size)
    micron_scale = max(voxel_size)
    input_shape = gp.Coordinate(config.data.input_shape)
    output_shape = gp.Coordinate(config.data.output_shape)
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    distance_attr = config.eval.distance_attribute
    coordinate_scale = (
        config.um_loss.coordinate_scale * np.array(voxel_size) / micron_scale
    )
    num_thresholds = config.eval.num_thresholds
    threshold_range = config.eval.threshold_range

    candidate_threshold = config.candidates.threshold

    edge_threshold_fg = config.eval.edge_threshold_fg
    component_threshold_fg = config.eval.component_threshold_fg
    component_threshold_emb = config.eval.component_threshold_emb

    clip_limit = config.clahe.clip_limit
    normalize = config.clahe.normalize

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

        raw = gp.ArrayKey(f"RAW_{block}")
        fg_pred = gp.ArrayKey(f"FG_PRED_{block}")

        skeleton = gp.ArrayKey(f"SKELETON_{block}")
        candidates_1 = gp.ArrayKey(f"CANDIDATES_1_{block}")

        fg_mst_full = gp.GraphKey(f"FG_MST_FULL_{block}")
        fg_mst_dense_full = gp.GraphKey(f"FG_MST_DENSE_FULL_{block}")
        fg_mst_thresholded = gp.GraphKey(f"FG_MST_THRESHOLDED_{block}")
        fg_mst_dense_thresholded = gp.GraphKey(f"FG_MST_DENSE_THRESHOLDED_{block}")
        emb_mst = gp.GraphKey(f"EMB_MST_{block}")
        # mst_dense_2 = gp.GraphKey(f"MST_DENSE_2_{block}")
        gt = gp.GraphKey(f"GT_{block}")
        score = gp.ArrayKey(f"SCORE_{block}")
        details = gp.GraphKey(f"DETAILS_{block}")
        optimal_mst = gp.GraphKey(f"OPTIMAL_MST_{block}")

        # Volume Source
        array_datasets = {raw: raw_path.format(block=block)}
        if candidates_path is not None:
            array_datasets[candidates_1] = candidates_path.format(block=block)
        if fg_pred_path is not None:
            array_datasets[fg_pred] = fg_pred_path.format(block=block)
        raw_source = SnapshotSource(snapshot_file, datasets=array_datasets)

        # Graph Source
        graph_datasets = {gt: gt_path.format(block=block)}
        graph_directionality = {gt: False}
        edge_attrs = {}
        if candidates_mst_path is not None and candidates_mst_dense_path is not None:
            raise Exception(candidates_mst_path, candidates_mst_dense_path)
            graph_datasets[fg_mst_full] = candidates_mst_path.format(block=block)
            graph_directionality[fg_mst_full] = False
            edge_attrs[fg_mst_full] = [distance_attr]

            graph_datasets[fg_mst_dense_full] = candidates_mst_dense_path.format(
                block=block
            )
            graph_directionality[fg_mst_dense_full] = False
            edge_attrs[fg_mst_dense_full] = [distance_attr]
        gt_source = SnapshotSource(
            snapshot_file,
            datasets=graph_datasets,
            directed=graph_directionality,
            edge_attrs=edge_attrs,
        )

        if config.eval.clahe.enabled:
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

        reference_sizes = {
            raw: input_size,
            emb: output_size,
            # candidates_1: output_size,
            fg_pred: output_size,
            # skeleton: output_size,
        }
        if neighborhood is not None:
            reference_sizes[neighborhood] = output_size

        emb_source = add_scan(emb_source, reference_sizes)

        if candidates_path is None:
            assert (
                fg_pred_path is not None
            ), "Must provide fg_pred_path if candidates_path is None"
            emb_source += Skeletonize_V2(fg_pred, skeleton, config.candidates.threshold)
            emb_source += SinSample(
                skeleton,
                candidates_1,
                sample_distance=config.candidates.spacing,
                deterministic_sins=True,
            )

        if candidates_mst_path is None or candidates_mst_dense_path is None:
            emb_source += MiniMax(
                intensities=fg_pred,
                mask=candidates_1,
                mst=fg_mst_full,
                dense_mst=fg_mst_dense_full,
                distance_attr="distance",
                threshold=candidate_threshold,
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
        block_spec[candidates_1] = gp.ArraySpec(cube_roi_shifted)
        block_spec[emb] = gp.ArraySpec(cube_roi_shifted)
        if neighborhood is not None:
            block_spec[neighborhood] = gp.ArraySpec(cube_roi_shifted)
        block_spec[gt] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[fg_mst_full] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[fg_mst_dense_full] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[fg_mst_thresholded] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[fg_mst_dense_thresholded] = gp.GraphSpec(
            cube_roi_shifted, directed=False
        )
        block_spec[emb_mst] = gp.GraphSpec(cube_roi_shifted, directed=False)
        # block_spec[mst_dense_2] = gp.GraphSpec(cube_roi_shifted, directed=False)
        block_spec[score] = gp.ArraySpec(nonspatial=True)
        block_spec[optimal_mst] = gp.GraphSpec(cube_roi_shifted, directed=False)

        additional_request = BatchRequest()
        additional_request[raw] = gp.ArraySpec(input_roi)
        additional_request[candidates_1] = gp.ArraySpec(cube_roi_shifted)
        additional_request[emb] = gp.ArraySpec(cube_roi_shifted)
        if fg_pred_path is not None:
            additional_request[fg_pred] = gp.ArraySpec(cube_roi_shifted)
            additional_request[skeleton] = gp.ArraySpec(cube_roi_shifted)
        if neighborhood is not None:
            additional_request[neighborhood] = gp.ArraySpec(cube_roi_shifted)
        additional_request[gt] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[fg_mst_full] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[fg_mst_dense_full] = gp.GraphSpec(
            cube_roi_shifted, directed=False
        )
        additional_request[fg_mst_thresholded] = gp.GraphSpec(
            cube_roi_shifted, directed=False
        )
        additional_request[fg_mst_dense_thresholded] = gp.GraphSpec(
            cube_roi_shifted, directed=False
        )
        additional_request[emb_mst] = gp.GraphSpec(cube_roi_shifted, directed=False)
        # additional_request[mst_dense_2] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[details] = gp.GraphSpec(cube_roi_shifted, directed=False)
        additional_request[optimal_mst] = gp.GraphSpec(cube_roi_shifted, directed=False)

        pipeline = (emb_source, gt_source) + gp.MergeProvider()

        pipeline += ThresholdEdges(
            (fg_mst_full, fg_mst_thresholded),
            edge_threshold_fg,
            component_threshold_fg,
            msts_dense=(fg_mst_dense_full, fg_mst_dense_thresholded),
            distance_attr=distance_attr,
        )

        pipeline += ComponentWiseEMST(
            emb,
            fg_mst_thresholded,
            emb_mst,
            distance_attr=distance_attr,
            coordinate_scale=coordinate_scale,
        )

        # pipeline += ScoreEdges(
        #     mst, mst_dense, emb, distance_attr=distance_attr, path_stat=path_stat
        # )

        pipeline += Evaluate(
            gt,
            emb_mst,
            score,
            roi=cube_roi_shifted,
            details=details,
            edge_threshold_attr=distance_attr,
            num_thresholds=num_thresholds,
            threshold_range=threshold_range,
            small_component_threshold=component_threshold_emb,
            connectivity=fg_mst_thresholded,
            output_graph=optimal_mst,
        )

        if config.eval.snapshot.every > 0:
            snapshot_datasets = {
                raw: f"volumes/raw",
                emb: f"volumes/embeddings",
                fg_pred: f"volumes/fg_pred",
                skeleton: f"volumes/skeleton",
                candidates_1: f"volumes/candidates_1",
                fg_mst_full: f"points/fg_mst_full",
                fg_mst_dense_full: f"points/fg_mst_dense_full",
                fg_mst_thresholded: f"points/fg_mst_thresholded",
                fg_mst_dense_thresholded: f"points/fg_mst_dense_thresholded",
                emb_mst: f"points/emb_mst",
                gt: f"points/gt",
                details: f"points/details",
                optimal_mst: f"points/optimal_mst",
            }
            if neighborhood is not None:
                snapshot_datasets[neighborhood] = f"volumes/neighborhood"
            pipeline += gp.Snapshot(
                snapshot_datasets,
                output_dir=config.eval.snapshot.directory,
                output_filename=config.eval.snapshot.file_name.format(
                    checkpoint=checkpoint, block=block
                ),
                edge_attrs={
                    fg_mst_full: [distance_attr],
                    fg_mst_dense_full: [distance_attr],
                    fg_mst_thresholded: [distance_attr],
                    fg_mst_dense_thresholded: [distance_attr],
                    emb_mst: [distance_attr],
                    # optimal_mst: [distance_attr],
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


def fg_validation_pipeline(config, snapshot_file, raw_path, gt_path, mse_diff=False):
    checkpoint = config.fg_model.checkpoint
    blocks = config.eval.blocks
    tile_scan = config.eval.tile_scan
    benchmark_datasets_path = Path(config.data.benchmark_data_path)
    sample = config.eval.sample
    transform_template = config.data.transform_template

    voxel_size = gp.Coordinate(config.data.voxel_size)
    input_shape = gp.Coordinate(config.data.input_shape)
    output_shape = gp.Coordinate(config.data.output_shape)
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    candidate_spacing = config.candidates.spacing
    candidate_threshold = config.candidates.threshold

    distance_attr = config.eval.distance_attribute
    num_thresholds = config.eval.num_thresholds
    threshold_range = config.eval.threshold_range

    component_threshold = config.eval.component_threshold_fg

    clip_limit = config.clahe.clip_limit
    normalize = config.clahe.normalize
    kernel_size = gp.Coordinate(config.clahe.kernel_size)

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

        raw = gp.ArrayKey(f"RAW_{block}")
        gt = gp.GraphKey(f"GT_{block}")
        score = gp.ArrayKey(f"SCORE_{block}")

        raw_source = SnapshotSource(
            snapshot_file, datasets={raw: raw_path.format(block=block)}
        )
        if tile_scan:
            raw_source = raw_source + gp.Pad(raw, None)
        gt_source = SnapshotSource(
            snapshot_file,
            datasets={gt: gt_path.format(block=block)},
            directed={gt: False},
        )

        if config.clahe.enabled:
            logger.warning(
                f"Applying clahe with clip_limit: {clip_limit} "
                f"and normalization: {normalize}!"
            )
            raw_source = raw_source + scipyCLAHE(
                [raw],
                kernel_size=kernel_size * voxel_size,
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
        output_roi = gp.Roi((0,) * len(cube_roi.get_shape()), cube_roi.get_shape())
        input_roi = output_roi.grow(
            (input_size - output_size) // 2, (input_size - output_size) // 2
        )

        if tile_scan and False:
            # make sure the output roi is a multiple of the output size for scan:
            output_tiles = (
                output_roi.get_shape()
                + output_shape
                - gp.Coordinate((1,) * output_roi.dims())
            ) / output_shape
            tiled_output_size = output_tiles * output_shape
            size_diff = tiled_output_size - output_roi.get_shape()
            halved = size_diff / 2

            tiled_output_roi = output_roi.grow(halved, size_diff - halved)
            tiled_input_roi = input_roi.grow(halved, size_diff - halved)

            input_roi = tiled_input_roi
            output_roi = tiled_output_roi
        else:
            logger.warning(
                "Should change the shape of the roi to avoid " "scan border artifacts"
            )

        block_spec = specs.setdefault(block, {})
        block_spec[score] = gp.ArraySpec(nonspatial=True)

        additional_request = BatchRequest()
        additional_request[raw] = gp.ArraySpec(input_roi)
        additional_request[fg] = gp.ArraySpec(output_roi)
        additional_request[gt] = gp.GraphSpec(output_roi, directed=False)

        snapshot_datasets = {
            raw: f"volumes/raw",
            fg: f"volumes/foreground",
            gt: f"points/gt",
        }
        edge_attrs = {}
        node_attrs = {}

        if not mse_diff:
            candidates = gp.ArrayKey(f"CANDIDATES_{block}")
            mst = gp.GraphKey(f"MST_{block}")
            mst_dense = gp.GraphKey(f"MST_DENSE_{block}")
            details = gp.GraphKey(f"DETAILS_{block}")

            snapshot_datasets.update(
                {
                    candidates: f"volumes/candidates",
                    mst: f"points/mst",
                    mst_dense: f"points/mst_dense",
                    details: f"points/details",
                }
            )
            edge_attrs.update(
                {mst: [distance_attr], details: ["details", "label_pair"]}
            )
            node_attrs.update({details: ["details", "label_pair"]})

            additional_request[candidates] = gp.ArraySpec(output_roi)
            additional_request[mst] = gp.GraphSpec(output_roi, directed=False)
            additional_request[details] = gp.GraphSpec(output_roi, directed=False)

            pipeline = (
                (fg_source, gt_source)
                + gp.MergeProvider()
                + Skeletonize(fg, candidates, candidate_spacing, candidate_threshold)
                + MiniMax(
                    intensities=fg,
                    mask=candidates,
                    mst=mst,
                    dense_mst=mst_dense,
                    distance_attr=distance_attr,
                    threshold=candidate_threshold,
                )
            )

            pipeline += Evaluate(
                gt,
                mst,
                score,
                roi=output_roi,
                details=details,
                edge_threshold_attr=distance_attr,
                num_thresholds=num_thresholds,
                threshold_range=threshold_range,
                small_component_threshold=component_threshold,
            )

        else:
            labels = gp.ArrayKey(f"LABELS_{block}")

            gt_source = gt_source + RasterizeSkeleton(
                points=gt,
                array=labels,
                array_spec=gp.ArraySpec(
                    interpolatable=False, voxel_size=voxel_size, dtype=np.int64
                ),
            )
            gt_source, gt_fg, loss_weights = add_label_processing(
                gt_source, config, labels, f"_{block}"
            )

            additional_request[labels] = gp.ArraySpec(output_roi)
            additional_request[gt_fg] = gp.ArraySpec(output_roi)
            additional_request[loss_weights] = gp.ArraySpec(output_roi)

            snapshot_datasets.update(
                {
                    labels: f"volumes/labels",
                    gt_fg: f"volumes/gt_fg",
                    loss_weights: f"volumes/loss_weights",
                }
            )

            pipeline = (fg_source, gt_source) + gp.MergeProvider()

            pipeline += MSEEvaluate(gt_fg, fg, score, output_roi, weights=loss_weights)

        if config.eval.snapshot.every > 0:
            pipeline += gp.Snapshot(
                dataset_names=snapshot_datasets,
                output_dir=config.eval.snapshot.directory,
                output_filename=config.eval.snapshot.file_name.format(
                    checkpoint=checkpoint, block=block
                ),
                edge_attrs=edge_attrs,
                node_attrs=node_attrs,
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
    blocks = config.eval.blocks
    benchmark_datasets_path = Path(config.data.benchmark_data_path)
    sample = config.eval.sample
    transform_template = config.data.transform_template

    voxel_size = gp.Coordinate(config.data.voxel_size)
    input_shape = gp.Coordinate(config.data.input_shape)
    output_shape = gp.Coordinate(config.data.output_shape)
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    candidate_spacing = config.candidates.spacing
    candidate_threshold = config.candidates.threshold

    distance_attr = config.eval.distance_attribute
    num_thresholds = config.eval.num_thresholds
    threshold_range = config.eval.threshold_range

    component_threshold = config.eval.component_threshold_fg

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

        if config.eval.snapshot.every > 0:
            pipeline += gp.Snapshot(
                {
                    raw: f"volumes/raw",
                    fg: f"volumes/foreground",
                    candidates: f"volumes/candidates",
                    mst: f"points/mst",
                    gt: f"points/gt",
                    details: f"points/details",
                },
                output_dir=config.eval.snapshot.directory,
                output_filename=config.eval.snapshot.file_name.format(block=block),
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
    # default model config
    model_config = copy.deepcopy(DEFAULT_CONFIG)

    # update with setup model config
    setup = config.emb_model.setup
    model_config_file = Path(config.emb_model.directory, setup, "config.yaml")
    model_config = OmegaConf.merge(
        model_config, OmegaConf.load(model_config_file.open())
    )

    model = nl.networks.pytorch.EmbeddingUnet(model_config)

    device = config.eval.device
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint_file = Path(
        config.emb_model.directory,
        setup,
        f"{config.emb_model.net_name}_checkpoint_{config.emb_model.checkpoint}",
    )

    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Checkpoint {checkpoint_file} does not exist!")

    return model


def add_emb_pred(config, pipeline, raw, block, model):

    emb_pred = gp.ArrayKey(f"EMB_PRED_{block}")
    if config.emb_model.aux_task.enabled:
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


def get_fg_model(config, ret_checkpoint=False):
    model_config = copy.deepcopy(DEFAULT_CONFIG)

    setup = config.fg_model.setup
    model_config_file = Path(config.fg_model.directory, setup, "config.yaml")
    model_config.update(
        OmegaConf.merge(model_config, OmegaConf.load(model_config_file.open()))
    )

    model = nl.networks.pytorch.ForegroundUnet(model_config)

    device = config.eval.device
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint_file = Path(
        config.fg_model.directory,
        setup,
        f"{config.fg_model.net_name}_checkpoint_{config.fg_model.checkpoint}",
    )

    if checkpoint_file.exists() and not ret_checkpoint:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    elif ret_checkpoint:
        return model, checkpoint_file
    else:
        raise ValueError(f"Checkpoint {checkpoint_file} does not exist!")

    return model


def add_fg_pred(config, pipeline, raw, block, model):
    fg_pred = gp.ArrayKey(f"FG_PRED_{block}")

    pipeline = (
        pipeline
        + nl.gunpowder.nodes.helpers.UnSqueeze(raw)
        + gp.torch.Predict(model, inputs={"raw": raw}, outputs={0: fg_pred})
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
    raise NotImplementedError(
        "This seems redundant, but not sure if I can delete it yet"
    )
    blocks = config["BLOCKS"]
    benchmark_datasets_path = Path(config.data.benchmark_data_path)
    sample = config.eval.sample
    sample_dir = Path(config["SAMPLES_PATH"])
    raw_n5 = config["RAW_N5"]
    transform_template = config.data.transform_template

    neuron_width = int(config["NEURON_RADIUS"])
    voxel_size = gp.Coordinate(config.data.voxel_size)
    micron_scale = max(voxel_size)
    input_shape = gp.Coordinate(config.data.input_shape)
    output_shape = gp.Coordinate(config.data.output_shape)
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    distance_attr = config.eval.distance_attribute
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

        input_roi = cube_roi.grow(
            (input_size - output_size) // 2, (input_size - output_size) // 2
        )
        output_roi = cube_roi

        additional_request = BatchRequest()
        block_spec = specs.setdefault(block, {})
        block_spec["raw"] = (raw, gp.ArraySpec(input_roi))
        additional_request[raw] = gp.ArraySpec(roi=input_roi)
        block_spec["ground_truth"] = (ground_truth, gp.GraphSpec(output_roi))
        additional_request[ground_truth] = gp.GraphSpec(roi=output_roi)
        block_spec["labels"] = (labels, gp.ArraySpec(output_roi))
        additional_request[labels] = gp.ArraySpec(roi=output_roi)
        block_spec["fg_pred"] = (fg, gp.ArraySpec(output_roi))
        additional_request[fg] = gp.ArraySpec(roi=output_roi)
        block_spec["emb_pred"] = (emb, gp.ArraySpec(output_roi))
        additional_request[emb] = gp.ArraySpec(roi=output_roi)
        block_spec["candidates"] = (candidates, gp.ArraySpec(output_roi))
        additional_request[candidates] = gp.ArraySpec(roi=output_roi)
        block_spec["mst_pred"] = (mst, gp.GraphSpec(output_roi))
        additional_request[mst] = gp.GraphSpec(roi=output_roi)

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
