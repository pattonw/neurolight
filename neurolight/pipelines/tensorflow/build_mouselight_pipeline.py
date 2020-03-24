from neurolight.tensorflow.add_loss import create_custom_loss
from neurolight.gunpowder.nodes import (
    TopologicalMatcher,
    RasterizeSkeleton,
    GetNeuronPair,
    FusionAugment,
    GrowLabels,
)
import gunpowder as gp
import numpy as np

import math
from pathlib import Path
import copy
import logging

logger = logging.getLogger(__file__)


class BinarizeGt(gp.BatchFilter):
    def __init__(self, gt, gt_binary):

        self.gt = gt
        self.gt_binary = gt_binary

    def setup(self):

        spec = self.spec[self.gt].copy()
        spec.dtype = np.uint8
        self.provides(self.gt_binary, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        spec.dtype = np.int32

        binarized = gp.Array(data=(batch[self.gt].data > 0).astype(np.int32), spec=spec)

        batch[self.gt_binary] = binarized


class Crop(gp.BatchFilter):
    def __init__(self, input_array: gp.ArrayKey, output_array: gp.ArrayKey):

        self.input_array = input_array
        self.output_array = output_array

    def setup(self):

        spec = self.spec[self.input_array].copy()
        self.provides(self.output_array, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):
        input_data = batch[self.input_array].data
        input_spec = batch[self.input_array].spec
        input_roi = input_spec.roi
        output_roi_shape = request[self.output_array].roi.get_shape()
        shift = (input_roi.get_shape() - output_roi_shape) / 2
        output_roi = gp.Roi(shift, output_roi_shape)
        output_data = input_data[
            tuple(
                map(
                    slice,
                    output_roi.get_begin() / input_spec.voxel_size,
                    output_roi.get_end() / input_spec.voxel_size,
                )
            )
        ]
        output_spec = copy.deepcopy(input_spec)
        output_spec.roi = output_roi

        output_array = gp.Array(output_data, output_spec)

        batch[self.output_array] = output_array


class RejectIfEmpty(gp.BatchFilter):
    def __init__(self, ensure_nonempty: gp.ArrayKey, request_limit: int = 100):
        self.ensure_nonempty = ensure_nonempty
        self.request_limit = request_limit

    def setup(self):
        self.enable_autoskip()
        self.updates(
            self.ensure_nonempty, copy.deepcopy(self.spec[self.ensure_nonempty])
        )

    def prepare(self, request):
        return copy.deepcopy(request)

    def process(self, batch, request):

        k = 0
        found_valid = False
        while (
            not found_valid
            and k < self.request_limit
            and self.ensure_nonempty in request
        ):
            batch = self.get_upstream_provider().request_batch(request)
            if len(batch[self.ensure_nonempty].graph.nodes) > 0:
                found_valid = True
            k += 1

        return batch


def train_fusion_pipeline(
    n_iterations, setup_config, mknet_tensor_names, loss_tensor_names
):
    input_shape = gp.Coordinate(setup_config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(setup_config["OUTPUT_SHAPE"])
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    shift_attempts = setup_config["SHIFT_ATTEMPTS"]
    request_attempts = setup_config["REQUEST_ATTEMPTS"]
    num_iterations = setup_config["NUM_ITERATIONS"]
    cache_size = setup_config["CACHE_SIZE"]
    num_workers = setup_config["NUM_WORKERS"]
    snapshot_every = setup_config["SNAPSHOT_EVERY"]
    checkpoint_every = setup_config["CHECKPOINT_EVERY"]
    profile_every = setup_config["PROFILE_EVERY"]
    seperate_by = setup_config["SEPERATE_BY"]
    gap_crossing_dist = setup_config["GAP_CROSSING_DIST"]
    match_distance_threshold = setup_config["MATCH_DISTANCE_THRESHOLD"]
    point_balance_radius = setup_config["POINT_BALANCE_RADIUS"]
    neuron_radius = setup_config["NEURON_RADIUS"]
    blend_smoothness = setup_config["BLEND_SMOOTHNESS"]

    samples_path = Path(setup_config["SAMPLES_PATH"])
    mongo_url = setup_config["MONGO_URL"]

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    # voxels have size ~= 1 micron on z axis
    # use this value to scale anything that depends on world unit distance
    micron_scale = voxel_size[0]
    seperate_distance = (np.array(seperate_by)).tolist()

    # array keys for data sources
    raw = gp.ArrayKey("RAW")
    consensus = gp.PointsKey("CONSENSUS")
    skeletonization = gp.PointsKey("SKELETONIZATION")
    matched = gp.PointsKey("MATCHED")
    nonempty_placeholder = gp.PointsKey("NONEMPTY")
    labels = gp.ArrayKey("LABELS")

    # array keys for base volume
    raw_base = gp.ArrayKey("RAW_BASE")
    labels_base = gp.ArrayKey("LABELS_BASE")
    matched_base = gp.PointsKey("MATCHED_BASE")
    skeletonization_base = gp.PointsKey("SKELETONIZATION_BASE")
    consensus_base = gp.PointsKey("CONSENSUS_BASE")

    # array keys for add volume
    raw_add = gp.ArrayKey("RAW_ADD")
    labels_add = gp.ArrayKey("LABELS_ADD")
    matched_add = gp.PointsKey("MATCHED_ADD")
    skeletonization_add = gp.PointsKey("SKELETONIZATION_ADD")
    consensus_add = gp.PointsKey("CONSENSUS_ADD")

    # array keys for fused volume
    raw_fused = gp.ArrayKey("RAW_FUSED")
    labels_fused = gp.ArrayKey("LABELS_FUSED")
    matched_fused = gp.PointsKey("MATCHED_FUSED")
    labels_fg = gp.ArrayKey("LABELS_FG")
    labels_fg_bin = gp.ArrayKey("LABELS_FG_BIN")
    loss_weights = gp.ArrayKey("LOSS_WEIGHTS")

    # debug array keys
    soft_mask = gp.ArrayKey("SOFT_MASK")
    masked_base = gp.ArrayKey("MASKED_BASE")
    masked_add = gp.ArrayKey("MASKED_ADD")
    mask_maxed = gp.ArrayKey("MASK_MAXED")

    # tensorflow tensors
    gt_fg = gp.ArrayKey("GT_FG")
    fg_pred = gp.ArrayKey("FG_PRED")
    embedding = gp.ArrayKey("EMBEDDING")
    fg = gp.ArrayKey("FG")
    maxima = gp.ArrayKey("MAXIMA")
    gradient_embedding = gp.ArrayKey("GRADIENT_EMBEDDING")
    gradient_fg = gp.ArrayKey("GRADIENT_FG")
    emst = gp.ArrayKey("EMST")
    edges_u = gp.ArrayKey("EDGES_U")
    edges_v = gp.ArrayKey("EDGES_V")
    ratio_pos = gp.ArrayKey("RATIO_POS")
    ratio_neg = gp.ArrayKey("RATIO_NEG")
    dist = gp.ArrayKey("DIST")
    num_pos_pairs = gp.ArrayKey("NUM_POS")
    num_neg_pairs = gp.ArrayKey("NUM_NEG")

    # add request
    request = gp.BatchRequest()
    request.add(raw_fused, input_size)
    request.add(labels_fused, input_size)
    request.add(matched_fused, input_size)
    request.add(labels_fg, output_size)
    request.add(labels_fg_bin, output_size)
    request.add(loss_weights, output_size)

    # add snapshot request
    snapshot_request = gp.BatchRequest()
    request.add(labels_fg, output_size)
    request.add(raw_base, input_size)
    request.add(raw_add, input_size)
    request.add(labels_base, input_size)
    request.add(labels_add, input_size)
    request.add(matched_base, input_size)
    request.add(matched_add, input_size)
    request.add(skeletonization_base, input_size)
    request.add(skeletonization_add, input_size)
    request.add(consensus_base, input_size)
    request.add(consensus_add, input_size)

    # add debug requests
    snapshot_request.add(soft_mask, input_size, voxel_size=voxel_size)
    snapshot_request.add(masked_base, input_size, voxel_size=voxel_size)
    snapshot_request.add(masked_add, input_size, voxel_size=voxel_size)
    snapshot_request.add(mask_maxed, input_size, voxel_size=voxel_size)

    # tensorflow requests
    snapshot_request.add(raw_fused, input_size)  # input_size request for positioning
    snapshot_request.add(embedding, output_size, voxel_size=voxel_size)
    snapshot_request.add(fg, output_size, voxel_size=voxel_size)
    snapshot_request.add(gt_fg, output_size, voxel_size=voxel_size)
    snapshot_request.add(fg_pred, output_size, voxel_size=voxel_size)
    snapshot_request.add(maxima, output_size, voxel_size=voxel_size)
    snapshot_request.add(gradient_embedding, output_size, voxel_size=voxel_size)
    snapshot_request.add(gradient_fg, output_size, voxel_size=voxel_size)
    snapshot_request[emst] = gp.ArraySpec()
    snapshot_request[edges_u] = gp.ArraySpec()
    snapshot_request[edges_v] = gp.ArraySpec()
    snapshot_request[ratio_pos] = gp.ArraySpec()
    snapshot_request[ratio_neg] = gp.ArraySpec()
    snapshot_request[dist] = gp.ArraySpec()
    snapshot_request[num_pos_pairs] = gp.ArraySpec()
    snapshot_request[num_neg_pairs] = gp.ArraySpec()

    data_sources = tuple(
        (
            gp.N5Source(
                filename=str((sample / "fluorescence-near-consensus.n5").absolute()),
                datasets={raw: "volume"},
                array_specs={
                    raw: gp.ArraySpec(
                        interpolatable=True, voxel_size=voxel_size, dtype=np.uint16
                    )
                },
            ),
            gp.DaisyGraphProvider(
                f"mouselight-{sample.name}-consensus",
                mongo_url,
                points=[consensus, nonempty_placeholder],
                directed=True,
                node_attrs=[],
                edge_attrs=[],
            ),
            gp.DaisyGraphProvider(
                f"mouselight-{sample.name}-skeletonization",
                mongo_url,
                points=[skeletonization],
                directed=False,
                node_attrs=[],
                edge_attrs=[],
            ),
        )
        + gp.MergeProvider()
        + gp.RandomLocation(
            ensure_nonempty=nonempty_placeholder,
            ensure_centered=True,
            point_balance_radius=point_balance_radius * micron_scale,
        )
        + TopologicalMatcher(
            skeletonization,
            consensus,
            matched,
            failures=Path("matching_failures_slow"),
            match_distance_threshold=match_distance_threshold * micron_scale,
            max_gap_crossing=gap_crossing_dist * micron_scale,
            try_complete=False,
            use_gurobi=True,
        )
        + RasterizeSkeleton(
            points=matched,
            array=labels,
            array_spec=gp.ArraySpec(
                interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
            ),
        )
        + GrowLabels(labels, radii=[neuron_radius * micron_scale])
        # TODO: Do these need to be scaled by world units?
        + gp.ElasticAugment(
            [40, 10, 10],
            [0.25, 1, 1],
            [0, math.pi / 2.0],
            subsample=4,
            use_fast_points_transform=True,
            recompute_missing_points=False,
        )
        # + gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2])
        + gp.Normalize(raw) + gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001)
        for sample in samples_path.iterdir()
        if sample.name in ("2018-07-02", "2018-08-01")
    )

    pipeline = (
        data_sources
        + gp.RandomProvider()
        + GetNeuronPair(
            matched,
            raw,
            labels,
            (matched_base, matched_add),
            (raw_base, raw_add),
            (labels_base, labels_add),
            output_shape=output_size,
            seperate_by=seperate_distance,
            shift_attempts=shift_attempts,
            request_attempts=request_attempts,
            nonempty_placeholder=nonempty_placeholder,
            extra_keys={
                skeletonization: (skeletonization_base, skeletonization_add),
                consensus: (consensus_base, consensus_add),
            },
        )
        + FusionAugment(
            raw_base,
            raw_add,
            labels_base,
            labels_add,
            matched_base,
            matched_add,
            raw_fused,
            labels_fused,
            matched_fused,
            soft_mask=soft_mask,
            masked_base=masked_base,
            masked_add=masked_add,
            mask_maxed=mask_maxed,
            blend_mode="labels_mask",
            blend_smoothness=blend_smoothness * micron_scale,
            gaussian_smooth_mode="mirror",
            num_blended_objects=0,
        )
        + Crop(labels_fused, labels_fg)
        + BinarizeGt(labels_fg, labels_fg_bin)
        + gp.BalanceLabels(labels_fg_bin, loss_weights)
        + gp.PreCache(cache_size=cache_size, num_workers=num_workers)
        + gp.tensorflow.Train(
            "train_net",
            optimizer=create_custom_loss(mknet_tensor_names, setup_config),
            loss=None,
            inputs={
                mknet_tensor_names["loss_weights"]: loss_weights,
                mknet_tensor_names["raw"]: raw_fused,
                mknet_tensor_names["gt_labels"]: labels_fg,
            },
            outputs={
                mknet_tensor_names["embedding"]: embedding,
                mknet_tensor_names["fg"]: fg,
                loss_tensor_names["fg_pred"]: fg_pred,
                loss_tensor_names["maxima"]: maxima,
                loss_tensor_names["gt_fg"]: gt_fg,
                loss_tensor_names["emst"]: emst,
                loss_tensor_names["edges_u"]: edges_u,
                loss_tensor_names["edges_v"]: edges_v,
                loss_tensor_names["ratio_pos"]: ratio_pos,
                loss_tensor_names["ratio_neg"]: ratio_neg,
                loss_tensor_names["dist"]: dist,
                loss_tensor_names["num_pos_pairs"]: num_pos_pairs,
                loss_tensor_names["num_neg_pairs"]: num_neg_pairs,
            },
            gradients={
                mknet_tensor_names["embedding"]: gradient_embedding,
                mknet_tensor_names["fg"]: gradient_fg,
            },
            save_every=checkpoint_every,
            summary="Merge/MergeSummary:0",
            log_dir="tensorflow_logs",
        )
        + gp.PrintProfilingStats(every=profile_every)
        + gp.Snapshot(
            additional_request=snapshot_request,
            output_filename="snapshot_{}_{}.hdf".format(
                int(np.min(seperate_distance)), "{id}"
            ),
            dataset_names={
                # raw data
                raw_fused: "volumes/raw_fused",
                raw_base: "volumes/raw_base",
                raw_add: "volumes/raw_add",
                # labeled data
                labels_fused: "volumes/labels_fused",
                labels_base: "volumes/labels_base",
                labels_add: "volumes/labels_add",
                # trees
                skeletonization_base: "points/skeletonization_base",
                skeletonization_add: "points/skeletonization_add",
                consensus_base: "points/consensus_base",
                consensus_add: "points/consensus_add",
                matched_base: "points/matched_base",
                matched_add: "points/matched_add",
                matched_fused: "points/matched_fused",
                # output volumes
                embedding: "volumes/embedding",
                fg: "volumes/fg",
                maxima: "volumes/maxima",
                gt_fg: "volumes/gt_fg",
                fg_pred: "volumes/fg_pred",
                gradient_embedding: "volumes/gradient_embedding",
                gradient_fg: "volumes/gradient_fg",
                # output trees
                emst: "emst",
                edges_u: "edges_u",
                edges_v: "edges_v",
                # output debug data
                ratio_pos: "ratio_pos",
                ratio_neg: "ratio_neg",
                dist: "dist",
                num_pos_pairs: "num_pos_pairs",
                num_neg_pairs: "num_neg_pairs",
                # debug volumes
                soft_mask: "volumes/soft_mask",
                masked_base: "volumes/masked_base",
                masked_add: "volumes/masked_add",
                mask_maxed: "volumes/mask_maxed",
                loss_weights: "volumes/loss_weights",
            },
            every=snapshot_every,
        )
    )

    with gp.build(pipeline):
        for _ in range(num_iterations):
            pipeline.request_batch(request)


def train_simple_pipeline(
    n_iterations, setup_config, mknet_tensor_names, loss_tensor_names
):
    input_shape = gp.Coordinate(setup_config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(setup_config["OUTPUT_SHAPE"])
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    num_iterations = setup_config["NUM_ITERATIONS"]
    cache_size = setup_config["CACHE_SIZE"]
    num_workers = setup_config["NUM_WORKERS"]
    snapshot_every = setup_config["SNAPSHOT_EVERY"]
    checkpoint_every = setup_config["CHECKPOINT_EVERY"]
    profile_every = setup_config["PROFILE_EVERY"]
    seperate_by = setup_config["SEPERATE_BY"]
    gap_crossing_dist = setup_config["GAP_CROSSING_DIST"]
    match_distance_threshold = setup_config["MATCH_DISTANCE_THRESHOLD"]
    point_balance_radius = setup_config["POINT_BALANCE_RADIUS"]
    neuron_radius = setup_config["NEURON_RADIUS"]

    samples_path = Path(setup_config["SAMPLES_PATH"])
    mongo_url = setup_config["MONGO_URL"]

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    # voxels have size ~= 1 micron on z axis
    # use this value to scale anything that depends on world unit distance
    micron_scale = voxel_size[0]
    seperate_distance = (np.array(seperate_by)).tolist()

    # array keys for data sources
    raw = gp.ArrayKey("RAW")
    consensus = gp.PointsKey("CONSENSUS")
    skeletonization = gp.PointsKey("SKELETONIZATION")
    matched = gp.PointsKey("MATCHED")
    labels = gp.ArrayKey("LABELS")

    labels_fg = gp.ArrayKey("LABELS_FG")
    labels_fg_bin = gp.ArrayKey("LABELS_FG_BIN")
    loss_weights = gp.ArrayKey("LOSS_WEIGHTS")

    # tensorflow tensors
    gt_fg = gp.ArrayKey("GT_FG")
    fg_pred = gp.ArrayKey("FG_PRED")
    embedding = gp.ArrayKey("EMBEDDING")
    fg = gp.ArrayKey("FG")
    maxima = gp.ArrayKey("MAXIMA")
    gradient_embedding = gp.ArrayKey("GRADIENT_EMBEDDING")
    gradient_fg = gp.ArrayKey("GRADIENT_FG")
    emst = gp.ArrayKey("EMST")
    edges_u = gp.ArrayKey("EDGES_U")
    edges_v = gp.ArrayKey("EDGES_V")
    ratio_pos = gp.ArrayKey("RATIO_POS")
    ratio_neg = gp.ArrayKey("RATIO_NEG")
    dist = gp.ArrayKey("DIST")
    num_pos_pairs = gp.ArrayKey("NUM_POS")
    num_neg_pairs = gp.ArrayKey("NUM_NEG")

    # add request
    request = gp.BatchRequest()
    request.add(labels_fg, output_size)
    request.add(labels_fg_bin, output_size)
    request.add(loss_weights, output_size)
    request.add(raw, input_size)
    request.add(labels, input_size)
    request.add(matched, input_size)
    request.add(skeletonization, input_size)
    request.add(consensus, input_size)

    # add snapshot request
    snapshot_request = gp.BatchRequest()
    request.add(labels_fg, output_size)

    # tensorflow requests
    snapshot_request.add(raw, input_size)  # input_size request for positioning
    snapshot_request.add(embedding, output_size, voxel_size=voxel_size)
    snapshot_request.add(fg, output_size, voxel_size=voxel_size)
    snapshot_request.add(gt_fg, output_size, voxel_size=voxel_size)
    snapshot_request.add(fg_pred, output_size, voxel_size=voxel_size)
    snapshot_request.add(maxima, output_size, voxel_size=voxel_size)
    snapshot_request.add(gradient_embedding, output_size, voxel_size=voxel_size)
    snapshot_request.add(gradient_fg, output_size, voxel_size=voxel_size)
    snapshot_request[emst] = gp.ArraySpec()
    snapshot_request[edges_u] = gp.ArraySpec()
    snapshot_request[edges_v] = gp.ArraySpec()
    snapshot_request[ratio_pos] = gp.ArraySpec()
    snapshot_request[ratio_neg] = gp.ArraySpec()
    snapshot_request[dist] = gp.ArraySpec()
    snapshot_request[num_pos_pairs] = gp.ArraySpec()
    snapshot_request[num_neg_pairs] = gp.ArraySpec()

    data_sources = tuple(
        (
            gp.N5Source(
                filename=str((sample / "fluorescence-near-consensus.n5").absolute()),
                datasets={raw: "volume"},
                array_specs={
                    raw: gp.ArraySpec(
                        interpolatable=True, voxel_size=voxel_size, dtype=np.uint16
                    )
                },
            ),
            gp.DaisyGraphProvider(
                f"mouselight-{sample.name}-consensus",
                mongo_url,
                points=[consensus],
                directed=True,
                node_attrs=[],
                edge_attrs=[],
            ),
            gp.DaisyGraphProvider(
                f"mouselight-{sample.name}-skeletonization",
                mongo_url,
                points=[skeletonization],
                directed=False,
                node_attrs=[],
                edge_attrs=[],
            ),
        )
        + gp.MergeProvider()
        + gp.RandomLocation(
            ensure_nonempty=consensus,
            ensure_centered=True,
            point_balance_radius=point_balance_radius * micron_scale,
        )
        + TopologicalMatcher(
            skeletonization,
            consensus,
            matched,
            failures=Path("matching_failures_slow"),
            match_distance_threshold=match_distance_threshold * micron_scale,
            max_gap_crossing=gap_crossing_dist * micron_scale,
            try_complete=False,
            use_gurobi=True,
        )
        + RejectIfEmpty(matched)
        + RasterizeSkeleton(
            points=matched,
            array=labels,
            array_spec=gp.ArraySpec(
                interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
            ),
        )
        + GrowLabels(labels, radii=[neuron_radius * micron_scale])
        # TODO: Do these need to be scaled by world units?
        + gp.ElasticAugment(
            [40, 10, 10],
            [0.25, 1, 1],
            [0, math.pi / 2.0],
            subsample=4,
            use_fast_points_transform=True,
            recompute_missing_points=False,
        )
        # + gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2])
        + gp.Normalize(raw) + gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001)
        for sample in samples_path.iterdir()
        if sample.name in ("2018-07-02", "2018-08-01")
    )

    pipeline = (
        data_sources
        + gp.RandomProvider()
        + Crop(labels, labels_fg)
        + BinarizeGt(labels_fg, labels_fg_bin)
        + gp.BalanceLabels(labels_fg_bin, loss_weights)
        + gp.PreCache(cache_size=cache_size, num_workers=num_workers)
        + gp.tensorflow.Train(
            "train_net",
            optimizer=create_custom_loss(mknet_tensor_names, setup_config),
            loss=None,
            inputs={
                mknet_tensor_names["loss_weights"]: loss_weights,
                mknet_tensor_names["raw"]: raw,
                mknet_tensor_names["gt_labels"]: labels_fg,
            },
            outputs={
                mknet_tensor_names["embedding"]: embedding,
                mknet_tensor_names["fg"]: fg,
                loss_tensor_names["fg_pred"]: fg_pred,
                loss_tensor_names["maxima"]: maxima,
                loss_tensor_names["gt_fg"]: gt_fg,
                loss_tensor_names["emst"]: emst,
                loss_tensor_names["edges_u"]: edges_u,
                loss_tensor_names["edges_v"]: edges_v,
                loss_tensor_names["ratio_pos"]: ratio_pos,
                loss_tensor_names["ratio_neg"]: ratio_neg,
                loss_tensor_names["dist"]: dist,
                loss_tensor_names["num_pos_pairs"]: num_pos_pairs,
                loss_tensor_names["num_neg_pairs"]: num_neg_pairs,
            },
            gradients={
                mknet_tensor_names["embedding"]: gradient_embedding,
                mknet_tensor_names["fg"]: gradient_fg,
            },
            save_every=checkpoint_every,
            summary="Merge/MergeSummary:0",
            log_dir="tensorflow_logs",
        )
        + gp.PrintProfilingStats(every=profile_every)
        + gp.Snapshot(
            additional_request=snapshot_request,
            output_filename="snapshot_{}_{}.hdf".format(
                int(np.min(seperate_distance)), "{id}"
            ),
            dataset_names={
                # raw data
                raw: "volumes/raw",
                # labeled data
                labels: "volumes/labels",
                # trees
                skeletonization: "points/skeletonization",
                consensus: "points/consensus",
                matched: "points/matched",
                # output volumes
                embedding: "volumes/embedding",
                fg: "volumes/fg",
                maxima: "volumes/maxima",
                gt_fg: "volumes/gt_fg",
                fg_pred: "volumes/fg_pred",
                gradient_embedding: "volumes/gradient_embedding",
                gradient_fg: "volumes/gradient_fg",
                # output trees
                emst: "emst",
                edges_u: "edges_u",
                edges_v: "edges_v",
                # output debug data
                ratio_pos: "ratio_pos",
                ratio_neg: "ratio_neg",
                dist: "dist",
                num_pos_pairs: "num_pos_pairs",
                num_neg_pairs: "num_neg_pairs",
                loss_weights: "volumes/loss_weights",
            },
            every=snapshot_every,
        )
    )

    with gp.build(pipeline):
        for _ in range(num_iterations):
            pipeline.request_batch(request)


def train(n_iterations, setup_config, mknet_tensor_names, loss_tensor_names):
    if setup_config["FUSION_PIPELINE"]:
        train_fusion_pipeline(
            n_iterations, setup_config, mknet_tensor_names, loss_tensor_names
        )
    else:
        train_simple_pipeline(
            n_iterations, setup_config, mknet_tensor_names, loss_tensor_names
        )

