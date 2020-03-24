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


class ThresholdMask(gp.BatchFilter):
    def __init__(self, array, mask, threshold):
        self.array = array
        self.mask = mask
        self.threshold = threshold

    def setup(self):
        mask_spec = copy.deepcopy(self.spec[self.array])
        mask_spec.dtype = np.uint32
        self.provides(self.mask, mask_spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = copy.deepcopy(request[self.mask])
        return deps

    def process(self, batch, request):
        mask = (batch[self.array].data > self.threshold).astype(np.uint32)
        mask_spec = copy.deepcopy(batch[self.array].spec)
        mask_spec.dtype = np.uint32
        batch[self.mask] = gp.Array(mask, mask_spec)

        return batch


class RejectIfEmpty(gp.BatchProvider):
    def __init__(
        self,
        ensure_nonempty: gp.ArrayKey,
        center_size: gp.Coordinate = None,
        request_limit: int = 100,
    ):
        self.ensure_nonempty = ensure_nonempty
        self.center_size = center_size
        self.request_limit = request_limit

    @property
    def upstream_provider(self):
        return self.get_upstream_providers()[0]

    @property
    def spec(self) -> gp.ProviderSpec:
        return self.upstream_provider.spec

    def setup(self):
        pass

    def provide(self, request):

        k = 0
        found_valid = False
        while (
            not found_valid
            and k < self.request_limit
            and self.ensure_nonempty in request
        ):
            pre_request = gp.BatchRequest()
            seed = pre_request._random_seed
            pre_request[self.ensure_nonempty] = request[self.ensure_nonempty]
            for key, value in request.items():
                pre_request.place_holders[key] = value

            batch = self.upstream_provider.request_batch(pre_request)
            input_roi = batch[self.ensure_nonempty].spec.roi
            output_size = (
                self.center_size
                if self.center_size is not None
                else input_roi.get_shape()
            )
            inner_roi = gp.Roi(
                gp.Coordinate(
                    input_roi.get_begin() + (input_roi.get_shape() - output_size) / 2
                ),
                output_size,
            )
            node_count = len(batch[self.ensure_nonempty].graph.crop(inner_roi).nodes)
            if node_count > 0:
                found_valid = True
            k += 1

        if not found_valid:
            logger.info("Never found a valid batch")

        full_request = gp.BatchRequest()
        full_request._random_seed = seed
        for key, spec in request.items():
            full_request[key] = spec

        batch = self.upstream_provider.request_batch(full_request)
        if found_valid:
            node_count2 = len(batch[self.ensure_nonempty].graph.crop(inner_roi).nodes)
            if not node_count == node_count2:
                logger.warning(
                    f"node counts don't match! Seen {node_count2}, expected {node_count}"
                )

        return batch


def train(n_iterations, setup_config, mknet_tensor_names, loss_tensor_names):
    if setup_config["DISTANCES"]:
        train_distance_pipeline(
            n_iterations, setup_config, mknet_tensor_names, loss_tensor_names
        )
    else:
        train_simple_pipeline(
            n_iterations, setup_config, mknet_tensor_names, loss_tensor_names
        )


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
    fg = gp.ArrayKey("FG")
    gradient_fg = gp.ArrayKey("GRADIENT_FG")

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
    snapshot_request.add(fg, output_size, voxel_size=voxel_size)
    snapshot_request.add(gt_fg, output_size, voxel_size=voxel_size)
    snapshot_request.add(gradient_fg, output_size, voxel_size=voxel_size)

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
        + RejectIfEmpty(matched, center_size=output_size)
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
        + gp.BalanceLabels(labels_fg_bin, loss_weights, clipmin=0.01)
        + gp.PreCache(cache_size=cache_size, num_workers=num_workers)
        + gp.tensorflow.Train(
            "train_net_foreground",
            optimizer=mknet_tensor_names["optimizer"],
            loss=mknet_tensor_names["fg_loss"],
            inputs={
                mknet_tensor_names["loss_weights"]: loss_weights,
                mknet_tensor_names["raw"]: raw,
                mknet_tensor_names["gt_labels"]: labels_fg,
            },
            outputs={mknet_tensor_names["fg"]: fg, mknet_tensor_names["gt_fg"]: gt_fg},
            gradients={mknet_tensor_names["fg_logits"]: gradient_fg},
            save_every=checkpoint_every,
            summary=mknet_tensor_names["summaries"],
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
                fg: "volumes/fg",
                gt_fg: "volumes/gt_fg",
                gradient_fg: "volumes/gradient_fg",
                # output debug data
                loss_weights: "volumes/loss_weights",
            },
            every=snapshot_every,
        )
    )

    with gp.build(pipeline):
        for _ in range(num_iterations):
            pipeline.request_batch(request)


def train_distance_pipeline(
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
    max_label_dist = setup_config["MAX_LABEL_DIST"]

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

    dist = gp.ArrayKey("DIST")
    dist_mask = gp.ArrayKey("DIST_MASK")
    dist_cropped = gp.ArrayKey("DIST_CROPPED")
    loss_weights = gp.ArrayKey("LOSS_WEIGHTS")

    # tensorflow tensors
    fg_dist = gp.ArrayKey("FG_DIST")
    gradient_fg = gp.ArrayKey("GRADIENT_FG")

    # add request
    request = gp.BatchRequest()
    request.add(dist_mask, output_size)
    request.add(dist_cropped, output_size)
    request.add(raw, input_size)
    request.add(labels, input_size)
    request.add(dist, input_size)
    request.add(matched, input_size)
    request.add(skeletonization, input_size)
    request.add(consensus, input_size)
    request.add(loss_weights, output_size)

    # add snapshot request
    snapshot_request = gp.BatchRequest()

    # tensorflow requests
    snapshot_request.add(raw, input_size)  # input_size request for positioning
    snapshot_request.add(gradient_fg, output_size, voxel_size=voxel_size)
    snapshot_request.add(fg_dist, output_size, voxel_size=voxel_size)

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
        + RejectIfEmpty(matched, center_size=output_size)
        + RasterizeSkeleton(
            points=matched,
            array=labels,
            array_spec=gp.ArraySpec(
                interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
            ),
        )
        + gp.contrib.nodes.add_distance.AddDistance(
            labels, dist, dist_mask, max_distance=max_label_dist * micron_scale
        )
        + gp.contrib.nodes.tanh_saturate.TanhSaturate(
            dist, scale=micron_scale, offset=1
        )
        + ThresholdMask(dist, loss_weights, 1e-4)
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
        + Crop(dist, dist_cropped)
        # + gp.PreCache(cache_size=cache_size, num_workers=num_workers)
        + gp.tensorflow.Train(
            "train_net_foreground",
            optimizer=mknet_tensor_names["optimizer"],
            loss=mknet_tensor_names["fg_loss"],
            inputs={
                mknet_tensor_names["raw"]: raw,
                mknet_tensor_names["gt_distances"]: dist_cropped,
                mknet_tensor_names["loss_weights"]: loss_weights,
            },
            outputs={mknet_tensor_names["fg_pred"]: fg_dist},
            gradients={mknet_tensor_names["fg_pred"]: gradient_fg},
            save_every=checkpoint_every,
            # summary=mknet_tensor_names["summaries"],
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
                labels: "volumes/labels",
                # labeled data
                dist_cropped: "volumes/dist",
                # trees
                skeletonization: "points/skeletonization",
                consensus: "points/consensus",
                matched: "points/matched",
                # output volumes
                fg_dist: "volumes/fg_dist",
                gradient_fg: "volumes/gradient_fg",
                # output debug data
                dist_mask: "volumes/dist_mask",
                loss_weights: "volumes/loss_weights"
            },
            every=snapshot_every,
        )
    )

    with gp.build(pipeline):
        for _ in range(num_iterations):
            pipeline.request_batch(request)
