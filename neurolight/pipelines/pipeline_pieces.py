import gunpowder as gp
from gunpowder.coordinate import Coordinate
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.points import PointsKey
from gunpowder.torch import Predict, Train
import numpy as np
import torch
import daisy

from neurolight.gunpowder.nodes import (
    TopologicalMatcher,
    RasterizeSkeleton,
    GrowLabels,
    BinarizeGt,
    ThresholdMask,
    GetNeuronPair,
    FusionAugment,
    RejectIfEmpty,
    NonMaxSuppression,
    FilterComponents,
)
from neurolight.gunpowder.nodes.helpers import UnSqueeze, Squeeze, ToInt64
from neurolight.gunpowder.contrib.nodes import AddDistance, TanhSaturate

from neurolight.gunpowder.pytorch.nodes.train_embedding import TrainEmbedding
from neurolight.gunpowder.pytorch.nodes.train_foreground import TrainForeground
from neurolight.networks.pytorch.radam import RAdam
from neurolight.networks.pytorch import (
    EmbeddingUnet,
    EmbeddingLoss,
    ForegroundUnet,
    ForegroundDistLoss,
    ForegroundBinLoss,
)


from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import math
import logging
import pickle

logger = logging.getLogger(__file__)


class RandomLocations(gp.RandomLocation):
    def __init__(self, *args, **kwargs):
        self.loc = kwargs.pop("loc")
        super(RandomLocations, self).__init__(*args, **kwargs)

    def setup(self):
        super(RandomLocations, self).setup()
        self.provides(self.loc, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        deps = super(RandomLocations, self).prepare(request)
        if self.loc in deps:
            del deps[self.loc]
        return deps

    def process(self, batch, request):
        batch[self.loc] = gp.Array(
            np.array(batch.get_total_roi().get_center()),
            spec=gp.ArraySpec(nonspatial=True),
        )
        super(RandomLocations, self).process(batch, request)


def get_training_inputs(setup_config, get_data_sources=None, locations=True):
    input_shape = gp.Coordinate(setup_config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(setup_config["OUTPUT_SHAPE"])
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    if setup_config["DATA"] == "train":
        samples = setup_config["TRAIN_SAMPLES"]
    elif setup_config["DATA"] == "validation":
        samples = setup_config["VALIDATION_SAMPLES"]
    elif setup_config["DATA"] == "test":
        samples = setup_config["TEST_SAMPLES"]
    logger.info(f"Using samples {samples} to generate data")

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    datasets = []

    if get_data_sources is None:
        (
            data_sources,
            raw,
            labels,
            consensus,
            nonempty_placeholder,
            matched,
        ) = get_mouselight_data_sources(setup_config, samples, locations=True)
    else:
        (
            data_sources,
            raw,
            labels,
            consensus,
            nonempty_placeholder,
            matched,
        ) = get_data_sources(setup_config)

    if setup_config.get("FUSION_PIPELINE"):
        (
            pipeline,
            raw,
            labels,
            matched,
            raw_base,
            labels_base,
            matched_base,
            consensus_base,
            raw_add,
            labels_add,
            matched_add,
            consensus_add,
            soft_mask,
            masked_base,
            masked_add,
            mask_maxed,
        ) = get_neuron_pair(
            data_sources,
            setup_config,
            raw,
            labels,
            matched,
            nonempty_placeholder,
            consensus,
        )

        datasets += [
            # get pair debugging data
            (raw_base, input_size, "volumes/raw_base", logging.DEBUG),
            (raw_add, input_size, "volumes/raw_add", logging.DEBUG),
            (labels_base, input_size, "volumes/labels_base", logging.DEBUG),
            (labels_add, input_size, "volumes/labels_add", logging.DEBUG),
            (matched_base, input_size, "points/matched_base", logging.DEBUG),
            (matched_add, input_size, "points/matched_add", logging.DEBUG),
            (consensus_base, input_size, "points/consensus_base", logging.DEBUG),
            (consensus_add, input_size, "points/consensus_add", logging.DEBUG),
        ]

        datasets += [
            # fusion debugging data
            (soft_mask, input_size, "volumes/soft_mask", logging.DEBUG),
            (masked_base, input_size, "volumes/masked_base", logging.DEBUG),
            (masked_add, input_size, "volumes/masked_add", logging.DEBUG),
            (mask_maxed, input_size, "volumes/mask_maxed", logging.DEBUG),
        ]
    else:

        pipeline = data_sources

    datasets += [
        # input data
        (raw, input_size, "volumes/raw", logging.INFO),
        (labels, output_size, "volumes/labels", logging.INFO),
        (matched, input_size, "points/matched", logging.INFO),
    ]

    if setup_config["GUARANTEE_NONEMPTY"]:
        pipeline = guarantee_nonempty(pipeline, setup_config, matched)

    return pipeline, datasets, raw, labels, matched


def get_mouselight_data_sources(
    setup_config: Dict[str, Any], source_samples: List[str], locations=False
):
    # Source Paths and accessibility
    raw_n5 = setup_config["RAW_N5"]
    mongo_url = setup_config["MONGO_URL"]
    samples_path = Path(setup_config["SAMPLES_PATH"])

    specified_locations = setup_config.get("SPECIFIED_LOCATIONS")

    # Graph matching parameters
    point_balance_radius = setup_config["POINT_BALANCE_RADIUS"]
    matching_failures_dir = setup_config["MATCHING_FAILURES_DIR"]
    matching_failures_dir = (
        matching_failures_dir
        if matching_failures_dir is None
        else Path(matching_failures_dir)
    )

    # Data Properties
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    micron_scale = voxel_size[0]

    # New array keys
    # Note: These are intended to be requested with size input_size
    raw = ArrayKey("RAW")
    consensus = gp.PointsKey("CONSENSUS")
    matched = gp.PointsKey("MATCHED")
    nonempty_placeholder = gp.PointsKey("NONEMPTY")
    labels = ArrayKey("LABELS")

    if setup_config["FUSION_PIPELINE"]:
        ensure_nonempty = nonempty_placeholder
    else:
        ensure_nonempty = consensus

    node_offset = {
        sample.name: (
            daisy.persistence.MongoDbGraphProvider(
                f"mouselight-{sample.name}-skeletonization", mongo_url
            ).num_nodes(None)
            + 1
        )
        for sample in samples_path.iterdir()
        if sample.name in source_samples
    }

    if specified_locations is not None:
        centers = pickle.load(open(specified_locations, "rb"))
        random = gp.SpecifiedLocation
        kwargs = {"locations": centers, "choose_randomly": True}
        logger.info(f"Using specified locations from {specified_locations}")
    elif not locations:
        random = gp.RandomLocation
        kwargs = {
            "ensure_nonempty": ensure_nonempty,
            "ensure_centered": True,
            "point_balance_radius": point_balance_radius * micron_scale,
        }
    else:
        random = RandomLocations
        kwargs = {
            "ensure_nonempty": ensure_nonempty,
            "ensure_centered": True,
            "point_balance_radius": point_balance_radius * micron_scale,
            "loc": gp.ArrayKey("RANDOM_LOCATION"),
        }

    data_sources = (
        tuple(
            (
                gp.ZarrSource(
                    filename=str((sample / raw_n5).absolute()),
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
                    f"mouselight-{sample.name}-matched",
                    mongo_url,
                    points=[matched],
                    directed=True,
                    node_attrs=[],
                    edge_attrs=[],
                ),
            )
            + gp.MergeProvider()
            + random(**kwargs)
            + gp.Normalize(raw)
            + FilterComponents(matched, node_offset[sample.name])
            + RasterizeSkeleton(
                points=matched,
                array=labels,
                array_spec=gp.ArraySpec(
                    interpolatable=False, voxel_size=voxel_size, dtype=np.int64
                ),
            )
            for sample in samples_path.iterdir()
            if sample.name in source_samples
        )
        + gp.RandomProvider()
    )

    return (data_sources, raw, labels, consensus, nonempty_placeholder, matched)


def get_snapshot_source(setup_config: Dict[str, Any], source_samples: List[str]):
    snapshot = setup_config.get("SNAPSHOT_SOURCE", "snapshots/snapshot_1.hdf")

    # Data Properties
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])

    # New array keys
    # Note: These are intended to be requested with size input_size
    raw = ArrayKey("RAW")
    consensus = gp.PointsKey("CONSENSUS")
    skeletonization = gp.PointsKey("SKELETONIZATION")
    matched = gp.PointsKey("MATCHED")
    nonempty_placeholder = gp.PointsKey("NONEMPTY")
    labels = ArrayKey("LABELS")

    data_sources = SnapshotSource(
        snapshot=snapshot,
        outputs={
            "volumes/raw": raw,
            "points/consensus": consensus,
            "points/skeletonization": skeletonization,
            "points/matched": matched,
            "points/matched": nonempty_placeholder,
            "points/labels": labels,
        },
        voxel_size=voxel_size,
    )

    return (
        data_sources,
        raw,
        labels,
        consensus,
        nonempty_placeholder,
        skeletonization,
        matched,
    )


def get_neuron_pair(
    pipeline,
    setup_config,
    raw: ArrayKey,
    labels: ArrayKey,
    matched: PointsKey,
    nonempty_placeholder: PointsKey,
    consensus: PointsKey,
):

    # Data Properties
    output_shape = gp.Coordinate(setup_config["OUTPUT_SHAPE"])
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    output_size = output_shape * voxel_size
    micron_scale = voxel_size[0]

    # Somewhat arbitrary hyperparameters
    shift_attempts = setup_config["SHIFT_ATTEMPTS"]
    request_attempts = setup_config["REQUEST_ATTEMPTS"]
    blend_smoothness = setup_config["BLEND_SMOOTHNESS"]
    seperate_by = setup_config["SEPERATE_BY"]
    seperate_distance = (np.array(seperate_by)).tolist()

    # array keys for fused volume
    raw_fused = ArrayKey("RAW_FUSED")
    labels_fused = ArrayKey("LABELS_FUSED")
    matched_fused = gp.PointsKey("MATCHED_FUSED")

    # array keys for base volume
    raw_base = ArrayKey("RAW_BASE")
    labels_base = ArrayKey("LABELS_BASE")
    matched_base = gp.PointsKey("MATCHED_BASE")
    consensus_base = gp.PointsKey("CONSENSUS_BASE")

    # array keys for add volume
    raw_add = ArrayKey("RAW_ADD")
    labels_add = ArrayKey("LABELS_ADD")
    matched_add = gp.PointsKey("MATCHED_ADD")
    consensus_add = gp.PointsKey("CONSENSUS_ADD")

    # debug array keys
    soft_mask = gp.ArrayKey("SOFT_MASK")
    masked_base = gp.ArrayKey("MASKED_BASE")
    masked_add = gp.ArrayKey("MASKED_ADD")
    mask_maxed = gp.ArrayKey("MASK_MAXED")

    pipeline = (
        pipeline
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
            extra_keys={consensus: (consensus_base, consensus_add)},
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
            blend_mode="labels_mask",  # TODO: Config this
            blend_smoothness=blend_smoothness * micron_scale,
            gaussian_smooth_mode="mirror",  # TODO: Config this
            num_blended_objects=0,  # TODO: Config this
        )
    )

    return (
        pipeline,
        raw_fused,
        labels_fused,
        matched_fused,
        raw_base,
        labels_base,
        matched_base,
        consensus_base,
        raw_add,
        labels_add,
        matched_add,
        consensus_add,
        soft_mask,
        masked_base,
        masked_add,
        mask_maxed,
    )


def add_data_augmentation(pipeline, raw):
    # TODO: fix elastic augment parameters
    # TODO: Config these
    pipeline = (
        pipeline
        + gp.ElasticAugment(
            [40, 10, 10],
            [0.25, 1, 1],
            [0, math.pi / 2.0],
            subsample=4,
            use_fast_points_transform=True,
            recompute_missing_points=False,
        )
        # + gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2])
        + gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001)
    )
    return pipeline


def add_label_processing(pipeline, setup_config, labels):
    if setup_config["DISTANCES"]:
        return add_distance_label_processing(pipeline, setup_config, labels)
    else:
        return add_binary_label_processing(pipeline, setup_config, labels)


def add_binary_label_processing(pipeline, setup_config, labels):

    # Data Properties
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    micron_scale = voxel_size[0]

    # Somewhat arbitrary Hyperparameter
    neuron_radius = setup_config["NEURON_RADIUS"]
    mask_radius = setup_config["MASK_RADIUS"]

    # New array keys
    gt_fg = ArrayKey("GT_FG")
    loss_weights = ArrayKey("LOSS_WEIGHTS")
    loss_mask = ArrayKey("LOSS_MASK")

    pipeline = (
        pipeline
        + GrowLabels(labels, radii=[neuron_radius * micron_scale])
        + BinarizeGt(labels, gt_fg)
        + GrowLabels(gt_fg, radii=[mask_radius * micron_scale], output=loss_mask)
        + gp.BalanceLabels(gt_fg, loss_weights, mask=loss_mask)
    )
    return pipeline, gt_fg, loss_weights


def guarantee_nonempty(pipeline, setup_config, key):
    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    output_size = Coordinate(setup_config["OUTPUT_SHAPE"]) * voxel_size
    num_components = setup_config["NUM_COMPONENTS"]

    pipeline = pipeline + RejectIfEmpty(
        key, centroid_size=output_size, num_components=num_components
    )

    return pipeline


def add_distance_label_processing(pipeline, setup_config, labels):

    # Data Properties
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    micron_scale = voxel_size[0]

    # Somewhat arbitrary Hyperparameter
    distance_threshold = setup_config["DISTANCE_THRESHOLD"]  # default 1e-4
    scale = setup_config["DISTANCE_SCALE"]

    # New array keys
    gt_fg = ArrayKey("GT_FG")
    loss_weights = ArrayKey("LOSS_WEIGHTS")  # binary

    pipeline = (
        pipeline
        + AddDistance(labels, gt_fg)
        + TanhSaturate(gt_fg, scale=micron_scale * scale)
        + ThresholdMask(gt_fg, loss_weights, distance_threshold)
    )
    return pipeline, gt_fg, loss_weights


def grow_labels(pipeline, setup_config, labels):

    # Data Properties
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    micron_scale = voxel_size[0]

    label_radius = setup_config["NEURON_RADIUS"]

    pipeline = pipeline + GrowLabels(labels, radii=[label_radius * micron_scale])

    return pipeline


def add_foreground_prediction(pipeline, setup_config, raw):
    checkpoint = setup_config.get("FOREGROUND_CHECKPOINT")
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    if checkpoint is None or not Path(checkpoint).exists():
        checkpoint = None
    else:
        checkpoint = Path(checkpoint)

    # New array keys
    fg_pred = ArrayKey("FG_PRED")

    pipeline = (
        pipeline
        + UnSqueeze(raw)
        + Predict(
            model=ForegroundUnet(setup_config),
            checkpoint=checkpoint,
            inputs={"raw": raw},
            outputs={0: fg_pred},
            array_specs={fg_pred: ArraySpec(dtype=np.float32, voxel_size=voxel_size)},
        )
    )

    return pipeline, fg_pred


def add_embedding_prediction(pipeline, setup_config, raw):
    checkpoint = setup_config.get("EMBEDDING_CHECKPOINT")
    voxel_size = Coordinate(setup_config.get("VOXEL_SIZE"))
    if checkpoint is None or not Path(checkpoint).exists():
        checkpoint = None
    else:
        checkpoint = Path(checkpoint)

    # New array keys
    embedding = ArrayKey("EMBEDDING")

    pipeline = pipeline + Predict(
        model=EmbeddingUnet(setup_config),
        checkpoint=checkpoint,
        inputs={"raw": raw},
        outputs={0: embedding},
        array_specs={embedding: ArraySpec(dtype=np.float32, voxel_size=voxel_size)},
    )

    return pipeline, embedding


def add_non_max_suppression(pipeline, setup_config, foreground):

    # Data properties
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    micron_scale = voxel_size[0]

    # Config options
    window_size = gp.Coordinate(setup_config["NMS_WINDOW_SIZE"]) * micron_scale
    threshold = setup_config["NMS_THRESHOLD"]

    # New array Key
    maxima = ArrayKey("MAXIMA")

    pipeline = pipeline + NonMaxSuppression(foreground, maxima, window_size, threshold)

    return pipeline, maxima


def add_caching(pipeline, setup_config):
    cache_size = setup_config["CACHE_SIZE"]
    num_workers = setup_config["NUM_WORKERS"]
    pipeline = pipeline + gp.PreCache(cache_size=cache_size, num_workers=num_workers)
    return pipeline


def add_embedding_training(pipeline, setup_config, raw, gt_labels, mask):

    # Network params
    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    input_size = Coordinate(setup_config["INPUT_SHAPE"]) * voxel_size
    output_size = Coordinate(setup_config["OUTPUT_SHAPE"]) * voxel_size

    # Config options
    embedding_net_name = setup_config["EMBEDDING_NET_NAME"]
    checkpoint_every = setup_config["CHECKPOINT_EVERY"]
    tensorboard_log_dir = setup_config["TENSORBOARD_LOG_DIR"]

    # New array Keys
    embedding = ArrayKey("EMBEDDING")
    embedding_gradient = ArrayKey("EMBEDDING_GRADIENT")

    # model, optimizer, loss
    model = EmbeddingUnet(setup_config)
    loss = EmbeddingLoss(setup_config)
    if setup_config.get("RADAM"):
        optimizer = RAdam(model.parameters(), lr=0.5e-5, betas=(0.95, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.5e-5, betas=(0.95, 0.999), eps=1e-8
        )

    pipeline = (
        pipeline
        + ToInt64(gt_labels)
        + Squeeze(mask)
        + Train(
            model=model,
            optimizer=optimizer,
            loss=loss,
            inputs={"raw": raw},
            loss_inputs={0: embedding, "target": gt_labels, "mask": mask},
            outputs={0: embedding},
            gradients={0: embedding_gradient},
            array_specs={
                embedding: ArraySpec(dtype=np.float32, voxel_size=voxel_size),
                embedding_gradient: ArraySpec(dtype=np.float32, voxel_size=voxel_size),
            },
            save_every=checkpoint_every,
            log_dir=tensorboard_log_dir,
            checkpoint_basename=embedding_net_name,
        )
    )

    return pipeline, embedding, embedding_gradient


def weighted_mse(pred, target, weights):
    mse = (pred - target) ** 2
    w_mse = weights * mse
    return w_mse.sum()


def add_foreground_training(pipeline, setup_config, raw, gt_fg, loss_weights):

    # Config options
    foreground_net_name = setup_config["FOREGROUND_NET_NAME"]
    checkpoint_every = setup_config["CHECKPOINT_EVERY"]
    tensorboard_log_dir = setup_config["TENSORBOARD_LOG_DIR"]

    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    input_size = Coordinate(setup_config["INPUT_SHAPE"]) * voxel_size
    output_size = Coordinate(setup_config["OUTPUT_SHAPE"]) * voxel_size

    # New array Keys
    fg_pred = ArrayKey("FG_PRED")
    fg_logits = ArrayKey("FG_LOGITS")
    fg_gradient = ArrayKey("FG_GRADIENT")
    fg_logits_gradient = ArrayKey("FG_LOGITS_GRADIENT")

    # model, optimizer, loss
    model = ForegroundUnet(setup_config)
    if setup_config.get("RADAM"):
        optimizer = RAdam(model.parameters(), lr=0.5e-5, betas=(0.95, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.5e-5, betas=(0.95, 0.999), eps=1e-8
        )
    if setup_config["DISTANCES"]:
        loss = ForegroundDistLoss()
    else:
        loss = ForegroundBinLoss()

    pipeline = pipeline + TrainForeground(
        model=model,
        optimizer=optimizer,
        loss=loss,
        inputs={"raw": raw},
        outputs={0: fg_pred, 1: fg_logits},
        target=gt_fg,
        gradients={0: fg_gradient, 1: fg_logits_gradient},
        save_every=checkpoint_every,
        log_dir=tensorboard_log_dir,
        weights=loss_weights,
        input_size=input_size,
        output_size=output_size,
        checkpoint_basename=foreground_net_name,
    )

    return pipeline, fg_pred, fg_gradient, fg_logits, fg_logits_gradient


def add_snapshot(
    pipeline,
    setup_config,
    datasets: List[Tuple[Union[ArrayKey, PointsKey], gp.Coordinate, str]],
):

    # Config options
    snapshot_every = setup_config["SNAPSHOT_EVERY"]
    snapshot_file_name = setup_config["SNAPSHOT_FILE_NAME"]

    # Snapshot request:
    snapshot_request = gp.BatchRequest()
    for key, size, *_ in datasets:
        snapshot_request.add(key, size)

    pipeline = pipeline + gp.Snapshot(
        additional_request=snapshot_request,
        output_filename=snapshot_file_name,
        dataset_names={key: location for key, _, location, *_ in datasets},
        every=snapshot_every,
    )

    return pipeline
