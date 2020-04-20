import gunpowder as gp
import neurolight as nl
from gunpowder.coordinate import Coordinate
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.points import PointsKey
from gunpowder.torch import Predict, Train
import numpy as np
import torch
import daisy

from gunpowder.nodes.random_location import RandomLocation
from neurolight.gunpowder.nodes.daisy_graph_provider import DaisyGraphProvider
from neurolight.gunpowder.nodes.filtered_daisy_graph_provider import (
    FilteredDaisyGraphProvider,
)

from neurolight.gunpowder.pytorch.nodes.train_embedding import TrainEmbedding

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
# from neurolight.gunpowder.nodes.helpers import UnSqueeze, Squeeze
from neurolight.gunpowder.nodes.helpers import ToInt64
from neurolight.gunpowder.pytorch.nodes.helpers import UnSqueeze, Squeeze
from neurolight.gunpowder.contrib.nodes import AddDistance, TanhSaturate
from neurolight.gunpowder.nodes.clahe import scipyCLAHE

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
            nonempty_placeholder,
            matched,
        ) = get_mouselight_data_sources(setup_config, samples, locations=True)
    else:
        (data_sources, raw, labels, nonempty_placeholder, matched) = get_data_sources(
            setup_config
        )

    if setup_config.get("FUSION_PIPELINE"):
        (
            pipeline,
            raw,
            labels,
            matched,
            raw_base,
            labels_base,
            matched_base,
            raw_add,
            labels_add,
            matched_add,
            soft_mask,
            masked_base,
            masked_add,
            mask_maxed,
        ) = get_neuron_pair(
            data_sources, setup_config, raw, labels, matched, nonempty_placeholder
        )

        datasets += [
            # get pair debugging data
            (raw_base, input_size, "volumes/raw_base", logging.DEBUG),
            (raw_add, input_size, "volumes/raw_add", logging.DEBUG),
            (labels_base, input_size, "volumes/labels_base", logging.DEBUG),
            (labels_add, input_size, "volumes/labels_add", logging.DEBUG),
            (matched_base, input_size, "points/matched_base", logging.DEBUG),
            (matched_add, input_size, "points/matched_add", logging.DEBUG),
        ]

        if setup_config["BLEND_MODE"] == "label_mask":
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

    # specified_locations = setup_config.get("SPECIFIED_LOCATIONS")

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
    output_shape = gp.Coordinate(setup_config["OUTPUT_SHAPE"])
    output_size = output_shape * voxel_size
    micron_scale = voxel_size[0]

    distance_attr = setup_config["DISTANCE_ATTRIBUTE"]
    target_distance = float(setup_config["MIN_DIST_TO_FALLBACK"])
    max_nonempty_points = int(setup_config["MAX_RANDOM_LOCATION_POINTS"])

    mongo_db_template = setup_config["MONGO_DB_TEMPLATE"]
    matched_source = setup_config.get("MATCHED_SOURCE", "matched")

    # New array keys
    # Note: These are intended to be requested with size input_size
    raw = ArrayKey("RAW")
    matched = gp.PointsKey("MATCHED")
    nonempty_placeholder = gp.PointsKey("NONEMPTY")
    labels = ArrayKey("LABELS")

    ensure_nonempty = nonempty_placeholder

    node_offset = {
        sample.name: (
            daisy.persistence.MongoDbGraphProvider(
                mongo_db_template.format(sample=sample.name, source="skeletonization"),
                mongo_url,
            ).num_nodes(None)
            + 1
        )
        for sample in samples_path.iterdir()
        if sample.name in source_samples
    }

    # if specified_locations is not None:
    #     centers = pickle.load(open(specified_locations, "rb"))
    #     random = gp.SpecifiedLocation
    #     kwargs = {"locations": centers, "choose_randomly": True}
    #     logger.info(f"Using specified locations from {specified_locations}")
    # elif locations:
    #     random = RandomLocations
    #     kwargs = {
    #         "ensure_nonempty": ensure_nonempty,
    #         "ensure_centered": True,
    #         "point_balance_radius": point_balance_radius * micron_scale,
    #         "loc": gp.ArrayKey("RANDOM_LOCATION"),
    #     }
    # else:

    random = RandomLocation
    kwargs = {
        "ensure_nonempty": ensure_nonempty,
        "ensure_centered": True,
        "point_balance_radius": point_balance_radius * micron_scale,
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
                DaisyGraphProvider(
                    mongo_db_template.format(sample=sample.name, source=matched_source),
                    mongo_url,
                    points=[matched],
                    directed=True,
                    node_attrs=[],
                    edge_attrs=[],
                ),
                FilteredDaisyGraphProvider(
                    mongo_db_template.format(sample=sample.name, source=matched_source),
                    mongo_url,
                    points=[nonempty_placeholder],
                    directed=True,
                    node_attrs=["distance_to_fallback"],
                    edge_attrs=[],
                    num_nodes=max_nonempty_points,
                    dist_attribute=distance_attr,
                    min_dist=target_distance,
                ),
            )
            + gp.MergeProvider()
            + random(**kwargs)
            + gp.Normalize(raw)
            + FilterComponents(matched, node_offset[sample.name], centroid_size=output_size)
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

    return (data_sources, raw, labels, nonempty_placeholder, matched)


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
):

    # Data Properties
    output_shape = gp.Coordinate(setup_config["OUTPUT_SHAPE"])
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    output_size = output_shape * voxel_size
    micron_scale = voxel_size[0]

    # Somewhat arbitrary hyperparameters
    blend_mode = setup_config["BLEND_MODE"]
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

    # array keys for add volume
    raw_add = ArrayKey("RAW_ADD")
    labels_add = ArrayKey("LABELS_ADD")
    matched_add = gp.PointsKey("MATCHED_ADD")

    # debug array keys
    soft_mask = gp.ArrayKey("SOFT_MASK")
    masked_base = gp.ArrayKey("MASKED_BASE")
    masked_add = gp.ArrayKey("MASKED_ADD")
    mask_maxed = gp.ArrayKey("MASK_MAXED")

    pipeline = pipeline + GetNeuronPair(
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
        # nonempty_placeholder=nonempty_placeholder,
        nonempty_placeholder=nonempty_placeholder,
    )
    if blend_mode == "add":
        pipeline = pipeline + scipyCLAHE([raw_add, raw_base], [20, 64, 64])
    pipeline = pipeline + FusionAugment(
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
        blend_mode=blend_mode,
        blend_smoothness=blend_smoothness * micron_scale,
        gaussian_smooth_mode="mirror",  # TODO: Config this
        num_blended_objects=0,  # TODO: Config this
    )
    if blend_mode == "add":
        pipeline = pipeline + scipyCLAHE([raw_fused], [20, 64, 64])

    return (
        pipeline,
        raw_fused,
        labels_fused,
        matched_fused,
        raw_base,
        labels_base,
        matched_base,
        raw_add,
        labels_add,
        matched_add,
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
        + UnSqueeze([raw])
        + Predict(
            model=ForegroundUnet(setup_config),
            checkpoint=checkpoint,
            inputs={"raw": raw},
            outputs={0: fg_pred},
            array_specs={fg_pred: ArraySpec(dtype=np.float32, voxel_size=voxel_size)},
        )
        + Squeeze([raw, fg_pred])
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

    pipeline = (
        pipeline
        + UnSqueeze([raw])
        + Predict(
            model=EmbeddingUnet(setup_config),
            checkpoint=checkpoint,
            inputs={"raw": raw},
            outputs={0: embedding},
            array_specs={embedding: ArraySpec(dtype=np.float32, voxel_size=voxel_size)},
        )
        + Squeeze([raw, embedding])
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

    pipeline = (
        pipeline
        + UnSqueeze([foreground])
        + NonMaxSuppression(foreground, maxima, window_size, threshold)
        + Squeeze([foreground, maxima])
    )

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
        + UnSqueeze([raw, gt_labels, mask])
        + TrainEmbedding(
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
        + Squeeze([embedding, embedding_gradient, raw, gt_labels, mask])
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

    # New array Keys
    fg_pred = ArrayKey("FG_PRED")
    fg_gradient = ArrayKey("FG_GRADIENT")

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

    pipeline = (
        pipeline
        + UnSqueeze(raw)
        # + UnSqueeze(gt_fg)
        # + UnSqueeze(loss_weights)
        + Train(
            model=model,
            optimizer=optimizer,
            loss=loss,
            inputs={"raw": raw},
            loss_inputs={0: fg_pred, "target": gt_fg, "weights": loss_weights},
            outputs={0: fg_pred},
            gradients={0: fg_gradient},
            array_specs={
                fg_pred: ArraySpec(dtype=np.float32, voxel_size=voxel_size),
                fg_gradient: ArraySpec(dtype=np.float32, voxel_size=voxel_size),
            },
            save_every=checkpoint_every,
            log_dir=tensorboard_log_dir,
            checkpoint_basename=foreground_net_name,
        )
        + Squeeze(raw)
        # + Squeeze(gt_fg)
        # + Squeeze(loss_weights)
        + Squeeze(fg_pred)
        + Squeeze(fg_gradient)
    )

    return pipeline, fg_pred, fg_gradient


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
