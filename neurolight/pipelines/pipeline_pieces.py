import gunpowder as gp
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
from neurolight.gunpowder.nodes.maxima import Skeletonize

from neurolight.gunpowder.pytorch.nodes.train_embedding import TrainEmbedding

from neurolight.gunpowder.nodes import (
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
from neurolight.gunpowder.nodes.neighborhood import Neighborhood

from neurolight.networks.pytorch.radam import RAdam
from neurolight.networks.pytorch import (
    EmbeddingUnet,
    EmbeddingLoss,
    ForegroundUnet,
    ForegroundDistLoss,
    ForegroundBinLoss,
)

from omegaconf.dictconfig import DictConfig

from pathlib import Path
from typing import List, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


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


def get_training_inputs(
    setup_config: DictConfig, get_data_sources=None, locations=True
):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    input_shape = gp.Coordinate(setup_config.data.input_shape)
    output_shape = gp.Coordinate(setup_config.data.output_shape)
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)
    if setup_config.data.data_set.name == "TRAIN":
        samples = setup_config.data.train_samples
    elif setup_config.data.data_set.name == "VALIDATE":
        samples = setup_config.data.validation_samples
    elif setup_config.data.data_set.name == "TEST":
        samples = setup_config.data.test_samples
    else:
        raise Exception(setup_config.data.data_set)
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

    if setup_config.pipeline.fusion_pipeline:
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

        if setup_config.fusion.blend_mode == "label_mask":
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

    if setup_config.data_gen.guarantee_nonempty:
        pipeline = guarantee_nonempty(pipeline, setup_config, matched)

    return pipeline, datasets, raw, labels, matched


def get_mouselight_data_sources(
    setup_config: DictConfig, source_samples: List[str], locations=False
):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    # Source Paths and accessibility
    raw_n5 = setup_config.data.raw_n5
    mongo_url = setup_config.data.mongo_url
    samples_path = Path(setup_config.data.samples_path)

    # Set the total roi so examples are only picked from a designated region
    total_roi = gp.Roi(setup_config.data.roi.offset, setup_config.data.roi.shape)
    logger.warning(f"Cropping inputs to roi: {total_roi}")

    # Graph matching parameters
    point_balance_radius = setup_config.random_location.point_balance_radius
    matching_failures_dir = setup_config.matching.matching_failures_dir
    matching_failures_dir = (
        matching_failures_dir
        if matching_failures_dir is None
        else Path(matching_failures_dir)
    )

    # Data Properties
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)
    output_shape = gp.Coordinate(setup_config.data.output_shape)
    output_size = output_shape * voxel_size
    micron_scale = voxel_size[0]

    distance_attr = setup_config.random_location.distance_attribute
    target_distance = float(setup_config.random_location.min_dist_to_fallback)
    max_nonempty_points = int(setup_config.random_location.max_random_location_points)

    mongo_db_template = setup_config.data.mongo_db_template
    matched_source = setup_config.data.matched_source

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
                    datasets={raw: "volume-rechunked"},
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
            + gp.Crop(matched, total_roi)
            + gp.Crop(nonempty_placeholder, total_roi)
            + gp.Crop(raw, total_roi)
            + random(**kwargs)
            + gp.Normalize(raw)
            + FilterComponents(
                matched, node_offset[sample.name], centroid_size=output_size
            )
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


def get_snapshot_source(setup_config: DictConfig, source_samples: List[str]):
    raise NotImplementedError("Not yet migrated to refactored configs!")
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    snapshot = setup_config.get("SNAPSHOT_SOURCE", "snapshots/snapshot_1.hdf")

    # Data Properties
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)

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
    setup_config: DictConfig,
    raw: ArrayKey,
    labels: ArrayKey,
    matched: PointsKey,
    nonempty_placeholder: PointsKey,
):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # Data Properties
    output_shape = gp.Coordinate(setup_config.data.output_shape)
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)
    output_size = output_shape * voxel_size
    micron_scale = voxel_size[0]

    # Somewhat arbitrary hyperparameters
    blend_mode = setup_config.fusion.blend_mode
    shift_attempts = setup_config.data_gen.shift_attempts
    request_attempts = setup_config.data_gen.request_attempts
    blend_smoothness = setup_config.fusion.blend_smoothness
    seperate_by = setup_config.data_gen.seperate_by
    seperate_distance = (np.array(seperate_by)).tolist()

    clip_limit = float(setup_config.clahe.clip_limit)
    normalize = setup_config.clahe.normalize
    pre_fusion = setup_config.clahe.pre_fusion
    post_fusion = setup_config.clahe.post_fusion

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
        if pre_fusion:
            pipeline = pipeline + scipyCLAHE(
                [raw_add, raw_base],
                gp.Coordinate([20, 64, 64]) * voxel_size,
                clip_limit=clip_limit,
                normalize=normalize,
            )
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
        if post_fusion:
            pipeline = pipeline + scipyCLAHE(
                [raw_add, raw_base],
                gp.Coordinate([20, 64, 64]) * voxel_size,
                clip_limit=clip_limit,
                normalize=normalize,
            )

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
        + gp.IntensityAugment(raw, 0.8, 1.2, -0.001, 0.001)
    )
    return pipeline


def add_label_processing(pipeline, setup_config: DictConfig, labels):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    if setup_config.pipeline.distances:
        return add_distance_label_processing(pipeline, setup_config, labels)
    else:
        return add_binary_label_processing(pipeline, setup_config, labels)


def add_binary_label_processing(pipeline, setup_config: DictConfig, labels):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # Data Properties
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)
    micron_scale = voxel_size[0]

    # Somewhat arbitrary Hyperparameter
    neuron_radius = setup_config.data_processing.neuron_radius
    mask_radius = setup_config.data_processing.mask_radius

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


def guarantee_nonempty(pipeline, setup_config: DictConfig, key):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    voxel_size = Coordinate(setup_config.data.voxel_size)
    output_size = Coordinate(setup_config.data.output_shape) * voxel_size
    num_components = setup_config.data_gen.num_components

    pipeline = pipeline + RejectIfEmpty(
        key, centroid_size=output_size, num_components=num_components
    )

    return pipeline


def add_distance_label_processing(pipeline, setup_config: DictConfig, labels):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # Data Properties
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)
    micron_scale = voxel_size[0]

    # Somewhat arbitrary Hyperparameter
    distance_threshold = setup_config.data_processing.distance_threshold
    scale = setup_config.data_processing.distance_scale

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


def grow_labels(pipeline, setup_config: DictConfig, labels):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # Data Properties
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)
    micron_scale = voxel_size[0]

    label_radius = setup_config.data_processing.neuron_radius

    pipeline = pipeline + GrowLabels(labels, radii=[label_radius * micron_scale])

    return pipeline


def add_foreground_prediction(pipeline, setup_config: DictConfig, raw):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    checkpoint = setup_config.fg_model.checkpoint
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)

    checkpoint_file = (
        f"{setup_config.fg_model.directory}/{setup_config.fg_model.setup}"
        f"{setup_config.fg_model.net_name}_checkpoint_{setup_config.fg_model.checkpoint}"
    )
    if checkpoint is None:
        checkpoint = None
    else:
        checkpoint = Path(checkpoint_file)

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


def add_embedding_prediction(pipeline, setup_config: DictConfig, raw):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    checkpoint = setup_config.emb_model.checkpoint
    voxel_size = Coordinate(setup_config.data.voxel_size)
    if checkpoint is None or not Path(checkpoint).exists():
        checkpoint = None
    else:
        checkpoint = Path(checkpoint)

    # New array keys
    embedding = ArrayKey("EMBEDDING")

    if setup_config.emb_model.aux_task.enabled:
        neighborhood = ArrayKey("NEIGHBORHODD")
        pipeline = (
            pipeline
            + UnSqueeze([raw])
            + Predict(
                model=EmbeddingUnet(setup_config),
                checkpoint=checkpoint,
                inputs={"raw": raw},
                outputs={0: embedding, 1: neighborhood},
                array_specs={
                    embedding: ArraySpec(dtype=np.float32, voxel_size=voxel_size)
                },
            )
            + Squeeze([raw, embedding])
        )

        return pipeline, embedding, neighborhood

    else:
        pipeline = (
            pipeline
            + UnSqueeze([raw])
            + Predict(
                model=EmbeddingUnet(setup_config),
                checkpoint=checkpoint,
                inputs={"raw": raw},
                outputs={0: embedding},
                array_specs={
                    embedding: ArraySpec(dtype=np.float32, voxel_size=voxel_size)
                },
            )
            + Squeeze([raw, embedding])
        )

        return pipeline, embedding


def get_candidates(pipeline, setup_config: DictConfig, foreground):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # Data properties
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)
    micron_scale = voxel_size[0]

    # Config options
    window_size = gp.Coordinate(setup_config.candidates.nms_window_size) * micron_scale
    threshold = setup_config.candidates.nms_threshold

    # Candidate mode
    mode = setup_config.candidates.mode

    # New array Key
    maxima = ArrayKey("MAXIMA")

    if mode == "skel":
        pipeline = pipeline + Skeletonize(
            foreground, maxima, min(window_size), threshold=threshold
        )
    else:
        pipeline = (
            pipeline
            + UnSqueeze([foreground])
            + NonMaxSuppression(foreground, maxima, window_size, threshold)
            + Squeeze([foreground, maxima])
        )

    return pipeline, maxima


def add_caching(pipeline, setup_config):
    cache_size = setup_config.precache.cache_size
    num_workers = setup_config.precache.num_workers
    pipeline = pipeline + gp.PreCache(cache_size=cache_size, num_workers=num_workers)
    return pipeline


def add_neighborhood(pipeline, setup_config: DictConfig, gt):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # New array Keys
    neighborhood = ArrayKey("NEIGHBORHOOD_GT")
    neighborhood_mask = ArrayKey("NEIGHBORHOOD_MASK")

    # Config options
    distance = setup_config.emb_model.aux_task.distance
    neighborhood_k = setup_config.emb_model.aux_task.neighborhood.value

    # Data details
    voxel_size = Coordinate(setup_config.data.voxel_size)

    pipeline = pipeline + Neighborhood(
        gt,
        neighborhood,
        neighborhood_mask,
        distance,
        neighborhood_k,
        array_specs={
            neighborhood: ArraySpec(voxel_size=voxel_size),
            neighborhood_mask: ArraySpec(voxel_size=voxel_size),
        },
    )

    return pipeline, neighborhood, neighborhood_mask


def add_embedding_training(
    pipeline,
    setup_config: DictConfig,
    raw,
    gt_labels,
    mask,
    neighborhood_gt=None,
    neighborhood_mask=None,
):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # Network params
    voxel_size = Coordinate(setup_config.data.voxel_size)

    # Config options
    embedding_net_name = setup_config.emb_model.net_name
    checkpoint_every = setup_config.training.checkpoint_every
    tensorboard_log_dir = setup_config.training.tensorboard_log_dir

    # New array Keys
    embedding = ArrayKey("EMBEDDING")
    embedding_gradient = ArrayKey("EMBEDDING_GRADIENT")
    neighborhood = ArrayKey("NEIGHBORHOOD")
    neighborhood_gradient = ArrayKey("NEIGHBORHOOD_GRADIENT")

    # model, optimizer, loss
    model = EmbeddingUnet(setup_config)
    loss = EmbeddingLoss(setup_config)
    if setup_config.optimizer.radam:
        optimizer = RAdam(model.parameters(), lr=0.5e-5, betas=(0.95, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.5e-5, betas=(0.95, 0.999), eps=1e-8
        )

    if setup_config.emb_model.aux_task.enabled:
        pipeline = (
            pipeline
            + ToInt64(gt_labels)
            + UnSqueeze([raw, gt_labels, mask, neighborhood_gt, neighborhood_mask])
            + TrainEmbedding(
                model=model,
                optimizer=optimizer,
                loss=loss,
                inputs={"raw": raw},
                loss_inputs={
                    0: embedding,
                    "target": gt_labels,
                    "mask": mask,
                    "neighborhood": neighborhood_gt,
                    "neighborhood_mask": neighborhood_mask,
                    "neighborhood_target": neighborhood,
                },
                outputs={0: embedding, 1: neighborhood},
                gradients={0: embedding_gradient, 1: neighborhood_gradient},
                array_specs={
                    embedding: ArraySpec(dtype=np.float32, voxel_size=voxel_size),
                    embedding_gradient: ArraySpec(
                        dtype=np.float32, voxel_size=voxel_size
                    ),
                    neighborhood: ArraySpec(dtype=np.float32, voxel_size=voxel_size),
                    neighborhood_gradient: ArraySpec(
                        dtype=np.float32, voxel_size=voxel_size
                    ),
                },
                save_every=checkpoint_every,
                log_dir=tensorboard_log_dir,
                checkpoint_basename=embedding_net_name,
            )
            + Squeeze(
                [
                    embedding,
                    embedding_gradient,
                    neighborhood,
                    neighborhood_gradient,
                    neighborhood_gt,
                    neighborhood_mask,
                    raw,
                    gt_labels,
                    mask,
                ]
            )
        )
        return (
            pipeline,
            embedding,
            embedding_gradient,
            neighborhood,
            neighborhood_gradient,
        )
    else:
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
                    embedding_gradient: ArraySpec(
                        dtype=np.float32, voxel_size=voxel_size
                    ),
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


def add_foreground_training(
    pipeline, setup_config: DictConfig, raw, gt_fg, loss_weights
):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # Config options
    foreground_net_name = setup_config.fg_model.net_name
    checkpoint_every = setup_config.training.checkpoint_every
    tensorboard_log_dir = setup_config.training.tensorboard_log_dir

    voxel_size = Coordinate(setup_config.data.voxel_size)

    # New array Keys
    fg_pred = ArrayKey("FG_PRED")
    fg_gradient = ArrayKey("FG_GRADIENT")

    # model, optimizer, loss
    model = ForegroundUnet(setup_config)
    if setup_config.optimizer.radam:
        optimizer = RAdam(model.parameters(), lr=0.5e-5, betas=(0.95, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.5e-5, betas=(0.95, 0.999), eps=1e-8
        )
    if setup_config.pipeline.distances:
        loss = ForegroundDistLoss()
    else:
        loss = ForegroundBinLoss()

    pipeline = (
        pipeline
        + UnSqueeze([raw, gt_fg, loss_weights])
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
        + Squeeze([raw, gt_fg, loss_weights, fg_pred, fg_gradient])
    )

    return pipeline, fg_pred, fg_gradient


def add_snapshot(
    pipeline,
    setup_config: DictConfig,
    datasets: List[Tuple[Union[ArrayKey, PointsKey], gp.Coordinate, str]],
):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"

    # Config options
    snapshot_every = setup_config.snapshot.every
    snapshot_file_name = setup_config.snapshot.file_name
    snapshot_dir = setup_config.snapshot.directory

    # Snapshot request:
    snapshot_request = gp.BatchRequest()
    for key, size, *_ in datasets:
        snapshot_request.add(key, size)

    pipeline = pipeline + gp.Snapshot(
        additional_request=snapshot_request,
        output_dir=snapshot_dir,
        output_filename=snapshot_file_name,
        dataset_names={key: location for key, _, location, *_ in datasets},
        every=snapshot_every,
    )

    return pipeline
