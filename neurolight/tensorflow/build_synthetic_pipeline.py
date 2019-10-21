import neurolight as nl
from neurolight.tensorflow.add_loss import create_custom_loss
import gunpowder as gp
import numpy as np

from typing import List
import math


class LabelToFloat32(gp.BatchFilter):
    def __init__(self, array: gp.ArrayKey, intensities: List[float] = [1.0]):
        self.array = array
        self.intensities = intensities

    def setup(self):
        self.enable_autoskip()
        spec = self.spec[self.array].copy()
        spec.dtype = np.float32
        self.updates(self.array, spec)

    def process(self, batch, request):
        array = batch[self.array]
        intensity_data = np.zeros_like(array.data, dtype=np.float32)
        for i, label in enumerate(np.unique(array.data)):
            if label == 0:
                continue
            mask = array.data == label
            intensity_data[mask] = np.maximum(
                intensity_data[mask], self.intensities[i % len(self.intensities)]
            )
        spec = array.spec
        spec.dtype = np.float32
        batch[self.array] = gp.Array(intensity_data, spec)


def train(n_iterations, setup_config, tensor_names, TENSOR_NAMES):

    # Network hyperparams
    INPUT_SHAPE = setup_config["INPUT_SHAPE"]
    OUTPUT_SHAPE = setup_config["OUTPUT_SHAPE"]

    # Skeleton generation hyperparams
    SKEL_GEN_RADIUS = setup_config["SKEL_GEN_RADIUS"]
    THETAS = np.array(setup_config["THETAS"]) * math.pi
    SPLIT_PS = setup_config["SPLIT_PS"]
    NOISE_VAR = setup_config["NOISE_VAR"]
    N_OBJS = setup_config["N_OBJS"]

    # Skeleton variation hyperparams
    LABEL_RADII = setup_config["LABEL_RADII"]
    RAW_RADII = setup_config["RAW_RADII"]
    RAW_INTENSITIES = setup_config["RAW_INTENSITIES"]

    # Training hyperparams
    CACHE_SIZE = setup_config["CACHE_SIZE"]
    NUM_WORKERS = setup_config["NUM_WORKERS"]
    SNAPSHOT_EVERY = setup_config["SNAPSHOT_EVERY"]
    CHECKPOINT_EVERY = setup_config["CHECKPOINT_EVERY"]

    point_trees = gp.PointsKey("POINT_TREES")
    labels = gp.ArrayKey("LABELS")
    raw = gp.ArrayKey("RAW")
    gt_fg = gp.ArrayKey("GT_FG")
    embedding = gp.ArrayKey("EMBEDDING")
    fg = gp.ArrayKey("FG")
    maxima = gp.ArrayKey("MAXIMA")
    gradient_embedding = gp.ArrayKey("GRADIENT_EMBEDDING")
    gradient_fg = gp.ArrayKey("GRADIENT_FG")

    # tensorflow tensors
    emst = gp.ArrayKey("EMST")
    edges_u = gp.ArrayKey("EDGES_U")
    edges_v = gp.ArrayKey("EDGES_V")
    ratio_pos = gp.ArrayKey("RATIO_POS")
    ratio_neg = gp.ArrayKey("RATIO_NEG")
    dist = gp.ArrayKey("DIST")
    num_pos_pairs = gp.ArrayKey("NUM_POS")
    num_neg_pairs = gp.ArrayKey("NUM_NEG")

    request = gp.BatchRequest()
    request.add(raw, INPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    request.add(labels, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    request.add(point_trees, INPUT_SHAPE)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, INPUT_SHAPE)
    snapshot_request.add(embedding, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(gt_fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(maxima, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(
        gradient_embedding, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1))
    )
    snapshot_request.add(gradient_fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request[emst] = gp.ArraySpec()
    snapshot_request[edges_u] = gp.ArraySpec()
    snapshot_request[edges_v] = gp.ArraySpec()
    snapshot_request[ratio_pos] = gp.ArraySpec()
    snapshot_request[ratio_neg] = gp.ArraySpec()
    snapshot_request[dist] = gp.ArraySpec()
    snapshot_request[num_pos_pairs] = gp.ArraySpec()
    snapshot_request[num_neg_pairs] = gp.ArraySpec()

    pipeline = (
        nl.SyntheticLightLike(
            point_trees,
            dims=2,
            r=SKEL_GEN_RADIUS,
            n_obj=N_OBJS,
            thetas=THETAS,
            split_ps=SPLIT_PS,
        )
        # + gp.SimpleAugment()
        # + gp.ElasticAugment([10, 10], [0.1, 0.1], [0, 2.0 * math.pi], spatial_dims=2)
        + nl.RasterizeSkeleton(
            point_trees,
            raw,
            gp.ArraySpec(
                roi=gp.Roi((None,) * 2, (None,) * 2),
                voxel_size=gp.Coordinate((1, 1)),
                dtype=np.uint64,
            ),
        )
        + nl.RasterizeSkeleton(
            point_trees,
            labels,
            gp.ArraySpec(
                roi=gp.Roi((None,) * 2, (None,) * 2),
                voxel_size=gp.Coordinate((1, 1)),
                dtype=np.uint64,
            ),
            use_component=True,
            n_objs=int(setup_config["HIDE_SIGNAL"]),
        )
        + nl.GrowLabels(labels, radii=LABEL_RADII)
        + nl.GrowLabels(raw, radii=RAW_RADII)
        + LabelToFloat32(raw, intensities=RAW_INTENSITIES)
        + gp.NoiseAugment(raw, var=NOISE_VAR)
        + gp.PreCache(cache_size=CACHE_SIZE, num_workers=NUM_WORKERS)
        + gp.tensorflow.Train(
            "train_net",
            optimizer=create_custom_loss(tensor_names, setup_config),
            loss=None,
            inputs={tensor_names["raw"]: raw, tensor_names["gt_labels"]: labels},
            outputs={
                tensor_names["embedding"]: embedding,
                tensor_names["fg"]: fg,
                "strided_slice_1:0": maxima,
                "gt_fg:0": gt_fg,
                TENSOR_NAMES["emst"]: emst,
                TENSOR_NAMES["edges_u"]: edges_u,
                TENSOR_NAMES["edges_v"]: edges_v,
                TENSOR_NAMES["ratio_pos"]: ratio_pos,
                TENSOR_NAMES["ratio_neg"]: ratio_neg,
                TENSOR_NAMES["dist"]: dist,
                TENSOR_NAMES["num_pos_pairs"]: num_pos_pairs,
                TENSOR_NAMES["num_neg_pairs"]: num_neg_pairs,
            },
            gradients={
                tensor_names["embedding"]: gradient_embedding,
                tensor_names["fg"]: gradient_fg,
            },
            save_every=CHECKPOINT_EVERY,
            summary="Merge/MergeSummary:0",
            log_dir="tensorflow_logs",
        )
        + gp.Snapshot(
            output_filename="{iteration}.hdf",
            dataset_names={
                raw: "volumes/raw",
                labels: "volumes/labels",
                point_trees: "point_trees",
                embedding: "volumes/embedding",
                fg: "volumes/fg",
                maxima: "volumes/maxima",
                gt_fg: "volumes/gt_fg",
                gradient_embedding: "volumes/gradient_embedding",
                gradient_fg: "volumes/gradient_fg",
                emst: "emst",
                edges_u: "edges_u",
                edges_v: "edges_v",
                ratio_pos: "ratio_pos",
                ratio_neg: "ratio_neg",
                dist: "dist",
                num_pos_pairs: "num_pos_pairs",
                num_neg_pairs: "num_neg_pairs",
            },
            dataset_dtypes={maxima: np.float32, gt_fg: np.float32},
            every=SNAPSHOT_EVERY,
            additional_request=snapshot_request,
        )
        # + gp.PrintProfilingStats(every=100)
    )

    with gp.build(pipeline):
        for i in range(n_iterations + 1):
            pipeline.request_batch(request)
            request._update_random_seed()


def predict(n_iterations, setup_config, tensor_names, TENSOR_NAMES, checkpoint):

    # Network hyperparams
    INPUT_SHAPE = setup_config["INPUT_SHAPE"]
    OUTPUT_SHAPE = setup_config["OUTPUT_SHAPE"]

    # Skeleton generation hyperparams
    SKEL_GEN_RADIUS = setup_config["SKEL_GEN_RADIUS"]
    THETAS = np.array(setup_config["THETAS"]) * math.pi
    SPLIT_PS = setup_config["SPLIT_PS"]
    NOISE_VAR = setup_config["NOISE_VAR"]
    N_OBJS = setup_config["N_OBJS"]

    # Skeleton variation hyperparams
    LABEL_RADII = setup_config["LABEL_RADII"]
    RAW_RADII = setup_config["RAW_RADII"]
    RAW_INTENSITIES = setup_config["RAW_INTENSITIES"]

    # Training hyperparams
    CACHE_SIZE = setup_config["CACHE_SIZE"]
    NUM_WORKERS = setup_config["NUM_WORKERS"]
    SNAPSHOT_EVERY = 1

    point_trees = gp.PointsKey("POINT_TREES")
    labels = gp.ArrayKey("LABELS")
    raw = gp.ArrayKey("RAW")
    gt_fg = gp.ArrayKey("GT_FG")
    embedding = gp.ArrayKey("EMBEDDING")
    fg = gp.ArrayKey("FG")
    maxima = gp.ArrayKey("MAXIMA")
    gradient_embedding = gp.ArrayKey("GRADIENT_EMBEDDING")
    gradient_fg = gp.ArrayKey("GRADIENT_FG")

    # tensorflow tensors
    emst = gp.ArrayKey("EMST")
    edges_u = gp.ArrayKey("EDGES_U")
    edges_v = gp.ArrayKey("EDGES_V")
    ratio_pos = gp.ArrayKey("RATIO_POS")
    ratio_neg = gp.ArrayKey("RATIO_NEG")
    dist = gp.ArrayKey("DIST")
    num_pos_pairs = gp.ArrayKey("NUM_POS")
    num_neg_pairs = gp.ArrayKey("NUM_NEG")

    request = gp.BatchRequest()
    request.add(raw, INPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    request.add(labels, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    request.add(point_trees, INPUT_SHAPE)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, INPUT_SHAPE)
    snapshot_request.add(embedding, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(gt_fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(maxima, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(
        gradient_embedding, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1))
    )
    snapshot_request.add(gradient_fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request[emst] = gp.ArraySpec()
    snapshot_request[edges_u] = gp.ArraySpec()
    snapshot_request[edges_v] = gp.ArraySpec()
    snapshot_request[ratio_pos] = gp.ArraySpec()
    snapshot_request[ratio_neg] = gp.ArraySpec()
    snapshot_request[dist] = gp.ArraySpec()
    snapshot_request[num_pos_pairs] = gp.ArraySpec()
    snapshot_request[num_neg_pairs] = gp.ArraySpec()

    pipeline = (
        nl.SyntheticLightLike(
            point_trees,
            dims=2,
            r=SKEL_GEN_RADIUS,
            n_obj=N_OBJS,
            thetas=THETAS,
            split_ps=SPLIT_PS,
        )
        # + gp.SimpleAugment()
        # + gp.ElasticAugment([10, 10], [0.1, 0.1], [0, 2.0 * math.pi], spatial_dims=2)
        + nl.RasterizeSkeleton(
            point_trees,
            raw,
            gp.ArraySpec(
                roi=gp.Roi((None,) * 2, (None,) * 2),
                voxel_size=gp.Coordinate((1, 1)),
                dtype=np.uint64,
            ),
        )
        + nl.RasterizeSkeleton(
            point_trees,
            labels,
            gp.ArraySpec(
                roi=gp.Roi((None,) * 2, (None,) * 2),
                voxel_size=gp.Coordinate((1, 1)),
                dtype=np.uint64,
            ),
            use_component=True,
            n_objs=int(setup_config["HIDE_SIGNAL"]),
        )
        + nl.GrowLabels(labels, radii=LABEL_RADII)
        + nl.GrowLabels(raw, radii=RAW_RADII)
        + LabelToFloat32(raw, intensities=RAW_INTENSITIES)
        + gp.NoiseAugment(raw, var=NOISE_VAR)
        + gp.PreCache(cache_size=CACHE_SIZE, num_workers=NUM_WORKERS)
        + gp.tensorflow.Predict(
            checkpoint=checkpoint,
            inputs={tensor_names["raw"]: raw, tensor_names["gt_labels"]: labels},
            outputs={
                tensor_names["embedding"]: embedding,
                tensor_names["fg"]: fg,
                "strided_slice_1:0": maxima,
                "gt_fg:0": gt_fg,
                TENSOR_NAMES["emst"]: emst,
                TENSOR_NAMES["edges_u"]: edges_u,
                TENSOR_NAMES["edges_v"]: edges_v,
                TENSOR_NAMES["ratio_pos"]: ratio_pos,
                TENSOR_NAMES["ratio_neg"]: ratio_neg,
                TENSOR_NAMES["dist"]: dist,
                TENSOR_NAMES["num_pos_pairs"]: num_pos_pairs,
                TENSOR_NAMES["num_neg_pairs"]: num_neg_pairs,
            },
        )
        + gp.Snapshot(
            output_filename="pred_{iteration}.hdf",
            dataset_names={
                raw: "volumes/raw",
                labels: "volumes/labels",
                point_trees: "point_trees",
                embedding: "volumes/embedding",
                fg: "volumes/fg",
                maxima: "volumes/maxima",
                gt_fg: "volumes/gt_fg",
                gradient_embedding: "volumes/gradient_embedding",
                gradient_fg: "volumes/gradient_fg",
                emst: "emst",
                edges_u: "edges_u",
                edges_v: "edges_v",
                ratio_pos: "ratio_pos",
                ratio_neg: "ratio_neg",
                dist: "dist",
                num_pos_pairs: "num_pos_pairs",
                num_neg_pairs: "num_neg_pairs",
            },
            dataset_dtypes={maxima: np.float32, gt_fg: np.float32},
            every=SNAPSHOT_EVERY,
            additional_request=snapshot_request,
        )
        # + gp.PrintProfilingStats(every=100)
    )

    batches = []
    with gp.build(pipeline):
        for i in range(n_iterations + 1):
            batches.append(pipeline.request_batch(request))
            request._update_random_seed()
    return batches
