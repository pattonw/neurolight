from pathlib import Path

import neurolight
from neurolight.gunpowder.nodes.graph_source import GraphSource
from neurolight.gunpowder.nodes.topological_graph_matching import TopologicalMatcher
from neurolight.gunpowder.nodes.grow_labels import GrowLabels
from neurolight.gunpowder.nodes.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.nodes.get_neuron_pair import GetNeuronPair
from neurolight.gunpowder.nodes.fusion_augment import FusionAugment
import gunpowder as gp
from gunpowder import BatchRequest, build, Coordinate
import math
import copy
import time
import logging

import numpy as np

import sys

logging.basicConfig(level=logging.INFO)
logging.getLogger(gp.nodes.random_location.__name__).setLevel(logging.INFO)
logging.getLogger(neurolight.gunpowder.nodes.rasterize_skeleton.__name__).setLevel(
    logging.DEBUG
)


SEP_DIST = int(sys.argv[1])
SEPERATE_DISTANCE = [SEP_DIST * 0.9, SEP_DIST * 1.1]


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


input_size = Coordinate([74, 260, 260])
output_size = Coordinate([42, 168, 168])
path_to_data = Path("/nrs/funke/mouselight-v2")

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

# array keys for add volume
raw_add = gp.ArrayKey("RAW_ADD")
labels_add = gp.ArrayKey("LABELS_ADD")
matched_add = gp.PointsKey("MATCHED_ADD")

# array keys for fused volume
raw_fused = gp.ArrayKey("RAW_FUSED")
labels_fused = gp.ArrayKey("LABELS_FUSED")
matched_fused = gp.PointsKey("MATCHED_FUSED")

# output data
labels_fg = gp.ArrayKey("LABELS_FG")
labels_fg_bin = gp.ArrayKey("LABELS_FG_BIN")

loss_weights = gp.ArrayKey("LOSS_WEIGHTS")

voxel_size = gp.Coordinate((10, 3, 3))
input_size = input_size * voxel_size
output_size = output_size * voxel_size

# add request
request = gp.BatchRequest()
request.add(raw_fused, input_size)
request.add(labels_fused, input_size)
request.add(matched_fused, input_size)
request.add(labels_fg, output_size)
request.add(labels_fg_bin, output_size)
request.add(loss_weights, output_size)

# add snapshot request
# request.add(fg, output_size)
# request.add(labels_fg, output_size)
# request.add(gradient_fg, output_size)
request.add(raw_base, input_size)
request.add(raw_add, input_size)
request.add(labels_base, input_size)
request.add(labels_add, input_size)
request.add(matched_base, input_size)
request.add(matched_add, input_size)

data_sources = tuple(
    (
        gp.N5Source(
            filename=str(
                (
                    filename
                    / "consensus-neurons-with-machine-centerpoints-labelled-as-swcs-carved.n5"
                ).absolute()
            ),
            datasets={raw: "volume"},
            array_specs={
                raw: gp.ArraySpec(
                    interpolatable=True, voxel_size=voxel_size, dtype=np.uint16
                )
            },
        ),
        GraphSource(
            filename=str(
                Path(
                    "/groups/mousebrainmicro/home/pattonw/Code/Packages/neurolight/visualizations/skeletonization_carved-002-100.obj"
                ).absolute()
            ),
            points=(skeletonization,),
            scale=voxel_size,
            transpose=(2, 1, 0),
        ),
        GraphSource(
            filename=str(
                Path(
                    "/groups/mousebrainmicro/home/pattonw/Code/Packages/neurolight/visualizations/consensus-002.obj"
                ).absolute()
            ),
            points=(consensus, nonempty_placeholder),
            scale=voxel_size,
            transpose=(2, 1, 0),
        ),
    )
    + gp.MergeProvider()
    + gp.RandomLocation(
        ensure_nonempty=nonempty_placeholder,
        ensure_centered=True,
        point_balance_radius=700,
    )
    + TopologicalMatcher(skeletonization, consensus, matched)
    + RasterizeSkeleton(
        points=matched,
        array=labels,
        array_spec=gp.ArraySpec(
            interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
        ),
    )
    + GrowLabels(labels, radii=[30])
    # augment
    + gp.ElasticAugment(
        [40, 10, 10],
        [0.25, 1, 1],
        [0, math.pi / 2.0],
        subsample=4,
        use_fast_points_transform=True,
        recompute_missing_points=False,
    )
    + gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2])
    + gp.Normalize(raw)
    + gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001)
    for filename in path_to_data.iterdir()
    if "2018-07-02" in filename.name
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
        seperate_by=SEPERATE_DISTANCE,
        shift_attempts=int(SEPERATE_DISTANCE[1]) * 2,
        request_attempts=10,
        nonempty_placeholder=nonempty_placeholder,
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
        blend_mode="labels_mask",
        blend_smoothness=10,
        num_blended_objects=0,
    )
    + Crop(labels_fused, labels_fg)
    + BinarizeGt(labels_fg, labels_fg_bin)
    + gp.BalanceLabels(labels_fg_bin, loss_weights)
    + gp.PrintProfilingStats(every=3)
    + gp.Snapshot(
        output_filename="snapshot_{}_{}.hdf".format(SEP_DIST, "{id}"),
        dataset_names={
            raw_fused: "volumes/raw_fused",
            raw_base: "volumes/raw_base",
            raw_add: "volumes/raw_add",
            labels_fused: "volumes/labels_fused",
            labels_base: "volumes/labels_base",
            labels_add: "volumes/labels_add",
            labels_fg_bin: "volumes/labels_fg_bin",
            matched_base: "matched_base",
            matched_add: "matched_add",
        },
        every=1,
    )
)

request = BatchRequest()

# add request
request = gp.BatchRequest()
request.add(raw_fused, input_size)
request.add(labels_fused, input_size)
request.add(matched_fused, input_size)
request.add(labels_fg, output_size)
request.add(labels_fg_bin, output_size)
request.add(loss_weights, output_size)

# add snapshot request
# request.add(fg, output_size)
# request.add(labels_fg, output_size)
# request.add(gradient_fg, output_size)
request.add(raw_base, input_size)
request.add(raw_add, input_size)
request.add(labels_base, input_size)
request.add(labels_add, input_size)
request.add(matched_base, input_size)
request.add(matched_add, input_size)

with build(pipeline):
    t1 = time.time()
    for _ in range(3):
        pipeline.request_batch(request)
