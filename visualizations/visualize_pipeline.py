from pathlib import Path

from neurolight.gunpowder.nodes.mouselight_swc_file_source import (
    MouselightSwcFileSource,
)
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

logging.basicConfig(level=logging.DEBUG)
logging.getLogger(gp.nodes.random_location.__name__).setLevel(logging.INFO)


SEPERATE_DISTANCE = int(sys.argv[1])
SEPERATE_DISTANCE = [SEPERATE_DISTANCE * 0.7, SEPERATE_DISTANCE * 1.3]


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
swcs = gp.PointsKey("SWCS")
labels = gp.ArrayKey("LABELS")

# array keys for base volume
raw_base = gp.ArrayKey("RAW_BASE")
labels_base = gp.ArrayKey("LABELS_BASE")
swc_base = gp.PointsKey("SWC_BASE")

# array keys for add volume
raw_add = gp.ArrayKey("RAW_ADD")
labels_add = gp.ArrayKey("LABELS_ADD")
swc_add = gp.PointsKey("SWC_ADD")

# array keys for fused volume
raw_fused = gp.ArrayKey("RAW_FUSED")
labels_fused = gp.ArrayKey("LABELS_FUSED")
swc_fused = gp.PointsKey("SWC_FUSED")

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
request.add(swc_fused, input_size)
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
request.add(swc_base, input_size)
request.add(swc_add, input_size)

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
        MouselightSwcFileSource(
            filename=str(
                Path(
                    "/groups/mousebrainmicro/mousebrainmicro/scripts/carver/2018-07-02-v03/augmented-with-skeleton-nodes-as-swcs/G-002.swc"
                ).absolute()
            ),
            points=(swcs,),
            scale=voxel_size,
            transpose=(2, 1, 0),
            transform_file=str((filename / "transform.txt").absolute()),
        ),
    )
    + gp.MergeProvider()
    + gp.RandomLocation(
        ensure_nonempty=swcs, ensure_centered=True, balance_points=False
    )
    + RasterizeSkeleton(
        points=swcs,
        array=labels,
        array_spec=gp.ArraySpec(
            interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
        ),
    )
    + GrowLabels(labels, radii=[10])
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
        swcs,
        raw,
        labels,
        (swc_base, swc_add),
        (raw_base, raw_add),
        (labels_base, labels_add),
        seperate_by=SEPERATE_DISTANCE,
        shift_attempts=50,
        request_attempts=10,
    )
    + FusionAugment(
        raw_base,
        raw_add,
        labels_base,
        labels_add,
        swc_base,
        swc_add,
        raw_fused,
        labels_fused,
        swc_fused,
        blend_mode="labels_mask",
        blend_smoothness=10,
        num_blended_objects=0,
    )
    + Crop(labels_fused, labels_fg)
    + BinarizeGt(labels_fg, labels_fg_bin)
    + gp.BalanceLabels(labels_fg_bin, loss_weights)
    + gp.PrintProfilingStats()
    + gp.Snapshot(
        output_filename="snapshot_{}_{}.hdf".format(SEPERATE_DISTANCE, "{iteration}"),
        dataset_names={
            raw_fused: "volumes/raw_fused",
            raw_base: "volumes/raw_base",
            raw_add: "volumes/raw_add",
            labels_fused: "volumes/labels_fused",
            labels_base: "volumes/labels_base",
            labels_add: "volumes/labels_add",
            labels_fg_bin: "volumes/labels_fg_bin",
        },
        every=1,
    )
)

request = BatchRequest()

# add request
request = gp.BatchRequest()
request.add(raw_fused, input_size)
request.add(labels_fused, input_size)
request.add(swc_fused, input_size)
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
request.add(swc_base, input_size)
request.add(swc_add, input_size)

with build(pipeline):
    t1 = time.time()
    for _ in range(1):
        pipeline.request_batch(request)
