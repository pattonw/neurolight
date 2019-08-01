from .swc_base_test import SWCBaseTest
from neurolight.gunpowder.swc_file_source import SwcFileSource
from neurolight.gunpowder.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.grow_labels import GrowLabels
from gunpowder import (
    PointsSpec,
    ArraySpec,
    BatchRequest,
    Roi,
    build,
    Coordinate,
    PointsKey,
    ArrayKey,
)

import numpy as np

import time
import unittest
from pathlib import Path


class GrowLabelsTest(SWCBaseTest):
    def setUp(self):
        super(GrowLabelsTest, self).setUp()

    def test_grow_labels(self):

        bb = Roi(Coordinate([0, 0, 0]), ([256, 256, 256]))
        voxel_size = Coordinate([1, 1, 1])
        swc_file = "test_swc.swc"
        swc_path = Path(self.path_to(swc_file))

        swc_points = self._get_points(np.array([1, 1, 1]), np.array([1, 1, 1]), bb)
        self._write_swc(swc_path, swc_points.graph)

        # create swc sources
        swc_key = PointsKey("SWC")
        labels_key = ArrayKey("LABELS")

        # add request
        request = BatchRequest()
        request.add(labels_key, bb.get_shape())
        request.add(swc_key, bb.get_shape())

        # data source for swc a
        data_source = tuple()
        data_source = (
            data_source
            + SwcFileSource(swc_path, [swc_key], [PointsSpec(roi=bb)])
            + RasterizeSkeleton(
                points=swc_key,
                array=labels_key,
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
            )
            + GrowLabels(array=labels_key, radius=3)
        )

        pipeline = data_source

        with build(pipeline):
            batch = pipeline.request_batch(request)

        self.assertIn(labels_key, batch)

    @unittest.expectedFailure
    def test_grow_labels_speed(self):

        bb = Roi(Coordinate([0, 0, 0]), ([256, 256, 256]))
        voxel_size = Coordinate([1, 1, 1])
        swc_file = "test_swc.swc"
        swc_path = Path(self.path_to(swc_file))

        swc_points = self._get_points(np.array([1, 1, 1]), np.array([1, 1, 1]), bb)
        self._write_swc(swc_path, swc_points.graph)

        # create swc sources
        swc_key = PointsKey("SWC")
        labels_key = ArrayKey("LABELS")

        # add request
        request = BatchRequest()
        request.add(labels_key, bb.get_shape())
        request.add(swc_key, bb.get_shape())

        # data source for swc a
        data_source = tuple()
        data_source = (
            data_source
            + SwcFileSource(swc_path, [swc_key], [PointsSpec(roi=bb)])
            + RasterizeSkeleton(
                points=swc_key,
                array=labels_key,
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
            )
            + GrowLabels(array=labels_key, radius=3)
        )

        pipeline = data_source

        num_repeats = 10
        t1 = time.time()
        with build(pipeline):
            for i in range(num_repeats):
                pipeline.request_batch(request)

        t2 = time.time()
        self.assertLess((t2 - t1) / num_repeats, 0.1)

