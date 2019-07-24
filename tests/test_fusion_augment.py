from .swc_base_test import SWCBaseTest
from neurolight.gunpowder.swc_file_source import SwcFileSource, SwcPoint
from neurolight.gunpowder.fusion_augment import FusionAugment
from neurolight.gunpowder.rasterize_skeleton import RasterizeSkeleton
from gunpowder import (
    PointsKey,
    PointsSpec,
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Roi,
    build,
    Coordinate,
    MergeProvider,
)

import numpy as np

from typing import Dict, List, Tuple, Optional
from pathlib import Path


class FusionAugmentTest(SWCBaseTest):
    def setUp(self):
        super(FusionAugmentTest, self).setUp()

    def test_two_disjoint_lines_intensity(self):
        LABEL_RADIUS = 3
        RAW_RADIUS = 3
        # exagerated to show problem
        BLEND_SMOOTHNESS = 10

        bb = Roi(Coordinate([0, 0, 0]), ([256, 256, 256]))
        voxel_size = Coordinate([1, 1, 1])
        swc_files = ("test_line_a.swc", "test_line_b.swc")
        swc_paths = tuple(Path(self.path_to(file_name)) for file_name in swc_files)

        # create two lines seperated by a given distance and write them to swc files
        intercepts, slopes = self._get_line_pair(roi=bb, dist=3 * LABEL_RADIUS)
        for intercept, slope, swc_path in zip(intercepts, slopes, swc_paths):
            swc_points = self._get_points(intercept, slope, bb)
            self._write_swc(swc_path, swc_points)

        # create swc sources
        fused = ArrayKey("FUSED")
        fused_labels = ArrayKey("FUSED_LABELS")
        fused_swc = PointsKey("FUSED_SWC")
        swc_key_names = ("SWC_A", "SWC_B")
        labels_key_names = ("LABELS_A", "LABELS_B")
        raw_key_names = ("RAW_A", "RAW_B")

        swc_keys = tuple(PointsKey(name) for name in swc_key_names)
        labels_keys = tuple(ArrayKey(name) for name in labels_key_names)
        raw_keys = tuple(ArrayKey(name) for name in raw_key_names)

        # add request
        request = BatchRequest()
        request.add(fused, bb.get_shape())
        request.add(fused_labels, bb.get_shape())
        request.add(fused_swc, bb.get_shape())
        request.add(labels_keys[0], bb.get_shape())
        request.add(labels_keys[1], bb.get_shape())
        request.add(raw_keys[0], bb.get_shape())
        request.add(raw_keys[1], bb.get_shape())
        request.add(swc_keys[0], bb.get_shape())
        request.add(swc_keys[1], bb.get_shape())

        # data source for swc a
        data_sources_a = tuple()
        data_sources_a = (
            data_sources_a
            + SwcFileSource(swc_paths[0], [swc_keys[0]], [PointsSpec(roi=bb)])
            + RasterizeSkeleton(
                points=swc_keys[0],
                array=labels_keys[0],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=LABEL_RADIUS,
            )
            + RasterizeSkeleton(
                points=swc_keys[0],
                array=raw_keys[0],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=RAW_RADIUS,
            )
        )

        # data source for swc b
        data_sources_b = tuple()
        data_sources_b = (
            data_sources_b
            + SwcFileSource(swc_paths[1], [swc_keys[1]], [PointsSpec(roi=bb)])
            + RasterizeSkeleton(
                points=swc_keys[1],
                array=labels_keys[1],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=LABEL_RADIUS,
            )
            + RasterizeSkeleton(
                points=swc_keys[1],
                array=raw_keys[1],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=RAW_RADIUS,
            )
        )
        data_sources = tuple([data_sources_a, data_sources_b]) + MergeProvider()

        pipeline = data_sources + FusionAugment(
            raw_keys[0],
            raw_keys[1],
            labels_keys[0],
            labels_keys[1],
            swc_keys[0],
            swc_keys[1],
            fused,
            fused_labels,
            fused_swc,
            blend_mode="intensity",
            blend_smoothness=BLEND_SMOOTHNESS,
            num_blended_objects=0,
        )

        with build(pipeline):
            batch = pipeline.request_batch(request)

        fused_data = batch[fused].data
        fused_data = np.pad(fused_data, (1,), "constant", constant_values=(0,))

        a_data = batch[raw_keys[0]].data
        a_data = np.pad(a_data, (1,), "constant", constant_values=(0,))

        b_data = batch[raw_keys[1]].data
        b_data = np.pad(b_data, (1,), "constant", constant_values=(0,))

        diff = np.linalg.norm(fused_data - a_data - b_data)
        self.assertAlmostEqual(diff, 0)

    def test_two_disjoint_lines_softmask(self):
        LABEL_RADIUS = 3
        RAW_RADIUS = 3
        # exagerated to show problem
        BLEND_SMOOTHNESS = 10

        bb = Roi(Coordinate([0, 0, 0]), ([256, 256, 256]))
        voxel_size = Coordinate([1, 1, 1])
        swc_files = ("test_line_a.swc", "test_line_b.swc")
        swc_paths = tuple(Path(self.path_to(file_name)) for file_name in swc_files)

        # create two lines seperated by a given distance and write them to swc files
        intercepts, slopes = self._get_line_pair(roi=bb, dist=3 * LABEL_RADIUS)
        for intercept, slope, swc_path in zip(intercepts, slopes, swc_paths):
            swc_points = self._get_points(intercept, slope, bb)
            self._write_swc(swc_path, swc_points)

        # create swc sources
        fused = ArrayKey("FUSED")
        fused_labels = ArrayKey("FUSED_LABELS")
        fused_swc = PointsKey("FUSED_SWC")
        swc_key_names = ("SWC_A", "SWC_B")
        labels_key_names = ("LABELS_A", "LABELS_B")
        raw_key_names = ("RAW_A", "RAW_B")

        swc_keys = tuple(PointsKey(name) for name in swc_key_names)
        labels_keys = tuple(ArrayKey(name) for name in labels_key_names)
        raw_keys = tuple(ArrayKey(name) for name in raw_key_names)

        # add request
        request = BatchRequest()
        request.add(fused, bb.get_shape())
        request.add(fused_labels, bb.get_shape())
        request.add(fused_swc, bb.get_shape())
        request.add(labels_keys[0], bb.get_shape())
        request.add(labels_keys[1], bb.get_shape())
        request.add(raw_keys[0], bb.get_shape())
        request.add(raw_keys[1], bb.get_shape())
        request.add(swc_keys[0], bb.get_shape())
        request.add(swc_keys[1], bb.get_shape())

        # data source for swc a
        data_sources_a = tuple()
        data_sources_a = (
            data_sources_a
            + SwcFileSource(swc_paths[0], [swc_keys[0]], [PointsSpec(roi=bb)])
            + RasterizeSkeleton(
                points=swc_keys[0],
                array=labels_keys[0],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=LABEL_RADIUS,
            )
            + RasterizeSkeleton(
                points=swc_keys[0],
                array=raw_keys[0],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=RAW_RADIUS,
            )
        )

        # data source for swc b
        data_sources_b = tuple()
        data_sources_b = (
            data_sources_b
            + SwcFileSource(swc_paths[1], [swc_keys[1]], [PointsSpec(roi=bb)])
            + RasterizeSkeleton(
                points=swc_keys[1],
                array=labels_keys[1],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=LABEL_RADIUS,
            )
            + RasterizeSkeleton(
                points=swc_keys[1],
                array=raw_keys[1],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=RAW_RADIUS,
            )
        )
        data_sources = tuple([data_sources_a, data_sources_b]) + MergeProvider()

        pipeline = data_sources + FusionAugment(
            raw_keys[0],
            raw_keys[1],
            labels_keys[0],
            labels_keys[1],
            swc_keys[0],
            swc_keys[1],
            fused,
            fused_labels,
            fused_swc,
            blend_mode="labels_mask",
            blend_smoothness=BLEND_SMOOTHNESS,
            num_blended_objects=0,
        )

        with build(pipeline):
            batch = pipeline.request_batch(request)

        fused_data = batch[fused].data
        fused_data = np.pad(fused_data, (1,), "constant", constant_values=(0,))

        a_data = batch[raw_keys[0]].data
        a_data = np.pad(a_data, (1,), "constant", constant_values=(0,))

        b_data = batch[raw_keys[1]].data
        b_data = np.pad(b_data, (1,), "constant", constant_values=(0,))

        all_data = np.zeros((5,) + fused_data.shape)
        all_data[0, :, :, :] = fused_data
        all_data[1, :, :, :] = a_data + b_data
        all_data[2, :, :, :] = fused_data - a_data - b_data
        all_data[3, :, :, :] = a_data
        all_data[4, :, :, :] = b_data

        diff = np.linalg.norm(fused_data - a_data - b_data)
        self.assertAlmostEqual(diff, 0)

