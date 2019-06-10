from .provider_test import TestWithTempFiles
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
from spimagine import volshow

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
import unittest


class FusionAugmentTest(TestWithTempFiles):
    def setUp(self):
        super(FusionAugmentTest, self).setUp()

    def _write_swc(
        self,
        file_path: Path,
        points: List[SwcPoint],
        constants: Dict[str, Coordinate] = {},
    ):
        swc = ""
        for key, shape in constants.items():
            swc += "# {} {}\n".format(key.upper(), " ".join([str(x) for x in shape]))
        swc += "\n".join(
            [
                "{} {} {} {} {} {} {}".format(
                    p.point_id, p.point_type, *p.location, p.radius, p.parent_id
                )
                for p in points
            ]
        )
        with file_path.open("w") as f:
            f.write(swc)

    def _get_points(
        self, inside: np.ndarray, slope: np.ndarray, bb: Roi
    ) -> List[SwcPoint]:
        slope = slope / max(slope)
        shape = np.array(bb.get_shape())
        outside_down = inside - shape * slope
        outside_up = inside + shape * slope
        down_intercept = self._resample_relative(inside, outside_down, bb)
        up_intercept = self._resample_relative(inside, outside_up, bb)

        points = [
            # line
            SwcPoint(0, 0, down_intercept, 0, 0),
            SwcPoint(1, 0, up_intercept, 0, 0),
        ]
        return points

    def _resample_relative(
        self, inside: np.ndarray, outside: np.ndarray, bb: Roi
    ) -> Optional[np.ndarray]:
        offset = outside - inside
        # get_end() is not contained in the Roi. We want the point to be included,
        # thus we decriment by 1. Technically we only need to decriment by 0.000001,
        # but that is not possible using Roi's and Coordinates. Should we change this?
        bb_x = np.asarray(
            [
                (np.asarray(bb.get_begin()) - inside) / offset,
                (np.asarray(bb.get_end() - Coordinate([1, 1, 1])) - inside) / offset,
            ]
        )

        if np.sum(np.logical_and((bb_x > 0), (bb_x <= 1))) > 0:
            s = np.min(bb_x[np.logical_and((bb_x > 0), (bb_x <= 1))])
            return np.array(inside) + s * offset
        else:
            return None

    def _get_line_pair(
        self,
        roi: Roi = Roi(Coordinate([0, 0, 0]), Coordinate([10, 10, 10])),
        dist: float = 3,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        bb_size = np.array(roi.get_shape()) - Coordinate([1, 1, 1])
        pad = min(dist / np.array(bb_size))
        center = np.random.random((3,)).clip(pad, 1 - pad) * (bb_size)
        slope = np.random.random((3,))
        slope /= np.linalg.norm(slope)

        intercepts = (center + slope * dist / 2, center - slope * dist / 2)
        slope_a = np.random.random(3)
        slope_a -= np.dot(slope_a, slope) * slope
        slope_a /= np.linalg.norm(slope_a)
        slope_b = np.cross(slope_a, slope)

        return (intercepts, (slope_a, slope_b))

    @unittest.expectedFailure
    def test_rasterize_speed(self):
        # This is worryingly slow for such a small volume (256**3) and only 2
        # straight lines for skeletons.

        LABEL_RADIUS = 3

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
        swc_key_names = ("SWC_A", "SWC_B")
        labels_key_names = ("LABELS_A", "LABELS_B")

        swc_keys = tuple(PointsKey(name) for name in swc_key_names)
        labels_keys = tuple(ArrayKey(name) for name in labels_key_names)

        # add request
        request = BatchRequest()
        request.add(labels_keys[0], bb.get_shape())
        request.add(labels_keys[1], bb.get_shape())
        request.add(swc_keys[0], bb.get_shape())
        request.add(swc_keys[1], bb.get_shape())

        # data source for swc a
        data_sources_a = tuple()
        data_sources_a = (
            data_sources_a
            + SwcFileSource(swc_paths[0], swc_keys[0], PointsSpec(roi=bb))
            + RasterizeSkeleton(
                points=swc_keys[0],
                array=labels_keys[0],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=LABEL_RADIUS,
            )
        )

        # data source for swc b
        data_sources_b = tuple()
        data_sources_b = (
            data_sources_b
            + SwcFileSource(swc_paths[1], swc_keys[1], PointsSpec(roi=bb))
            + RasterizeSkeleton(
                points=swc_keys[1],
                array=labels_keys[1],
                array_spec=ArraySpec(
                    interpolatable=False, dtype=np.uint32, voxel_size=voxel_size
                ),
                radius=LABEL_RADIUS,
            )
        )
        data_sources = tuple([data_sources_a, data_sources_b]) + MergeProvider()

        pipeline = data_sources

        t1 = time.time()
        with build(pipeline):
            batch = pipeline.request_batch(request)

        a_data = batch[labels_keys[0]].data
        a_data = np.pad(a_data, (1,), "constant", constant_values=(0,))

        b_data = batch[labels_keys[1]].data
        b_data = np.pad(b_data, (1,), "constant", constant_values=(0,))

        t2 = time.time()
        self.assertLess(t2 - t1, 0.1)

