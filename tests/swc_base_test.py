from .provider_test import TestWithTempFiles
from neurolight.gunpowder.swc_file_source import SwcPoint
from gunpowder import Roi, Coordinate

import numpy as np

from typing import Dict, List, Tuple, Optional
from pathlib import Path


class SWCBaseTest(TestWithTempFiles):
    def setUp(self):
        super(SWCBaseTest, self).setUp()

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

    def _toy_swc_points(self):
        """
        shape:

        -----------
        |
        |
        |----------
        |
        |
        -----------
        """
        points = [
            # backbone
            SwcPoint(0, 0, Coordinate([0, 0, 5]), 0, 0),
            SwcPoint(1, 0, Coordinate([1, 0, 5]), 0, 0),
            SwcPoint(2, 0, Coordinate([2, 0, 5]), 0, 1),
            SwcPoint(3, 0, Coordinate([3, 0, 5]), 0, 2),
            SwcPoint(4, 0, Coordinate([4, 0, 5]), 0, 3),
            SwcPoint(5, 0, Coordinate([5, 0, 5]), 0, 4),
            SwcPoint(6, 0, Coordinate([6, 0, 5]), 0, 5),
            SwcPoint(7, 0, Coordinate([7, 0, 5]), 0, 6),
            SwcPoint(8, 0, Coordinate([8, 0, 5]), 0, 7),
            SwcPoint(9, 0, Coordinate([9, 0, 5]), 0, 8),
            SwcPoint(10, 0, Coordinate([10, 0, 5]), 0, 9),
            # bottom line
            SwcPoint(11, 0, Coordinate([0, 1, 5]), 0, 0),
            SwcPoint(12, 0, Coordinate([0, 2, 5]), 0, 11),
            SwcPoint(13, 0, Coordinate([0, 3, 5]), 0, 12),
            SwcPoint(14, 0, Coordinate([0, 4, 5]), 0, 13),
            SwcPoint(15, 0, Coordinate([0, 5, 5]), 0, 14),
            SwcPoint(16, 0, Coordinate([0, 6, 5]), 0, 15),
            SwcPoint(17, 0, Coordinate([0, 7, 5]), 0, 16),
            SwcPoint(18, 0, Coordinate([0, 8, 5]), 0, 17),
            SwcPoint(19, 0, Coordinate([0, 9, 5]), 0, 18),
            SwcPoint(20, 0, Coordinate([0, 10, 5]), 0, 19),
            # mid line
            SwcPoint(21, 0, Coordinate([5, 1, 5]), 0, 5),
            SwcPoint(22, 0, Coordinate([5, 2, 5]), 0, 21),
            SwcPoint(23, 0, Coordinate([5, 3, 5]), 0, 22),
            SwcPoint(24, 0, Coordinate([5, 4, 5]), 0, 23),
            SwcPoint(25, 0, Coordinate([5, 5, 5]), 0, 24),
            SwcPoint(26, 0, Coordinate([5, 6, 5]), 0, 25),
            SwcPoint(27, 0, Coordinate([5, 7, 5]), 0, 26),
            SwcPoint(28, 0, Coordinate([5, 8, 5]), 0, 27),
            SwcPoint(29, 0, Coordinate([5, 9, 5]), 0, 28),
            SwcPoint(30, 0, Coordinate([5, 10, 5]), 0, 29),
            # top line
            SwcPoint(31, 0, Coordinate([10, 1, 5]), 0, 10),
            SwcPoint(32, 0, Coordinate([10, 2, 5]), 0, 31),
            SwcPoint(33, 0, Coordinate([10, 3, 5]), 0, 32),
            SwcPoint(34, 0, Coordinate([10, 4, 5]), 0, 33),
            SwcPoint(35, 0, Coordinate([10, 5, 5]), 0, 34),
            SwcPoint(36, 0, Coordinate([10, 6, 5]), 0, 35),
            SwcPoint(37, 0, Coordinate([10, 7, 5]), 0, 36),
            SwcPoint(38, 0, Coordinate([10, 8, 5]), 0, 37),
            SwcPoint(39, 0, Coordinate([10, 9, 5]), 0, 38),
            SwcPoint(40, 0, Coordinate([10, 10, 5]), 0, 39),
        ]
        return points

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
