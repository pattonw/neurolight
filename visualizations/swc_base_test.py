from .provider_test import TestWithTempFiles
from neurolight.gunpowder.nodes.swc_file_source import GraphPoint
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
        points: List[GraphPoint],
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
            GraphPoint(0, 0, Coordinate([0, 0, 5]), 0, 0),
            GraphPoint(1, 0, Coordinate([1, 0, 5]), 0, 0),
            GraphPoint(2, 0, Coordinate([2, 0, 5]), 0, 1),
            GraphPoint(3, 0, Coordinate([3, 0, 5]), 0, 2),
            GraphPoint(4, 0, Coordinate([4, 0, 5]), 0, 3),
            GraphPoint(5, 0, Coordinate([5, 0, 5]), 0, 4),
            GraphPoint(6, 0, Coordinate([6, 0, 5]), 0, 5),
            GraphPoint(7, 0, Coordinate([7, 0, 5]), 0, 6),
            GraphPoint(8, 0, Coordinate([8, 0, 5]), 0, 7),
            GraphPoint(9, 0, Coordinate([9, 0, 5]), 0, 8),
            GraphPoint(10, 0, Coordinate([10, 0, 5]), 0, 9),
            # bottom line
            GraphPoint(11, 0, Coordinate([0, 1, 5]), 0, 0),
            GraphPoint(12, 0, Coordinate([0, 2, 5]), 0, 11),
            GraphPoint(13, 0, Coordinate([0, 3, 5]), 0, 12),
            GraphPoint(14, 0, Coordinate([0, 4, 5]), 0, 13),
            GraphPoint(15, 0, Coordinate([0, 5, 5]), 0, 14),
            GraphPoint(16, 0, Coordinate([0, 6, 5]), 0, 15),
            GraphPoint(17, 0, Coordinate([0, 7, 5]), 0, 16),
            GraphPoint(18, 0, Coordinate([0, 8, 5]), 0, 17),
            GraphPoint(19, 0, Coordinate([0, 9, 5]), 0, 18),
            GraphPoint(20, 0, Coordinate([0, 10, 5]), 0, 19),
            # mid line
            GraphPoint(21, 0, Coordinate([5, 1, 5]), 0, 5),
            GraphPoint(22, 0, Coordinate([5, 2, 5]), 0, 21),
            GraphPoint(23, 0, Coordinate([5, 3, 5]), 0, 22),
            GraphPoint(24, 0, Coordinate([5, 4, 5]), 0, 23),
            GraphPoint(25, 0, Coordinate([5, 5, 5]), 0, 24),
            GraphPoint(26, 0, Coordinate([5, 6, 5]), 0, 25),
            GraphPoint(27, 0, Coordinate([5, 7, 5]), 0, 26),
            GraphPoint(28, 0, Coordinate([5, 8, 5]), 0, 27),
            GraphPoint(29, 0, Coordinate([5, 9, 5]), 0, 28),
            GraphPoint(30, 0, Coordinate([5, 10, 5]), 0, 29),
            # top line
            GraphPoint(31, 0, Coordinate([10, 1, 5]), 0, 10),
            GraphPoint(32, 0, Coordinate([10, 2, 5]), 0, 31),
            GraphPoint(33, 0, Coordinate([10, 3, 5]), 0, 32),
            GraphPoint(34, 0, Coordinate([10, 4, 5]), 0, 33),
            GraphPoint(35, 0, Coordinate([10, 5, 5]), 0, 34),
            GraphPoint(36, 0, Coordinate([10, 6, 5]), 0, 35),
            GraphPoint(37, 0, Coordinate([10, 7, 5]), 0, 36),
            GraphPoint(38, 0, Coordinate([10, 8, 5]), 0, 37),
            GraphPoint(39, 0, Coordinate([10, 9, 5]), 0, 38),
            GraphPoint(40, 0, Coordinate([10, 10, 5]), 0, 39),
        ]
        return points

    def _get_points(
        self, inside: np.ndarray, slope: np.ndarray, bb: Roi
    ) -> List[GraphPoint]:
        slope = slope / max(slope)
        shape = np.array(bb.get_shape())
        outside_down = inside - shape * slope
        outside_up = inside + shape * slope
        down_intercept = self._resample_relative(inside, outside_down, bb)
        up_intercept = self._resample_relative(inside, outside_up, bb)

        points = [
            # line
            GraphPoint(0, 0, down_intercept, 0, 0),
            GraphPoint(1, 0, up_intercept, 0, 0),
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
