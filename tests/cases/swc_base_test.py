from .provider_test import TestWithTempFiles
from gunpowder import Roi, Coordinate, GraphPoint, GraphPoints, PointsSpec

import numpy as np
import networkx as nx

from typing import Dict, List, Tuple, Optional
from pathlib import Path


class SWCBaseTest(TestWithTempFiles):
    def setUp(self):
        super(SWCBaseTest, self).setUp()

    def _write_swc(
        self, file_path: Path, graph: nx.DiGraph, constants: Dict[str, Coordinate] = {}
    ):
        def get_parent(g, u):
            preds = list(g.predecessors(u))
            if len(preds) == 1:
                return preds[0]
            elif len(preds) == 0:
                return u
            else:
                raise ValueError("Node cannot have 2 parents")

        swc = ""
        for key, shape in constants.items():
            swc += "# {} {}\n".format(key.upper(), " ".join([str(x) for x in shape]))
        swc += "\n".join(
            [
                "{} {} {} {} {} {} {}".format(
                    point_id,
                    point_attrs["node_type"],
                    *point_attrs["location"],
                    point_attrs["radius"],
                    get_parent(graph, point_id)
                )
                for point_id, point_attrs in graph.nodes.items()
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
        points = {
            # backbone
            0: GraphPoint(Coordinate([0, 0, 5]), radius=0, node_type=0),
            1: GraphPoint(Coordinate([1, 0, 5]), radius=0, node_type=0),
            2: GraphPoint(Coordinate([2, 0, 5]), radius=0, node_type=0),
            3: GraphPoint(Coordinate([3, 0, 5]), radius=0, node_type=0),
            4: GraphPoint(Coordinate([4, 0, 5]), radius=0, node_type=0),
            5: GraphPoint(Coordinate([5, 0, 5]), radius=0, node_type=0),
            6: GraphPoint(Coordinate([6, 0, 5]), radius=0, node_type=0),
            7: GraphPoint(Coordinate([7, 0, 5]), radius=0, node_type=0),
            8: GraphPoint(Coordinate([8, 0, 5]), radius=0, node_type=0),
            9: GraphPoint(Coordinate([9, 0, 5]), radius=0, node_type=0),
            10: GraphPoint(Coordinate([10, 0, 5]), radius=0, node_type=0),
            # bottom line
            11: GraphPoint(Coordinate([0, 1, 5]), radius=0, node_type=0),
            12: GraphPoint(Coordinate([0, 2, 5]), radius=0, node_type=0),
            13: GraphPoint(Coordinate([0, 3, 5]), radius=0, node_type=0),
            14: GraphPoint(Coordinate([0, 4, 5]), radius=0, node_type=0),
            15: GraphPoint(Coordinate([0, 5, 5]), radius=0, node_type=0),
            16: GraphPoint(Coordinate([0, 6, 5]), radius=0, node_type=0),
            17: GraphPoint(Coordinate([0, 7, 5]), radius=0, node_type=0),
            18: GraphPoint(Coordinate([0, 8, 5]), radius=0, node_type=0),
            19: GraphPoint(Coordinate([0, 9, 5]), radius=0, node_type=0),
            20: GraphPoint(Coordinate([0, 10, 5]), radius=0, node_type=0),
            # mid line
            21: GraphPoint(Coordinate([5, 1, 5]), radius=0, node_type=0),
            22: GraphPoint(Coordinate([5, 2, 5]), radius=0, node_type=0),
            23: GraphPoint(Coordinate([5, 3, 5]), radius=0, node_type=0),
            24: GraphPoint(Coordinate([5, 4, 5]), radius=0, node_type=0),
            25: GraphPoint(Coordinate([5, 5, 5]), radius=0, node_type=0),
            26: GraphPoint(Coordinate([5, 6, 5]), radius=0, node_type=0),
            27: GraphPoint(Coordinate([5, 7, 5]), radius=0, node_type=0),
            28: GraphPoint(Coordinate([5, 8, 5]), radius=0, node_type=0),
            29: GraphPoint(Coordinate([5, 9, 5]), radius=0, node_type=0),
            30: GraphPoint(Coordinate([5, 10, 5]), radius=0, node_type=0),
            # top line
            31: GraphPoint(Coordinate([10, 1, 5]), radius=0, node_type=0),
            32: GraphPoint(Coordinate([10, 2, 5]), radius=0, node_type=0),
            33: GraphPoint(Coordinate([10, 3, 5]), radius=0, node_type=0),
            34: GraphPoint(Coordinate([10, 4, 5]), radius=0, node_type=0),
            35: GraphPoint(Coordinate([10, 5, 5]), radius=0, node_type=0),
            36: GraphPoint(Coordinate([10, 6, 5]), radius=0, node_type=0),
            37: GraphPoint(Coordinate([10, 7, 5]), radius=0, node_type=0),
            38: GraphPoint(Coordinate([10, 8, 5]), radius=0, node_type=0),
            39: GraphPoint(Coordinate([10, 9, 5]), radius=0, node_type=0),
            40: GraphPoint(Coordinate([10, 10, 5]), radius=0, node_type=0),
        }

        edges = [
            (0, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (0, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (5, 21),
            (21, 22),
            (22, 23),
            (23, 24),
            (24, 25),
            (25, 26),
            (26, 27),
            (27, 28),
            (28, 29),
            (29, 30),
            (10, 31),
            (31, 32),
            (32, 33),
            (33, 34),
            (34, 35),
            (35, 36),
            (36, 37),
            (37, 38),
            (38, 39),
            (39, 40),
        ]

        return GraphPoints(
            points,
            PointsSpec(
                roi=Roi(Coordinate((-100, -100, -100)), Coordinate((300, 300, 300)))
            ),
            edges=edges,
        )

    def _graph_points(self, points, edges, spec=None):
        return GraphPoints(
            points,
            spec=PointsSpec(roi=Roi(*(Coordinate((None,) * 3),) * 2)),
            edges=edges,
        )

    def _get_points(
        self, inside: np.ndarray, slope: np.ndarray, bb: Roi
    ) -> Tuple[Dict[int, GraphPoint], List[Tuple[int, int]]]:
        slope = slope / max(slope)
        shape = np.array(bb.get_shape())
        outside_down = inside - shape * slope
        outside_up = inside + shape * slope
        down_intercept = self._resample_relative(inside, outside_down, bb)
        up_intercept = self._resample_relative(inside, outside_up, bb)

        points = {
            # line
            0: GraphPoint(down_intercept, point_type=0, radius=0),
            1: GraphPoint(up_intercept, point_type=0, radius=0),
        }
        edges = set((0, 1))
        return self._graph_points(points, edges)

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
