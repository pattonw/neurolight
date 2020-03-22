from provider_test import TestWithTempFiles
from gunpowder import Roi, Coordinate, Node, Graph, GraphSpec, Edge

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
        arr = np.array
        points = [
            # backbone
            Node(id=0, location=arr([0, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=1, location=arr([1, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=2, location=arr([2, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=3, location=arr([3, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=4, location=arr([4, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=5, location=arr([5, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=6, location=arr([6, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=7, location=arr([7, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=8, location=arr([8, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=9, location=arr([9, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=10, location=arr([10, 0, 5]), attrs={"radius": 0, "node_type": 0}),
            # bottom line
            Node(id=11, location=arr([0, 1, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=12, location=arr([0, 2, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=13, location=arr([0, 3, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=14, location=arr([0, 4, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=15, location=arr([0, 5, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=16, location=arr([0, 6, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=17, location=arr([0, 7, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=18, location=arr([0, 8, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=19, location=arr([0, 9, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=20, location=arr([0, 10, 5]), attrs={"radius": 0, "node_type": 0}),
            # mid line
            Node(id=21, location=arr([5, 1, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=22, location=arr([5, 2, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=23, location=arr([5, 3, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=24, location=arr([5, 4, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=25, location=arr([5, 5, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=26, location=arr([5, 6, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=27, location=arr([5, 7, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=28, location=arr([5, 8, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=29, location=arr([5, 9, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=30, location=arr([5, 10, 5]), attrs={"radius": 0, "node_type": 0}),
            # top line
            Node(id=31, location=arr([10, 1, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=32, location=arr([10, 2, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=33, location=arr([10, 3, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=34, location=arr([10, 4, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=35, location=arr([10, 5, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=36, location=arr([10, 6, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=37, location=arr([10, 7, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=38, location=arr([10, 8, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=39, location=arr([10, 9, 5]), attrs={"radius": 0, "node_type": 0}),
            Node(id=40, location=arr([10, 10, 5]), attrs={"radius": 0, "node_type": 0}),
        ]

        edges = [
            Edge(0, 0),
            Edge(0, 1),
            Edge(1, 2),
            Edge(2, 3),
            Edge(3, 4),
            Edge(4, 5),
            Edge(5, 6),
            Edge(6, 7),
            Edge(7, 8),
            Edge(8, 9),
            Edge(9, 10),
            Edge(0, 11),
            Edge(11, 12),
            Edge(12, 13),
            Edge(13, 14),
            Edge(14, 15),
            Edge(15, 16),
            Edge(16, 17),
            Edge(17, 18),
            Edge(18, 19),
            Edge(19, 20),
            Edge(5, 21),
            Edge(21, 22),
            Edge(22, 23),
            Edge(23, 24),
            Edge(24, 25),
            Edge(25, 26),
            Edge(26, 27),
            Edge(27, 28),
            Edge(28, 29),
            Edge(29, 30),
            Edge(10, 31),
            Edge(31, 32),
            Edge(32, 33),
            Edge(33, 34),
            Edge(34, 35),
            Edge(35, 36),
            Edge(36, 37),
            Edge(37, 38),
            Edge(38, 39),
            Edge(39, 40),
        ]

        return Graph(
            points,
            edges,
            GraphSpec(
                roi=Roi(Coordinate((-100, -100, -100)), Coordinate((300, 300, 300)))
            ),
        )

    def _graph_points(self, points, edges, spec=None):
        return Graph(
            points,
            spec=GraphSpec(roi=Roi(*(Coordinate((None,) * 3),) * 2)),
            edges=edges,
        )

    def _get_points(
        self, inside: np.ndarray, slope: np.ndarray, bb: Roi
    ) -> Tuple[Dict[int, Node], List[Tuple[int, int]]]:
        slope = slope / max(slope)
        shape = np.array(bb.get_shape())
        outside_down = inside - shape * slope
        outside_up = inside + shape * slope
        down_intercept = self._resample_relative(inside, outside_down, bb)
        up_intercept = self._resample_relative(inside, outside_up, bb)

        points = {
            # line
            Node(id=0, location=down_intercept, attrs={"node_type": 0, "radius": 0}),
            Node(id=1, location=up_intercept, attrs={"node_type": 0, "radius": 0}),
        }
        edges = [Edge(0, 1)]
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
