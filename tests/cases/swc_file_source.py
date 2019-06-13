from pathlib import Path

from .provider_test import TestWithTempFiles
from neurolight.gunpowder.swc_file_source import SwcFileSource, SwcPoint
from gunpowder import PointsKey, PointsSpec, BatchRequest, Roi, build, Coordinate
import numpy as np
import networkx as nx

from typing import Dict, List


class SwcTest(TestWithTempFiles):
    def setUp(self):
        super(SwcTest, self).setUp()

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

    def _toy_swc(self, file_path: Path):
        raise NotImplementedError

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
            SwcPoint(0, 0, Coordinate([0, 0, 0]), 0, 0),
            SwcPoint(1, 0, Coordinate([1, 0, 0]), 0, 0),
            SwcPoint(2, 0, Coordinate([2, 0, 0]), 0, 1),
            SwcPoint(3, 0, Coordinate([3, 0, 0]), 0, 2),
            SwcPoint(4, 0, Coordinate([4, 0, 0]), 0, 3),
            SwcPoint(5, 0, Coordinate([5, 0, 0]), 0, 4),
            SwcPoint(6, 0, Coordinate([6, 0, 0]), 0, 5),
            SwcPoint(7, 0, Coordinate([7, 0, 0]), 0, 6),
            SwcPoint(8, 0, Coordinate([8, 0, 0]), 0, 7),
            SwcPoint(9, 0, Coordinate([9, 0, 0]), 0, 8),
            SwcPoint(10, 0, Coordinate([10, 0, 0]), 0, 9),
            # bottom line
            SwcPoint(11, 0, Coordinate([0, 1, 0]), 0, 0),
            SwcPoint(12, 0, Coordinate([0, 2, 0]), 0, 11),
            SwcPoint(13, 0, Coordinate([0, 3, 0]), 0, 12),
            SwcPoint(14, 0, Coordinate([0, 4, 0]), 0, 13),
            SwcPoint(15, 0, Coordinate([0, 5, 0]), 0, 14),
            SwcPoint(16, 0, Coordinate([0, 6, 0]), 0, 15),
            SwcPoint(17, 0, Coordinate([0, 7, 0]), 0, 16),
            SwcPoint(18, 0, Coordinate([0, 8, 0]), 0, 17),
            SwcPoint(19, 0, Coordinate([0, 9, 0]), 0, 18),
            SwcPoint(20, 0, Coordinate([0, 10, 0]), 0, 19),
            # mid line
            SwcPoint(21, 0, Coordinate([5, 1, 0]), 0, 5),
            SwcPoint(22, 0, Coordinate([5, 2, 0]), 0, 21),
            SwcPoint(23, 0, Coordinate([5, 3, 0]), 0, 22),
            SwcPoint(24, 0, Coordinate([5, 4, 0]), 0, 23),
            SwcPoint(25, 0, Coordinate([5, 5, 0]), 0, 24),
            SwcPoint(26, 0, Coordinate([5, 6, 0]), 0, 25),
            SwcPoint(27, 0, Coordinate([5, 7, 0]), 0, 26),
            SwcPoint(28, 0, Coordinate([5, 8, 0]), 0, 27),
            SwcPoint(29, 0, Coordinate([5, 9, 0]), 0, 28),
            SwcPoint(30, 0, Coordinate([5, 10, 0]), 0, 29),
            # top line
            SwcPoint(31, 0, Coordinate([10, 1, 0]), 0, 10),
            SwcPoint(32, 0, Coordinate([10, 2, 0]), 0, 31),
            SwcPoint(33, 0, Coordinate([10, 3, 0]), 0, 32),
            SwcPoint(34, 0, Coordinate([10, 4, 0]), 0, 33),
            SwcPoint(35, 0, Coordinate([10, 5, 0]), 0, 34),
            SwcPoint(36, 0, Coordinate([10, 6, 0]), 0, 35),
            SwcPoint(37, 0, Coordinate([10, 7, 0]), 0, 36),
            SwcPoint(38, 0, Coordinate([10, 8, 0]), 0, 37),
            SwcPoint(39, 0, Coordinate([10, 9, 0]), 0, 38),
            SwcPoint(40, 0, Coordinate([10, 10, 0]), 0, 39),
        ]
        return points

    def test_read_single_swc(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points())

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, swc)

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 0), (11, 11, 1)))})
            )

        for point in self._toy_swc_points():
            self.assertCountEqual(
                point.location, batch.points[swc].data[point.point_id].location
            )

    def test_relabel_components(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points())

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, swc)

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 1, 0), (11, 10, 1)))})
            )

        temp_g = nx.DiGraph()
        for point_id, point in batch.points[swc].data.items():
            temp_g.add_node(point.point_id, label_id=point.label_id)
            if (
                point.parent_id != -1
                and point.parent_id != point.point_id
                and point.parent_id in batch.points[swc].data
            ):
                temp_g.add_edge(point.point_id, point.parent_id)

        previous_label = None
        ccs = list(nx.weakly_connected_components(temp_g))
        self.assertEqual(len(ccs), 3)
        for cc in ccs:
            self.assertEqual(len(cc), 10)
            label = None
            for point_id in cc:
                if label is None:
                    label = temp_g.nodes[point_id]["label_id"]
                    self.assertNotEqual(label, previous_label)
                self.assertEqual(temp_g.nodes[point_id]["label_id"], label)
            previous_label = label

    def test_create_boundary_nodes(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(
            path, self._toy_swc_points(), {"resolution": np.array([2, 2, 2])}
        )

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, swc)

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 5, 0), (1, 3, 1)))})
            )

        temp_g = nx.DiGraph()
        for point_id, point in batch.points[swc].data.items():
            temp_g.add_node(
                point.point_id, label_id=point.label_id, location=point.location
            )
            if (
                point.parent_id != -1
                and point.parent_id != point.point_id
                and point.parent_id in batch.points[swc].data
            ):
                temp_g.add_edge(point.point_id, point.parent_id)
            else:
                root = point.point_id

        current = root
        expected_path = [
            tuple(np.array([0.0, 5.0, 0.0])),
            tuple(np.array([0.0, 6.0, 0.0])),
            tuple(np.array([0.0, 7.0, 0.0])),
        ]
        expected_node_ids = [0, 1, 2]
        path = []
        node_ids = []
        while current is not None:
            node_ids.append(current)
            path.append(tuple(temp_g.nodes[current]["location"]))
            predecessors = list(temp_g._pred[current].keys())
            current = predecessors[0] if len(predecessors) == 1 else None

        self.assertCountEqual(path, expected_path)
        self.assertCountEqual(node_ids, expected_node_ids)

    def test_keep_node_ids(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(
            path, self._toy_swc_points(), {"resolution": np.array([2, 2, 2])}
        )

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, swc, keep_ids=True)

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 5, 0), (1, 3, 1)))})
            )

        temp_g = nx.DiGraph()
        for point_id, point in batch.points[swc].data.items():
            temp_g.add_node(
                point.point_id, label_id=point.label_id, location=point.location
            )
            if (
                point.parent_id != -1
                and point.parent_id != point.point_id
                and point.parent_id in batch.points[swc].data
            ):
                temp_g.add_edge(point.point_id, point.parent_id)
            else:
                root = point.point_id

        current = root
        expected_path = [
            (12, tuple(np.array([0.0, 5.0, 0.0]))),
            (13, tuple(np.array([0.0, 6.0, 0.0]))),
            (14, tuple(np.array([0.0, 7.0, 0.0]))),
        ]
        path = []
        while current is not None:
            path.append((current, tuple(temp_g.nodes[current]["location"])))
            predecessors = list(temp_g._pred[current].keys())
            current = predecessors[0] if len(predecessors) == 1 else None

        self.assertCountEqual(path, expected_path)

    def test_multiple_files(self):
        path = Path(self.path_to("test_swc_sources"))
        path.mkdir(parents=True, exist_ok=True)

        # write test swc
        for i in range(3):
            self._write_swc(
                path / "{}.swc".format(i),
                self._toy_swc_points(),
                {"offset": np.array([0, 0, i])},
            )

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, swc)

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 0), (11, 11, 3)))})
            )

        temp_g = nx.DiGraph()
        for point_id, point in batch.points[swc].data.items():
            temp_g.add_node(point.point_id, label_id=point.label_id)
            if (
                point.parent_id != -1
                and point.parent_id != point.point_id
                and point.parent_id in batch.points[swc].data
            ):
                temp_g.add_edge(point.point_id, point.parent_id)

        previous_label = None
        ccs = list(nx.weakly_connected_components(temp_g))
        self.assertEqual(len(ccs), 3)
        for cc in ccs:
            self.assertEqual(len(cc), 41)
            label = None
            for point_id in cc:
                if label is None:
                    label = temp_g.nodes[point_id]["label_id"]
                    self.assertNotEqual(label, previous_label)
                self.assertEqual(temp_g.nodes[point_id]["label_id"], label)
            previous_label = label

    def test_overlap(self):
        path = Path(self.path_to("test_swc_sources"))
        path.mkdir(parents=True, exist_ok=True)

        # write test swc
        for i in range(3):
            self._write_swc(
                path / "{}.swc".format(i),
                self._toy_swc_points(),
                {"offset": np.array([0, 0, 0])},
            )

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, swc)

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 0), (11, 11, 1)))})
            )

        temp_g = nx.DiGraph()
        for point_id, point in batch.points[swc].data.items():
            temp_g.add_node(point.point_id, label_id=point.label_id)
            if (
                point.parent_id != -1
                and point.parent_id != point.point_id
                and point.parent_id in batch.points[swc].data
            ):
                temp_g.add_edge(point.point_id, point.parent_id)

        previous_label = None
        ccs = list(nx.weakly_connected_components(temp_g))
        self.assertEqual(len(ccs), 3)
        for cc in ccs:
            self.assertEqual(len(cc), 41)
            label = None
            for point_id in cc:
                if label is None:
                    label = temp_g.nodes[point_id]["label_id"]
                    self.assertNotEqual(label, previous_label)
                self.assertEqual(temp_g.nodes[point_id]["label_id"], label)
            previous_label = label