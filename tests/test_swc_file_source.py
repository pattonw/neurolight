from pathlib import Path

from .swc_base_test import SWCBaseTest
from neurolight.gunpowder.swc_file_source import SwcFileSource, SwcPoint
from gunpowder import PointsKey, PointsSpec, BatchRequest, Roi, build, Coordinate
import numpy as np
import networkx as nx

from typing import Dict, List


class SwcTest(SWCBaseTest):
    def setUp(self):
        super(SwcTest, self).setUp()

    def test_read_single_swc(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points())

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 5), (11, 11, 1)))})
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
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 1, 5), (11, 10, 1)))})
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
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 5, 10), (1, 3, 1)))})
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
            tuple(np.array([0.0, 5.0, 10.0])),
            tuple(np.array([0.0, 6.0, 10.0])),
            tuple(np.array([0.0, 7.0, 10.0])),
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
        source = SwcFileSource(path, [swc], keep_ids=True)

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 5, 10), (1, 3, 1)))})
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
            (12, tuple(np.array([0.0, 5.0, 10.0]))),
            (13, tuple(np.array([0.0, 6.0, 10.0]))),
            (14, tuple(np.array([0.0, 7.0, 10.0]))),
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
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 5), (11, 11, 3)))})
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
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 5), (11, 11, 1)))})
            )

        temp_g = nx.DiGraph()
        for point_id, point in batch.points[swc].data.items():
            self.assertEqual(point_id, point.point_id)
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
