from pathlib import Path

from .swc_base_test import SWCBaseTest
from neurolight.gunpowder.nodes.swc_file_source import SwcFileSource
from gunpowder import PointsKey, PointsSpec, BatchRequest, Roi, build
import numpy as np
import networkx as nx


class SwcFileSourceTest(SWCBaseTest):
    def setUp(self):
        super(SwcFileSourceTest, self).setUp()

    def test_read_single_swc(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points().graph)

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 5), (11, 11, 1)))})
            )

        for point_id, point in self._toy_swc_points().data.items():
            self.assertCountEqual(
                point.location, batch.points[swc].data[point_id].location
            )

    def test_relabel_components(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points().graph)

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 1, 5), (11, 10, 1)))})
            )

        temp_g = batch.points[swc].graph

        previous_label = None
        ccs = list(nx.weakly_connected_components(temp_g))
        self.assertEqual(len(ccs), 3)
        for cc in ccs:
            self.assertEqual(len(cc), 10)
            label = None
            for point_id in cc:
                if label is None:
                    label = temp_g.nodes[point_id]["component"]
                    self.assertNotEqual(label, previous_label)
                self.assertEqual(temp_g.nodes[point_id]["component"], label)
            previous_label = label

    def test_create_boundary_nodes(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(
            path, self._toy_swc_points().graph, {"resolution": np.array([2, 2, 2])}
        )

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 5, 10), (1, 3, 1)))})
            )

        temp_g = batch.points[swc].graph

        # root is only node with in_degree 0
        current = [n for n, d in temp_g.in_degree() if d == 0][0]
        expected_path = [
            tuple(np.array([0.0, 5.0, 10.0])),
            tuple(np.array([0.0, 6.0, 10.0])),
            tuple(np.array([0.0, 7.0, 10.0])),
        ]
        # expect relabelled ids
        path = []
        while current is not None:
            path.append(tuple(temp_g.nodes[current]["location"]))
            successors = list(temp_g._succ[current].keys())
            current = successors[0] if len(successors) == 1 else None

        self.assertCountEqual(path, expected_path)

    def test_keep_node_ids(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(
            path, self._toy_swc_points().graph, {"resolution": np.array([2, 2, 2])}
        )

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, [swc], keep_ids=True)

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 5, 10), (1, 3, 1)))})
            )

        temp_g = batch.points[swc].graph

        # root is only node with in_degree 0
        current = [n for n, d in temp_g.in_degree() if d == 0][0]

        # edge nodes can't keep the same id in case one node has multiple children
        # in the roi.
        expected_path = [
            tuple(np.array([0.0, 5.0, 10.0])),
            tuple(np.array([0.0, 6.0, 10.0])),
            tuple(np.array([0.0, 7.0, 10.0])),
        ]
        path = []
        while current is not None:
            path.append(tuple(temp_g.nodes[current]["location"]))
            successors = list(temp_g._succ[current].keys())
            current = successors[0] if len(successors) == 1 else None

        self.assertCountEqual(path, expected_path)

    def test_multiple_files(self):
        path = Path(self.path_to("test_swc_sources"))
        path.mkdir(parents=True, exist_ok=True)

        # write test swc
        for i in range(3):
            self._write_swc(
                path / "{}.swc".format(i),
                self._toy_swc_points().graph,
                {"offset": np.array([0, 0, i])},
            )

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 5), (11, 11, 3)))})
            )

        temp_g = batch.points[swc].graph

        previous_label = None
        ccs = list(nx.weakly_connected_components(temp_g))
        self.assertEqual(len(ccs), 3)
        for cc in ccs:
            self.assertEqual(len(cc), 41)
            label = None
            for point_id in cc:
                if label is None:
                    label = temp_g.nodes[point_id]["component"]
                    self.assertNotEqual(label, previous_label)
                self.assertEqual(temp_g.nodes[point_id]["component"], label)
            previous_label = label

    def test_overlap(self):
        path = Path(self.path_to("test_swc_sources"))
        path.mkdir(parents=True, exist_ok=True)

        # write test swc
        for i in range(3):
            self._write_swc(
                path / "{}.swc".format(i),
                self._toy_swc_points().graph,
                {"offset": np.array([0, i, 0])},
            )

        # read arrays
        swc = PointsKey("SWC")
        source = SwcFileSource(path, [swc])

        with build(source):
            batch = source.request_batch(
                BatchRequest({swc: PointsSpec(roi=Roi((0, 0, 5), (11, 13, 1)))})
            )

        temp_g = batch.points[swc].graph

        previous_label = None
        ccs = list(nx.weakly_connected_components(temp_g))
        self.assertEqual(len(ccs), 3)
        for cc in ccs:
            self.assertEqual(len(cc), 41)
            label = None
            for point_id in cc:
                if label is None:
                    label = temp_g.nodes[point_id]["component"]
                    self.assertNotEqual(label, previous_label)
                self.assertEqual(temp_g.nodes[point_id]["component"], label)
            previous_label = label
