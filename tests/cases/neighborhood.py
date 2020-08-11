# 3rd party library imports
import numpy as np

# gunpowder imports
from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    GraphKey,
    GraphSpec,
    Graph,
    ArrayKey,
    ArraySpec,
    Roi,
    Coordinate,
    build,
)
from gunpowder.graph import Node, Edge

# neurolight imports
from neurolight.gunpowder.nodes.neighborhood import Neighborhood

# stdlib imports


class TestSource(BatchProvider):
    def __init__(self, graph):
        self.graph = graph
        self.graph_spec = GraphSpec(
            roi=Roi((-10, -10, -10), (30, 30, 30)), directed=False
        )
        self.component_1_nodes = [Node(i, np.array([0, i, 0])) for i in range(10)] + [
            Node(i + 10, np.array([i, 5, 0])) for i in range(10)
        ]
        self.component_1_edges = (
            [Edge(i, i + 1) for i in range(9)]
            + [Edge(i + 10, i + 11) for i in range(9)]
            + [Edge(5, 15)]
        )
        self.component_2_nodes = [Node(i + 20, np.array([i, 4, 0])) for i in range(10)]
        self.component_2_edges = [Edge(i + 20, i + 21) for i in range(9)]

    def setup(self):
        self.provides(self.graph, self.graph_spec)

    def provide(self, request):
        outputs = Batch()
        spec = self.graph_spec.copy()
        spec.roi = request[self.graph].roi
        outputs[self.graph] = Graph(
            self.component_1_nodes + self.component_2_nodes,
            self.component_1_edges + self.component_2_edges,
            spec,
        )
        return outputs


def test_neighborhood():
    # array keys
    graph = GraphKey("GRAPH")
    neighborhood = ArrayKey("NEIGHBORHOOD")
    neighborhood_mask = ArrayKey("NEIGHBORHOOD_MASK")

    distance = 1

    pipeline = TestSource(graph) + Neighborhood(
        graph,
        neighborhood,
        neighborhood_mask,
        distance,
        array_specs={
            neighborhood: ArraySpec(voxel_size=Coordinate((1, 1, 1))),
            neighborhood_mask: ArraySpec(voxel_size=Coordinate((1, 1, 1))),
        },
    )

    request = BatchRequest()
    request[neighborhood] = ArraySpec(roi=Roi((0, 0, 0), (10, 10, 10)))
    request[neighborhood_mask] = ArraySpec(roi=Roi((0, 0, 0), (10, 10, 10)))

    with build(pipeline):
        batch = pipeline.request_batch(request)
        n_data = batch[neighborhood].data
        n_mask = batch[neighborhood_mask].data
        masked_ind = list(
            set(
                [(0, i, 0) for i in range(10) if i not in [0, 4]]
                + [(i, 5, 0) for i in range(10)]
                + [(i, 4, 0) for i in range(10) if i not in [0]]
            )
        )
        assert all(
            n_mask[tuple(zip(*masked_ind))]
        ), f"expected {masked_ind} but saw {np.where(n_mask==1)}"


def test_6_neighborhood():
    # array keys
    graph = GraphKey("GRAPH")
    neighborhood = ArrayKey("NEIGHBORHOOD")
    neighborhood_mask = ArrayKey("NEIGHBORHOOD_MASK")

    distance = 1

    pipeline = TestSource(graph) + Neighborhood(
        graph,
        neighborhood,
        neighborhood_mask,
        distance,
        array_specs={
            neighborhood: ArraySpec(voxel_size=Coordinate((1, 1, 1))),
            neighborhood_mask: ArraySpec(voxel_size=Coordinate((1, 1, 1))),
        },
        k=6,
    )

    request = BatchRequest()
    request[neighborhood] = ArraySpec(roi=Roi((0, 0, 0), (10, 10, 10)))
    request[neighborhood_mask] = ArraySpec(roi=Roi((0, 0, 0), (10, 10, 10)))

    with build(pipeline):
        batch = pipeline.request_batch(request)
        n_data = batch[neighborhood].data
        n_mask = batch[neighborhood_mask].data
        masked_ind = list(
            set(
                [(0, i, 0) for i in range(10) if i not in [0, 4]]
                + [(i, 5, 0) for i in range(10)]
                + [(i, 4, 0) for i in range(10) if i not in [0]]
            )
        )
        assert all(
            n_mask[tuple(zip(*masked_ind))]
        ), f"expected {masked_ind} but saw {np.where(n_mask==1)}"
