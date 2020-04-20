from gunpowder import (
    BatchProvider,
    Batch,
    Graph,
    GraphKey,
    GraphKeys,
    GraphSpec,
    Roi,
    Coordinate,
    Node,
    Edge,
    BatchRequest,
    build,
)
import numpy as np

from neurolight.gunpowder.nodes.helpers import FilterComponents


class TestSource(BatchProvider):
    def setup(self):

        self.provides(
            GraphKeys.RAW, GraphSpec(roi=Roi((0, 0, 0), (100, 100, 100)), directed=True)
        )

    def provide(self, request):
        outputs = Batch()

        nodes = [
            Node(id=0, location=np.array((1, 1, 1))),
            Node(id=1, location=np.array((10, 10, 10))),
            Node(id=2, location=np.array((19, 19, 19))),
            Node(id=3, location=np.array((21, 21, 21))),
            Node(id=104, location=np.array((30, 30, 30))),
            Node(id=5, location=np.array((39, 39, 39))),
        ]
        edges = [Edge(0, 1), Edge(1, 2), Edge(3, 104), Edge(104, 5)]
        spec = self.spec[GraphKeys.RAW].copy()
        spec.roi = request[GraphKeys.RAW].roi
        graph = Graph(nodes, edges, spec)

        outputs[GraphKeys.RAW] = graph
        return outputs


def test_filter_components():
    raw = GraphKey("RAW")

    pipeline = TestSource() + FilterComponents(raw, 100, Coordinate((10, 10, 10)))

    request_no_fallback = BatchRequest()
    request_no_fallback[raw] = GraphSpec(roi=Roi((0, 0, 0), (20, 20, 20)))

    with build(pipeline):
        batch = pipeline.request_batch(request_no_fallback)
        assert raw in batch
        assert len(list(batch[raw].connected_components)) == 1

    request_fallback = BatchRequest()
    request_fallback[raw] = GraphSpec(roi=Roi((20, 20, 20), (20, 20, 20)))

    with build(pipeline):
        batch = pipeline.request_batch(request_fallback)
        assert raw in batch
        assert len(list(batch[raw].connected_components)) == 0
