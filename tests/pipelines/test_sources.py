from gunpowder import (
    BatchRequest,
    Batch,
    Coordinate,
    Roi,
    Array,
    ArrayKey,
    ArraySpec,
    GraphKey,
    GraphSpec,
    Graph,
    Node,
    Edge,
)
from gunpowder.nodes import MergeProvider, RandomLocation, BatchProvider, Normalize
import numpy as np

from neurolight.gunpowder.nodes import TopologicalMatcher, RasterizeSkeleton

from typing import Dict, List
import itertools
import logging

logger = logging.getLogger(__file__)


class TestImageSource(BatchProvider):
    def __init__(
        self,
        array: ArrayKey,
        array_specs: Dict[ArrayKey, ArraySpec],
        size: Coordinate,
        voxel_size: Coordinate,
    ):
        self.array = array
        self.array_specs = array_specs
        self.size = size
        self.voxel_size = voxel_size

    def setup(self):
        spec = self.array_specs[self.array]
        roi = Roi(Coordinate([0] * len(self.size)), self.size)
        spec.roi = roi
        spec.dtype = np.uint8
        self.provides(self.array, spec)

    def provide(self, request: BatchRequest) -> Batch:
        if self.array in request:
            spec = request[self.array].copy()
            spec.dtype = self.array_specs[self.array].dtype
            spec.voxel_size = self.voxel_size
            roi = spec.roi
            assert (a % b == 0 for a, b in zip(roi.get_shape(), spec.voxel_size))
            assert (a // b < 500 for a, b in zip(roi.get_shape(), spec.voxel_size))
            array_start = roi.get_begin() / spec.voxel_size
            array_size = roi.get_shape() / spec.voxel_size
            array = np.zeros((2,) + tuple(array_size), dtype=spec.dtype)
            for x, y, z in itertools.product(
                *[range(a, a + b) for a, b in zip(array_start, array_size)]
            ):
                i = x - array_start[0]
                j = y - array_start[1]
                k = z - array_start[2]
                if (
                    abs(x * spec.voxel_size[0] - y * spec.voxel_size[1])
                    <= max(spec.voxel_size)
                    and abs(x * spec.voxel_size[0] - z * spec.voxel_size[2])
                    <= max(spec.voxel_size)
                    and abs(y * spec.voxel_size[1] - z * spec.voxel_size[2])
                    <= max(spec.voxel_size)
                ):
                    array[:, int(i), int(j), int(k)] = 255

            batch = Batch()
            batch[self.array] = Array(array, spec)

            return batch
        else:
            return Batch()


class TestPointSource(BatchProvider):
    def __init__(
        self, points: List[GraphKey], directed: bool, size: Coordinate, num_points: int
    ):
        self.points = points
        self.directed = directed
        self.size = size
        self.num_points = num_points

    def setup(self):
        roi = Roi(Coordinate([0] * len(self.size)), self.size)
        for points_key in self.points:
            self.provides(points_key, GraphSpec(roi=roi))

        k = min(self.size)
        nodes = [
            Node(id=i, location=np.array([i * k / self.num_points] * 3))
            for i in range(self.num_points)
        ]
        edges = [Edge(i, i + 1) for i in range(self.num_points - 1)]

        self.graph = Graph(nodes, edges, GraphSpec(roi=roi, directed=self.directed))

    def provide(self, request: BatchRequest) -> Batch:
        batch = Batch()
        for points_key in self.points:
            if points_key in request:
                spec = request[points_key].copy()

                subgraph = self.graph.crop(roi=spec.roi, copy=True)

                batch[points_key] = subgraph
        return batch


def get_test_data_sources(setup_config):

    input_shape = Coordinate(setup_config["INPUT_SHAPE"])
    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    input_size = input_shape * voxel_size

    micron_scale = voxel_size[0]

    # New array keys
    # Note: These are intended to be requested with size input_size
    raw = ArrayKey("RAW")
    consensus = GraphKey("CONSENSUS")
    skeletonization = GraphKey("SKELETONIZATION")
    matched = GraphKey("MATCHED")
    nonempty_placeholder = GraphKey("NONEMPTY")
    labels = ArrayKey("LABELS")

    if setup_config["FUSION_PIPELINE"]:
        ensure_nonempty = nonempty_placeholder
    else:
        ensure_nonempty = consensus

    data_sources = (
        (
            TestImageSource(
                array=raw,
                array_specs={
                    raw: ArraySpec(
                        interpolatable=True, voxel_size=voxel_size, dtype=np.uint16
                    )
                },
                size=input_size * 3,
                voxel_size=voxel_size,
            ),
            TestPointSource(
                points=[consensus, nonempty_placeholder],
                directed=True,
                size=input_size * 3,
                num_points=30,
            ),
            TestPointSource(
                points=[skeletonization],
                directed=False,
                size=input_size * 3,
                num_points=333,
            ),
        )
        + MergeProvider()
        + RandomLocation(
            ensure_nonempty=ensure_nonempty,
            ensure_centered=True,
            point_balance_radius=10 * micron_scale,
        )
        + TopologicalMatcher(
            skeletonization,
            consensus,
            matched,
            match_distance_threshold=50 * micron_scale,
            max_gap_crossing=30 * micron_scale,
            try_complete=False,
            use_gurobi=True,
            expected_edge_len=1000,
        )
        + RasterizeSkeleton(
            points=matched,
            array=labels,
            array_spec=ArraySpec(
                interpolatable=False, voxel_size=voxel_size, dtype=np.uint64
            ),
        )
        + Normalize(raw)
    )

    return (
        data_sources,
        raw,
        labels,
        consensus,
        nonempty_placeholder,
        matched,
    )
