from gunpowder import (
    BatchFilter,
    BatchRequest,
    Batch,
    Coordinate,
    ArrayKey,
    ArraySpec,
    Array,
    Roi,
    ProviderSpec,
)
from gunpowder.graph import Node, Edge, Graph, GraphKey
from gunpowder.graph_spec import GraphSpec

import numpy as np

import maximin

import itertools


class MiniMax(BatchFilter):
    """Given a mask and an intensity array, return the minimum
        spanning tree over all points in the mask, where the
        distance between two points is the minimum intensity
        allong the path between two points in the intensity
        that maximizes the minimum intensity

        Note, some nodes in the output mst may not be on the input
        mask due to the possibility that branch points in the mst
        occur in the "background". Preserving the mst topology is
        not possible without including these nodes.

        Args:

            intensities (:class:``ArrayKey``):

                The intensity array

            mask (:class:``ArrayKey``):

                The mask of points to consider

            mst (:class:``Coordinate``):

                The Graph containing the output minimax minimal spanning tree
        """

    def __init__(
        self,
        intensities: ArrayKey,
        mask: ArrayKey,
        mst: GraphKey,
        distance_attr="distance",
    ):
        self.intensities = intensities
        self.mask = mask
        self.mst = mst
        self.distance_attr = distance_attr

    def setup(self):
        self.enable_autoskip()
        spatial_dims = len(self.spec[self.intensities].voxel_size)
        mst_spec = GraphSpec(
            roi=Roi((None,) * spatial_dims, (None,) * spatial_dims), directed=False
        )
        self.provides(self.mst, mst_spec)

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        upstream_dependencies = {
            self.intensities: self.spec[self.intensities],
            self.mask: self.spec[self.mask],
        }
        downstream_request = {self.mst: request[self.mst]}
        upstream_dependencies = ProviderSpec(
            array_specs=upstream_dependencies, graph_specs=downstream_request
        )
        upstream_roi = upstream_dependencies.get_common_roi()

        deps[self.intensities] = ArraySpec(roi=upstream_roi)
        deps[self.mask] = ArraySpec(roi=upstream_roi)

        return deps

    def process(self, batch, request: BatchRequest):
        outputs = Batch()

        voxel_size = batch[self.intensities].spec.voxel_size
        roi = batch[self.intensities].spec.roi
        offset = batch[self.intensities].spec.roi.get_begin()
        spatial_dims = len(voxel_size)
        intensities = batch[self.intensities].data
        intensities = intensities.reshape((-1,) + intensities.shape[-spatial_dims:])[0]
        maxima = batch[self.mask].data
        maxima = maxima.reshape((-1,) + maxima.shape[-spatial_dims:])[0]

        minimax_edges = maximin.maximin_tree_query(intensities, maxima.astype(np.uint8))
        maximin_id = itertools.count(start=0)

        nodes = set()
        edges = []
        ids = {}
        for a, b, cost in minimax_edges:
            a_id = ids.setdefault(a, next(maximin_id))
            b_id = ids.setdefault(b, next(maximin_id))
            a_loc = np.array(a) * voxel_size + offset
            b_loc = np.array(b) * voxel_size + offset
            assert roi.contains(a_loc), f"Roi {roi} does not contain {a_loc}"
            assert roi.contains(b_loc), f"Roi {roi} does not contain {b_loc}"

            nodes.add(Node(a_id, location=a_loc))
            nodes.add(Node(b_id, location=b_loc))
            edges.append(Edge(a_id, b_id, attrs={self.distance_attr: cost}))

        graph_spec = request[self.mst]
        graph_spec.directed = False

        outputs[self.mst] = Graph(nodes, edges, graph_spec)

        return outputs
