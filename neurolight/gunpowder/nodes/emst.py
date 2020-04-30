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

import mlpack as mlp

import numpy as np

import maximin

import itertools


class EMST(BatchFilter):
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
        embeddings: ArrayKey,
        mask: ArrayKey,
        mst: GraphKey,
        coordinate_scale: np.ndarray,
        distance_attr="distance",
    ):
        self.embeddings = embeddings
        self.mask = mask
        self.mst = mst
        self.coordinate_scale = coordinate_scale
        self.distance_attr = distance_attr

    def setup(self):
        self.enable_autoskip()
        spatial_dims = len(self.spec[self.embeddings].voxel_size)
        mst_spec = GraphSpec(
            roi=Roi((None,) * spatial_dims, (None,) * spatial_dims), directed=False
        )
        self.provides(self.mst, mst_spec)

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        upstream_dependencies = {
            self.embeddings: self.spec[self.embeddings],
            self.mask: self.spec[self.mask],
        }
        downstream_request = {self.mst: request[self.mst]}
        upstream_dependencies = ProviderSpec(
            array_specs=upstream_dependencies, graph_specs=downstream_request
        )
        upstream_roi = upstream_dependencies.get_common_roi()

        deps[self.embeddings] = ArraySpec(roi=upstream_roi)
        deps[self.mask] = ArraySpec(roi=upstream_roi)

        return deps

    def process(self, batch, request: BatchRequest):
        outputs = Batch()

        voxel_size = batch[self.embeddings].spec.voxel_size
        offset = batch[self.embeddings].spec.roi.get_begin()
        embeddings = batch[self.embeddings].data
        candidates = batch[self.mask].data
        _, depth, height, width = embeddings.shape
        coordinates = np.meshgrid(
            np.arange(
                0, (depth - 0.5) * self.coordinate_scale[0], self.coordinate_scale[0]
            ),
            np.arange(
                0, (height - 0.5) * self.coordinate_scale[1], self.coordinate_scale[1]
            ),
            np.arange(
                0, (width - 0.5) * self.coordinate_scale[2], self.coordinate_scale[2]
            ),
            indexing="ij",
        )
        for i in range(len(coordinates)):
            coordinates[i] = coordinates[i].astype(np.float32)

        embedding = np.concatenate([embeddings, coordinates], 0)
        embedding = np.transpose(embedding, axes=[1, 2, 3, 0])
        embedding = embedding.reshape(depth * width * height, -1)
        candidates = candidates.reshape(depth * width * height)
        embedding = embedding[candidates == 1, :]

        emst = mlp.emst(embedding)["output"]

        nodes = set()
        edges = []
        for u, v, distance in emst:
            u = int(u)
            pos_u = embedding[u][-3:] / self.coordinate_scale * voxel_size
            v = int(v)
            pos_v = embedding[v][-3:] / self.coordinate_scale * voxel_size
            nodes.add(Node(u, location=pos_u + offset))
            nodes.add(Node(v, location=pos_v + offset))
            edges.append(Edge(u, v, attrs={self.distance_attr: distance}))

        graph_spec = request[self.mst]
        graph_spec.directed = False

        outputs[self.mst] = Graph(nodes, edges, graph_spec)
        print(f"OUTPUTS CONTAINS MST WITH {len(list(outputs[self.mst].nodes))} NODES")

        return outputs
