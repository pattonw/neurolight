from gunpowder import BatchFilter, BatchRequest, Batch, ArraySpec, Graph

import numpy as np
import networkx as nx


class ScoreEdges(BatchFilter):
    """
    Score an edge based on a voxel-level path between two nodes.
    """
    def __init__(self, mst, dense_mst, embeddings, distance_attr, path_stat):
        self.mst = mst
        self.dense_mst = dense_mst
        self.embeddings = embeddings
        self.distance_attr = distance_attr
        self.path_stat = path_stat

    def setup(self):
        self.updates(self.mst, self.spec[self.mst].copy())

    def prepare(self, request):
        deps = BatchRequest()

        deps[self.dense_mst] = request[self.mst].copy()
        deps[self.mst] = request[self.mst].copy()
        deps[self.embeddings] = ArraySpec(roi=request[self.mst].roi)

        return deps

    def process(self, batch, request):
        mst = batch[self.mst].to_nx_graph()
        dense_mst = batch[self.dense_mst].to_nx_graph()
        embeddings = batch[self.embeddings].data
        voxel_size = batch[self.embeddings].spec.voxel_size
        offset = batch[self.embeddings].spec.roi.get_begin()

        for (u, v), chain in self.get_edge_chains(mst, dense_mst):
            chain_embeddings = []
            for n in chain:
                n_loc = dense_mst.nodes[n]["location"]
                n_ind = tuple(int(x) for x in ((n_loc - offset) // voxel_size))
                chain_embeddings.append(
                    embeddings[(slice(None),) * (len(embeddings.shape) - 3) + n_ind]
                )

            mst.edges[(u, v)][self.distance_attr] = self.get_stat(chain)

        outputs = Batch()
        outputs[self.mst] = Graph.from_nx_graph(mst, batch[self.mst].spec)

        return outputs

    def get_edge_chains(self, mst, dense_mst):
        location_lookup = {}
        for node, attrs in mst.nodes.items():
            location_lookup[tuple(int(x) for x in attrs["location"])] = node

        mst_nodes_to_dense = {}
        for node, attrs in dense_mst.nodes.items():
            if tuple(int(x) for x in attrs["location"]) in location_lookup:
                mst_nodes_to_dense[
                    location_lookup[tuple(int(x) for x in attrs["location"])]
                ] = node

        for u, v in mst.edges:
            yield (u, v), nx.shortest_path(
                dense_mst, mst_nodes_to_dense[u], mst_nodes_to_dense[v]
            )

    def get_stat(self, chain):
        if self.path_stat == "mean":
            return np.mean(
                [np.linalg.norm(c1 - c2) for c1, c2 in zip(chain[:-1], chain[1:])]
            )
        elif self.path_stat == "absolute_diff":
            return np.linalg.norm(chain[0] - chain[-1])
        elif self.path_stat == "total_diff":
            return np.sum(
                [np.linalg.norm(c1 - c2) for c1, c2 in zip(chain[:-1], chain[1:])]
            )
        elif self.path_stat == "max":
            return np.max(
                [np.linalg.norm(c1 - c2) for c1, c2 in zip(chain[:-1], chain[1:])]
            )

        elif isinstance(self.path_stat, float):
            assert (
                self.path_stat >= 0 and self.path_stat <= 1
            ), f"path_stat can be float, but only in range [0, 1], not {self.path_stat}"

        else:
            raise ValueError(f"Unsupported path_stat: {self.path_stat}")
