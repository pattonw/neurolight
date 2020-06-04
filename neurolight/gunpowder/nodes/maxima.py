from gunpowder import BatchFilter, BatchRequest, Coordinate, ArrayKey, Array
from gunpowder.graph import Node, Edge, Graph
from gunpowder.graph_spec import GraphSpec

from scipy.ndimage import maximum_filter
import sklearn
from sklearn.feature_extraction.image import grid_to_graph
from skimage.morphology import skeletonize_3d as skeletonize
import numpy as np
import networkx as nx

import time
import random
import math
import logging

from .nms import NonMaxSuppression  # noqa

logger = logging.getLogger(__file__)


class SimpleLocalMax(BatchFilter):
    """Simple local maxima of an array.

        Args:

            array (:class:``ArrayKey``):

                The array to run nms on.

            maxima (:class:``ArrayKey``):

                The array to store the maxima in.

            window_size (:class:``Coordinate``):

                The window size to check for local maxima.

            threshold (``float``, optional):

                The minimum value be considered for a local maxima.
        """

    def __init__(
        self,
        array: ArrayKey,
        maxima: ArrayKey,
        window_size: Coordinate,
        threshold: float = None,
    ):

        self.array = array
        self.maxima = maxima
        self.window_size = window_size
        self.threshold = threshold

    def setup(self):
        self.enable_autoskip()
        provided_spec = self.spec[self.array].copy()
        provided_spec.dtype = bool
        self.provides(self.maxima, provided_spec)

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        deps[self.array] = request[self.maxima].copy()

        return deps

    def process(self, batch, request: BatchRequest):
        data = batch[self.array].data

        if self.threshold is None:
            data_min = np.min(data)
            data_med = np.median(data)
            threshold = (data_min + data_med) / 2
        else:
            threshold = self.threshold

        voxel_size = batch[self.array].spec.voxel_size
        window_size = self.window_size / voxel_size

        window_size = (1,) * (len(data.shape) - len(window_size)) + window_size
        max_filter = maximum_filter(data, window_size)
        local_maxima = np.logical_and(max_filter == data, data > threshold)

        spec = batch[self.array].spec.copy()
        spec.dtype = bool

        batch.arrays[self.maxima] = Array(local_maxima, spec)


class Skeletonize(BatchFilter):
    """Get local maxima allong a skeleton via thresholding -> skeletonizing -> downsampling.

        Args:

            array (:class:``ArrayKey``):

                The array to calculate maxima from.

            maxima (:class:``ArrayKey``):

                The array to store the maxima in.

            sample_distance (:class:``Coordinate``):

                skeletonizing will produce a tree. That tree will then be downsampled
                to have points approx sample_dist appart. Note branch points in the
                skeletonization are guaranteed to be contained, so edge lengths may be
                shorter, but should never be longer than the provided length.

            threshold (``float``, optional):

                The minimum value be considered for a local maxima.
        """

    def __init__(
        self,
        array: ArrayKey,
        maxima: ArrayKey,
        sample_distance: float,
        threshold: float = None,
    ):

        self.array = array
        self.maxima = maxima
        self.sample_distance = sample_distance
        self.threshold = threshold

    def setup(self):
        self.enable_autoskip()
        provided_spec = self.spec[self.array].copy()
        provided_spec.dtype = bool
        self.provides(self.maxima, provided_spec)

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        deps[self.array] = request[self.maxima].copy()

        return deps

    def process(self, batch, request: BatchRequest):
        data = batch[self.array].data

        if self.threshold is None:
            data_min = np.min(data)
            data_med = np.median(data)
            threshold = (data_min + data_med) / 2
        else:
            threshold = self.threshold

        thresholded = data > threshold
        skeleton = np.squeeze(thresholded)
        t1 = time.time()
        skeleton = skeletonize(skeleton)
        t2 = time.time()
        logger.debug(f"SKELETONIZING TOOK {t2-t1} SECONDS!")
        skeleton = skeleton > 0

        spec = batch[self.array].spec.copy()
        spec.dtype = bool

        skeleton_array = Array(skeleton, spec)

        """
        t1 = time.time()
        skeleton_graph = self.array_to_graph(skeleton_array)
        t2 = time.time()
        logger.debug(f"GRID TO GRAPH TOOK {t2-t1} SECONDS!")

        t1 = time.time()
        skeleton_graph = self.downsample(skeleton_graph, self.sample_distance)
        t2 = time.time()
        logger.debug(f"DOWNSAMPLING TOOK {t2-t1} SECONDS!")

        t1 = time.time()
        candidates = self.graph_to_array(skeleton_graph, spec)
        t2 = time.time()
        logger.debug(f"GRAPH TO GRID TOOK {t2-t1} SECONDS!")
        """
        
        skeleton_array.data = self.candidates_via_sins(skeleton_array)
        candidates = skeleton_array
        candidates.data = np.expand_dims(candidates.data, 0)

        batch.arrays[self.maxima] = candidates

    def downsample(self, graph, sample_distance):
        g = graph.to_nx_graph()
        sampled_nodes = [n for n in g.nodes if g.degree(n) != 2]
        chain_nodes = [n for n in g.nodes if g.degree(n) == 2]
        chain_g = g.subgraph(chain_nodes)
        for cc in nx.connected_components(chain_g):

            if len(cc) < 2:
                continue
            cc_graph = chain_g.subgraph(cc)
            try:
                head, tail = [n for n in cc_graph.nodes if cc_graph.degree(n) == 1]
            except:
                head = list(cc_graph.nodes)[0]
                tail = head
            cable_len = 0
            previous_location = None
            for node in nx.algorithms.dfs_preorder_nodes(cc_graph, source=head):
                current_loc = cc_graph.nodes[node]["location"]
                if previous_location is not None:
                    diff = abs(previous_location - current_loc)
                    dist = np.linalg.norm(diff)
                    cable_len += dist

                previous_location = current_loc
                if node == tail:
                    break

            num_cuts = cable_len // self.sample_distance
            if num_cuts > 0:
                every = cable_len / (num_cuts + 1)
            else:
                every = float("inf")
            seen_cable = 0
            previous_location = None
            for node in nx.algorithms.dfs_preorder_nodes(cc_graph, source=head):
                current_loc = cc_graph.nodes[node]["location"]
                if previous_location is not None:
                    seen_cable += np.linalg.norm(previous_location - current_loc)
                previous_location = current_loc

                if seen_cable > every:
                    sampled_nodes.append(node)
                    seen_cable -= every

                if node == tail:
                    break
        downsampled = g.subgraph(sampled_nodes)
        return Graph.from_nx_graph(downsampled, graph.spec)

    def array_to_graph(self, array):
        # Override with local function
        sklearn.feature_extraction.image._make_edges_3d = _make_edges_3d

        s = array.data.shape
        # Identify connectivity
        t1 = time.time()
        adj_mat = grid_to_graph(n_x=s[0], n_y=s[1], n_z=s[2], mask=array.data)
        t2 = time.time()
        logger.debug(f"GRID TO GRAPH TOOK {t2-t1} SECONDS!")
        # Identify order of the voxels
        t1 = time.time()
        voxel_locs = compute_voxel_locs(
            mask=array.data,
            offset=array.spec.roi.get_begin(),
            scale=array.spec.voxel_size,
        )
        t2 = time.time()
        logger.debug(f"COMPUTING VOXEL LOCS TOOK {t2-t1} SECONDS!")

        t1 = time.time()
        nodes = [
            Node(node_id, voxel_loc) for node_id, voxel_loc in enumerate(voxel_locs)
        ]

        for a, b in zip(adj_mat.row, adj_mat.col):
            assert all(
                abs(voxel_locs[a] - voxel_locs[b]) <= array.spec.voxel_size
            ), f"{voxel_locs[a] - voxel_locs[b]}, {array.spec.voxel_size}"

        edges = [Edge(a, b) for a, b in zip(adj_mat.row, adj_mat.col) if a != b]
        graph = Graph(nodes, edges, GraphSpec(array.spec.roi, directed=False))
        t2 = time.time()
        logger.debug(f"BUILDING GRAPH TOOK {t2-t1} SECONDS!")
        return graph

    def graph_to_array(self, graph, array_spec):
        data = np.zeros(
            array_spec.roi.get_shape() / array_spec.voxel_size, dtype=array_spec.dtype
        )
        for node in graph.nodes:
            voxel_location = (
                node.location - array_spec.roi.get_begin()
            ) / array_spec.voxel_size
            data[tuple(int(x) for x in voxel_location)] = 1
        return Array(data, array_spec)

    def candidates_via_sins(self, array):

        voxel_shape = array.spec.roi.get_shape() / array.spec.voxel_size
        voxel_size = array.spec.voxel_size
        shifts = tuple(random.random() * 2 * math.pi for _ in range(len(voxel_shape)))
        sphere_radius = self.sample_distance

        ys = [
            np.sin(
                np.linspace(
                    shifts[i],
                    shifts[i]
                    + voxel_shape[i] * (2 * math.pi) * voxel_size[i] / sphere_radius,
                    voxel_shape[i],
                )
            ).reshape(
                tuple(1 if j != i else voxel_shape[i] for j in range(len(voxel_shape)))
            )
            for i in range(len(voxel_shape))
        ]

        x = np.sum(ys)

        weighted_skel = x * array.data

        candidates = np.logical_and(np.equal(maximum_filter(
            weighted_skel, size=(3 for _ in voxel_size), mode="nearest"
        ), weighted_skel), array.data)

        return candidates


def compute_voxel_locs(
    mask: np.ndarray, offset: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    locs = np.where(mask == 1)
    locs = np.stack(locs, axis=1)
    locs = locs * scale + offset
    return locs


# From https://github.com/neurodata/scikit-learn/blob/tom/grid_to_graph_26/sklearn/feature_extraction/image.py
# Used in grid_to_graph
# automatically set to 26-connectivity
def _make_edges_3d(n_x: int, n_y: int, n_z: int, connectivity=26):
    """Returns a list of edges for a 3D image.
    Parameters
    ----------
    n_x : int
        The size of the grid in the x direction.
    n_y : int
        The size of the grid in the y direction.
    n_z : integer, default=1
        The size of the grid in the z direction, defaults to 1
    connectivity : int in [6,18,26], default=26
        Defines what are considered neighbors in voxel space.
    """
    if connectivity not in [6, 18, 26]:
        raise ValueError("Invalid value for connectivity: %r" % connectivity)

    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))

    edges = []

    edges_self = np.vstack((vertices.ravel(), vertices.ravel()))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))

    edges = [edges_self, edges_deep, edges_right, edges_down]

    # Add the other connections
    if connectivity >= 18:
        edges_right_deep = np.vstack(
            (vertices[:, :-1, :-1].ravel(), vertices[:, 1:, 1:].ravel())
        )
        edges_down_right = np.vstack(
            (vertices[:-1, :-1, :].ravel(), vertices[1:, 1:, :].ravel())
        )
        edges_down_deep = np.vstack(
            (vertices[:-1, :, :-1].ravel(), vertices[1:, :, 1:].ravel())
        )
        edges_down_left = np.vstack(
            (vertices[:-1, 1:, :].ravel(), vertices[1:, :-1, :].ravel())
        )
        edges_down_shallow = np.vstack(
            (vertices[:-1, :, 1:].ravel(), vertices[1:, :, :-1].ravel())
        )
        edges_deep_left = np.vstack(
            (vertices[:, 1:, :-1].ravel(), vertices[:, :-1, 1:].ravel())
        )

        edges.extend(
            [
                edges_right_deep,
                edges_down_right,
                edges_down_deep,
                edges_down_left,
                edges_down_shallow,
                edges_deep_left,
            ]
        )

    if connectivity == 26:
        edges_down_right_deep = np.vstack(
            (vertices[:-1, :-1, :-1].ravel(), vertices[1:, 1:, 1:].ravel())
        )
        edges_down_left_deep = np.vstack(
            (vertices[:-1, 1:, :-1].ravel(), vertices[1:, :-1, 1:].ravel())
        )
        edges_down_right_shallow = np.vstack(
            (vertices[:-1, :-1, 1:].ravel(), vertices[1:, 1:, :-1].ravel())
        )
        edges_down_left_shallow = np.vstack(
            (vertices[:-1, 1:, 1:].ravel(), vertices[1:, :-1, :-1].ravel())
        )

        edges.extend(
            [
                edges_down_right_deep,
                edges_down_left_deep,
                edges_down_right_shallow,
                edges_down_left_shallow,
            ]
        )

    edges = np.hstack(edges)
    return edges
