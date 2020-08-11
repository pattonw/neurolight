from gunpowder import (
    BatchFilter,
    BatchRequest,
    Batch,
    ArrayKey,
    ArraySpec,
    Array,
    GraphKey,
    GraphSpec,
    Coordinate,
)

from scipy.spatial import cKDTree
import numpy as np

from typing import Dict
import itertools
import logging

logger = logging.getLogger(__file__)


class Neighborhood(BatchFilter):
    """
    Node for creating a "neighborhood" of shape descriptors for a graph.
    Iterates over connected components of G and calculates a
    "shape descriptor" for every node n in the connected component
    by sending out `k` rays with length of `distance`. From each
    of these ray end points, calculate the minimum distance back
    to the connected component to which it belongs.

    The shape descriptors are directly stored in an Array.

    Distances from end points back to the connected component are normalized
    by the length of the rays, `distance`. Thus a single point would have
    a shape discriptor of length `k` of all ones since the maximum distance
    from the end of a ray back to the connected component, cannot be longer
    than the distance to the point from which it came.

    Args:

        gt (``GraphKey``):

            The `GraphKey` corresponding to the graph for which
            you want descriptors.

        neighborhood (``ArrayKey``):

            The Array in which the "shape descriptors" will be stored.

        neighborhood_mask (``ArrayKey``):

            An Array that will store a binary mask of true on voxels where
            the shape descriptors are unambiguous. This corresponds only
            to voxels containing nodes from at most 1 connected component

        distance (``float``):

            The length of the "rays" extended from each node.

        k (``int``):

            The number of "rays" shot out from each node.
            Options are:
            6:
                6: (1 ray along each axis (+ and -))
            26:
                6: (1 ray along each axis (+ and -))
                12: (4 diagonals on each plane defined by a pair of axes)
                8: (1 ray through the center of each octant)

        array_specs (``dict``, ``ArrayKey`` -> ``ArraySpec``):

            A dictionary of ArraySpecs for neighborhood and neighborhood_mask.
            Useful if you want to assign voxel size or override the default
            dtypes of float32 for neighborhood or bool for neighborhood_mask
            or adjust the provided ``Roi``'s
    

    """

    def __init__(
        self,
        gt: GraphKey,
        neighborhood: ArrayKey,
        neighborhood_mask: ArrayKey,
        distance: float,
        k: int = 26,
        array_specs: Dict[ArrayKey, ArraySpec] = None,
    ):
        self.gt = gt
        self.neighborhood = neighborhood
        self.neighborhood_mask = neighborhood_mask
        self.k = k
        self.distance = distance
        self.array_specs = array_specs if array_specs is not None else {}
        assert self.k in [
            6,
            26,
        ], "Only supported neighborhood k's are 6 and 26 at the moment"

    def setup(self):
        provided_roi = self.spec[self.gt].roi
        neighborhood_spec = self.array_specs.get(self.neighborhood, ArraySpec())
        if neighborhood_spec.roi is None:
            neighborhood_spec.roi = provided_roi
        if neighborhood_spec.dtype is None:
            neighborhood_spec.dtype = np.float32
        self.array_specs[self.neighborhood] = neighborhood_spec
        self.provides(self.neighborhood, neighborhood_spec)
        neighborhood_mask_spec = self.array_specs.get(
            self.neighborhood_mask, ArraySpec()
        )
        if neighborhood_mask_spec.roi is None:
            neighborhood_mask_spec.roi = provided_roi
        if neighborhood_mask_spec.dtype is None:
            neighborhood_mask_spec.dtype = bool
        self.array_specs[self.neighborhood_mask] = neighborhood_mask_spec
        self.provides(self.neighborhood_mask, neighborhood_mask_spec)

    def prepare(self, request):
        deps = BatchRequest()

        request[
            self.neighborhood_mask
        ].roi, f"Requested {self.neighborhood} and {self.neighborhood_mask} with different roi's"

        request_roi = request[self.neighborhood].roi
        grow_distance = Coordinate(
            (np.ceil(self.distance),) * len(request_roi.get_shape())
        )
        request_roi = request_roi.grow(grow_distance, grow_distance)

        deps[self.gt] = GraphSpec(roi=request_roi)

        return deps

    def process(self, batch, request):
        gt = batch[self.gt]
        voxel_size = self.spec[self.neighborhood].voxel_size
        request_roi = request[self.neighborhood].roi
        if voxel_size is None:
            voxel_size = Coordinate((1,) * len(gt.spec.roi.get_shape()))

        neighborhood = np.zeros(
            (self.k,) + (request_roi.get_shape() / voxel_size), dtype=np.float32
        )
        neighborhood_mask = np.zeros((request_roi.get_shape() / voxel_size), dtype=int)

        k_neighborhood_offsets = self.get_neighborhood_offsets()
        for i, connected_component in enumerate(gt.connected_components):
            component_id = i + 1
            node_locations = [
                gt.node(node_id).location for node_id in connected_component
            ]
            component_kdtree = cKDTree(node_locations)
            for node_id in connected_component:
                node = gt.node(node_id)
                location = node.location
                if request_roi.contains(location):
                    query_points = k_neighborhood_offsets + location
                    query_neighbors = component_kdtree.query(query_points, 1)[0]
                    voxel_index = Coordinate(
                        (location - request_roi.get_offset()) // voxel_size
                    )
                    neighborhood[(slice(None),) + voxel_index] = (
                        query_neighbors / self.distance
                    )
                    if neighborhood_mask[voxel_index] == 0:
                        neighborhood_mask[voxel_index] = component_id
                    elif neighborhood_mask[voxel_index] != component_id:
                        neighborhood_mask[voxel_index] = -1

        neighborhood_mask = neighborhood_mask > 0

        outputs = Batch()
        neighborhood_spec = self.array_specs[self.neighborhood].copy()
        neighborhood_spec.roi = request_roi
        outputs[self.neighborhood] = Array(neighborhood, neighborhood_spec)
        neighborhood_mask_spec = self.array_specs[self.neighborhood_mask].copy()
        neighborhood_mask_spec.roi = request_roi
        outputs[self.neighborhood_mask] = Array(
            neighborhood_mask > 0, neighborhood_mask_spec
        )
        return outputs

    def get_neighborhood_offsets(self):
        if self.k == 6:
            return (
                np.array(
                    [
                        (i, j, k)
                        for i, j, k in itertools.product(*((-1, 0, 1),) * 3)
                        if np.isclose(abs(i) + abs(j) + abs(k), 1)
                    ]
                )
                * self.distance
            )
        elif self.k == 26:
            # 6 axis aligned vectors
            neighborhood_6 = [
                (i, j, k)
                for i, j, k in itertools.product(*((-1, 0, 1),) * 3)
                if np.isclose(abs(i) + abs(j) + abs(k), 1)
            ]
            # 12 axis pair plane aligned 45 degree rotations
            neighborhood_halves = [
                (i, j, k)
                for i, j, k in itertools.product(*((-2 ** (-0.5), 0, 2 ** (-0.5)),) * 3)
                if np.isclose((i ** 2 + j ** 2 + k ** 2) ** 0.5, 1)
            ]
            # 8 vectors going through each triangle face if you made a concave surface
            # around the 6 neighborhood
            neighborhood_diagonals = [
                (i, j, k)
                for i, j, k in itertools.product(*((-3 ** (-0.5), 3 ** (-0.5)),) * 3)
                if np.isclose((i ** 2 + j ** 2 + k ** 2) ** 0.5, 1)
            ]

            assert len(neighborhood_6) == 6, f"neighborhood_6: {neighborhood_6}"
            assert (
                len(neighborhood_halves) == 12
            ), f"neighborhood_halves: {neighborhood_halves}"
            assert (
                len(neighborhood_diagonals) == 8
            ), f"neighborhood_diagonals: {neighborhood_diagonals}"

            return (
                np.array(neighborhood_6 + neighborhood_halves + neighborhood_diagonals)
                * self.distance
            )

        else:
            raise NotImplementedError("Should not be reachable")
