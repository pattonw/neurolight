import numpy as np
from gunpowder import (
    ProviderSpec,
    BatchProvider,
    BatchRequest,
    Batch,
    Roi,
    Points,
    GraphPoints,
    Array,
    ArrayKey,
    Coordinate,
    PointsKey,
)
from gunpowder.profiling import Timing
from typing import Tuple, Optional
import logging
import time
import copy

logger = logging.getLogger(__name__)


class GetNeuronPair(BatchProvider):
    """Retrieves a pair of neurons represented by both points in an swc style
        Point set and volumetric image data.

        Args:

            point_keys (:class:``Tuple[PointsKey]``):

                A pair of PointsKeys from which two sets of points will be taken.
                If a single PointsKey is provided, both sets of points will be
                taken from the same dataset

            array_keys (:class:``Tuple[ArraySpec]``):

                A pair of ArrayKeys from which two sets of images will be taken.
                If a single ArrayKey is provided, both sets of images will be
                taken from the same dataset

            outputKeys (:class:``List[Tuple[PointsKey, ArrayKey]]``):
                The output keys in the form [(PointsKey, ArrayKey), (PointsKey, ArrayKey)].
                Where each Pair will have matching rois

            seperate_by (:class:``float``, optional):
                Distance in world units between the two center nodes.
                It is guaranteed that no other points in two point sets will be closer than
                this distance.
                This distance will be approximated since the volumetric data will be shifted
                on its grid.

        """

    def __init__(
        self,
        point_source: PointsKey,
        array_source: ArrayKey,
        label_source: ArrayKey,
        points: Tuple[PointsKey, PointsKey],
        arrays: Tuple[ArrayKey, ArrayKey],
        labels: Tuple[ArrayKey, ArrayKey],
        seperate_by: float = 1.0,
        shift_attempts: int = 3,
        request_attempts: int = 3,
        spec: ProviderSpec = None,
    ):
        self.point_source = point_source
        self.array_source = array_source
        self.label_source = label_source
        self.points = points
        self.arrays = arrays
        self.labels = labels
        self.seperate_by = seperate_by

        self.shift_attempts = shift_attempts
        self.request_attempts = request_attempts

    @property
    def seed(self) -> int:
        return hash(
            hash(self.points + self.arrays + self.labels) + hash(time.time() * 1e6)
        )

    @property
    def upstream_provider(self):
        return self.get_upstream_providers()[0]

    @property
    def spec(self) -> ProviderSpec:
        return self.upstream_provider.spec

    def setup(self):
        """
        Provide the two copies of the upstream points and images
        """
        for point_key, array_key, label_key in zip(
            self.points, self.arrays, self.labels
        ):
            self.provides(point_key, self.spec[self.point_source])
            self.provides(array_key, self.spec[self.array_source])
            self.provides(label_key, self.spec[self.label_source])

    def prepare_points(self, request: BatchRequest) -> Tuple[BatchRequest, int]:
        """
        Only request the points, keeping track of the seed
        """

        growth = self._get_growth()
        seed = self.seed

        seed_deps = BatchRequest(random_seed=seed)

        spec = copy.deepcopy(request[self.points[0]])
        spec.roi = spec.roi.grow(growth, growth)
        seed_deps[self.point_source] = spec

        return seed_deps, seed

    def prepare(self, request: BatchRequest, seed: int) -> Tuple[BatchRequest, int]:
        """
        Only request everything with the given seed
        """
        growth = self._get_growth()

        deps = BatchRequest(random_seed=seed)

        deps[self.point_source] = copy.deepcopy(request[self.points[0]])
        deps[self.point_source].roi = deps[self.point_source].roi.grow(growth, growth)
        deps[self.array_source] = copy.deepcopy(request[self.arrays[0]])
        deps[self.array_source].roi = deps[self.array_source].roi.grow(growth, growth)
        deps[self.label_source] = copy.deepcopy(request[self.labels[0]])
        deps[self.label_source].roi = deps[self.label_source].roi.grow(growth, growth)

        return deps

    def provide(self, request: BatchRequest) -> Batch:
        """
        First request points with specific seeds, then request the rest if
        valid points are found.
        """

        timing_prepare = Timing(self, "prepare")
        timing_prepare.start()

        base_seed, add_seed, direction = self.get_valid_seeds(request)
        request_base = self.prepare(request, base_seed)
        request_add = self.prepare(request, add_seed)

        base = self.upstream_provider.request_batch(request_base)
        add = self.upstream_provider.request_batch(request_add)

        timing_prepare.stop()

        timing_process = Timing(self, "process")
        timing_process.start()

        base = self.process(base, direction, request=request, batch_index=0)
        add = self.process(add, -direction, request=request, batch_index=1)

        assert self._valid_pair(
            base[self.point_source].graph, add[self.point_source].graph
        ), "Seeded request produced an invalid pair!"

        batch = self.merge_batches(base, add)

        timing_process.stop()

        return batch

    def merge_batches(self, base: Batch, add: Batch) -> Batch:
        combined = Batch()

        base_map = {
            self.point_source: self.points[0],
            self.array_source: self.arrays[0],
            self.label_source: self.labels[0],
        }
        for key, value in base.items():
            combined[base_map.get(key, key)] = value

        add_map = {
            self.point_source: self.points[1],
            self.array_source: self.arrays[1],
            self.label_source: self.labels[1],
        }
        for key, value in add.items():
            combined[add_map.get(key, key)] = value

        return combined

    def get_valid_seeds(self, request: BatchRequest) -> Tuple[int, int, Coordinate]:
        """
        Request pairs of point sets, making sure that you can seperate the
        two. If it is possible to seperate the two then keep those seeds, else
        try again with new seeds.
        """

        for _ in range(self.request_attempts):
            points_request_base, base_seed = self.prepare_points(request)
            points_request_add, add_seed = self.prepare_points(request)
            try:
                direction = self.check_valid_requests(
                    points_request_base, points_request_add, request
                )
                return base_seed, add_seed, direction  # successful return
            except ValueError:
                logging.debug("Request with seeds {} and {} failed!")
        raise ValueError(
            "Failed {} attempts to retrieve a pair of neurons!".format(
                self.request_attempts
            )
        )

    def check_valid_requests(
        self,
        points_request_base: BatchRequest,
        points_request_add: BatchRequest,
        request: BatchRequest,
    ) -> Coordinate:
        """
        check if the points returned by these requests are seperable.
        If they are return the direction vector used to seperate them.
        """
        points_base = self.upstream_provider.request_batch(points_request_base)

        points_add = self.upstream_provider.request_batch(points_request_add)

        for _ in range(self.shift_attempts):
            direction = self.random_direction()
            points_base = self.process(
                points_base, direction, request=request, batch_index=0
            )
            points_add = self.process(
                points_add, -direction, request=request, batch_index=0
            )
            if self._valid_pair(
                points_base[self.point_source].graph,
                points_add[self.point_source].graph,
            ):
                return direction

        raise ValueError("Failed to seperate these points")

    def random_direction(self) -> Coordinate:
        # there should be a better way of doing this. If distances are small,
        # the rounding may make a big difference. Rounding (1.3, 1.3, 1.3) would
        # give a euclidean distance of ~1.7 instead of ~2.25
        # A better solution might be to do a distance transform on an array with a centered
        # dot and then find all potential moves that have a distance equal to delta +- 0.5
        voxel_size = np.array(self.spec[self.array_source].voxel_size)  # should be lcm
        random_direction = np.random.randn(3)
        random_direction /= np.linalg.norm(random_direction)  # unit vector
        random_direction *= self.seperate_by  # physical units
        random_direction = (
            (np.round(random_direction / voxel_size) + 1) // 2
        ) * voxel_size  # physical units rounded to nearest voxel size
        random_direction = Coordinate(random_direction)
        return random_direction

    def process(
        self,
        batch: Batch,
        direction: Coordinate,
        request: BatchRequest,
        batch_index: int,
    ) -> Batch:
        points = batch.points[self.point_source]
        array = batch.arrays.get(self.array_source, None)
        label = batch.arrays.get(self.label_source, None)

        # Shift and crop points and array
        points, array, label = self._shift_and_crop(
            points,
            array,
            label,
            direction=direction,
            request=request,
            batch_index=batch_index,
        )
        batch.points[self.point_source] = points
        if array is not None:
            batch.arrays[self.array_source] = array
        if label is not None:
            batch.arrays[self.label_source] = label

        return batch

    def _shift_and_crop(
        self,
        points: Points,
        array: Optional[Array],
        labels: Optional[Array],
        direction: Coordinate,
        request: BatchRequest,
        batch_index: int,
    ) -> Tuple[Points, Optional[Array], Optional[Array]]:

        points = self._shift_and_crop_points(
            points, direction, request[self.points[batch_index]].roi
        )
        if array is not None:
            array = self._shift_and_crop_array(
                array, direction, request[self.arrays[batch_index]].roi
            )
        if labels is not None:
            labels = self._shift_and_crop_array(
                labels, direction, request[self.labels[batch_index]].roi
            )

        return points, array, labels

    def _extract_roi(self, larger, smaller, direction):
        center = larger.get_offset() + larger.get_shape() // 2
        centered_offset = center - smaller.get_shape() // 2
        centered_smaller_roi = Roi(centered_offset, smaller.get_shape())
        return centered_smaller_roi.shift(direction)

    def _shift_and_crop_array(self, array, direction, output_roi):
        # Shift and crop the array
        shifted_smaller_roi = self._extract_roi(array.spec.roi, output_roi, direction)

        array = array.crop(shifted_smaller_roi)
        array.spec.roi = output_roi
        return array

    def _shift_and_crop_points(self, points, direction, output_roi):
        # Shift and crop the array
        shifted_smaller_roi = self._extract_roi(points.spec.roi, output_roi, direction)

        point_graph = points.graph
        point_graph.crop(shifted_smaller_roi)
        point_graph.shift(-shifted_smaller_roi.get_offset())
        points = GraphPoints._from_graph(point_graph)
        points.spec.roi = output_roi
        return points

    def _get_growth(self):
        """
        The amount by which the volumes need to be expanded to accomodate
        cropping the two volumes such that the center voxels of each volume
        are a certain distance apart.

        Assuming you want a distance of 3 voxels, each volume must be expanded
        by 4.
        """
        voxel_size = np.array(self.spec[self.array_source].voxel_size)
        distance = np.array((self.seperate_by,) * 3)
        # voxel shift is rounded up to the nearest voxel in each axis
        voxel_shift = (distance + voxel_size - 1) // voxel_size
        # expand positive and negative sides enough to contain any desired shift
        half_shift = (voxel_shift + 1) // 2
        return Coordinate(half_shift * 2 * voxel_size)

    def _valid_pair(self, base_graph, add_graph):
        """
        Simply checks for every pair of points, is the distance between them
        greater than the desired seperation criterion.
        """
        voxel_size = np.array(self.spec[self.array_source].voxel_size)

        min_dist = self.seperate_by + 2 * voxel_size.mean()
        for point_id_base, point_base in base_graph.nodes.items():
            for point_id_add, point_add in add_graph.nodes.items():
                min_dist = min(
                    np.linalg.norm(point_base["location"] - point_add["location"]),
                    min_dist,
                )
        if (
            self.seperate_by - voxel_size.max()
            <= min_dist
            <= self.seperate_by + voxel_size.max()
        ):
            logger.debug(("Got a min distance of {}").format(min_dist))
            return True
        else:
            logger.debug(
                (
                    "expected a minimum distance between the two neurons"
                    + "to be in the range ({}, {}), however saw a min distance of {}"
                ).format(
                    self.seperate_by - voxel_size.max(),
                    self.seperate_by + voxel_size.max(),
                    min_dist,
                )
            )
            return False
