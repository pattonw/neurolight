import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from gunpowder import (
    ProviderSpec,
    BatchProvider,
    BatchRequest,
    Batch,
    Roi,
    Points,
    PointsSpec,
    GraphPoints,
    Array,
    ArrayKey,
    Coordinate,
    PointsKey,
)
from gunpowder.profiling import Timing, ProfilingStats
from typing import Tuple, Optional
import logging
import time
import copy
import itertools

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
        seperate_by: Tuple[float, float] = (0.0, 1.0),
        shift_attempts: int = 50,
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
        return (
            hash(self.points + self.arrays + self.labels) + hash(time.time() * 1e6)
        ) % (2 ** 32)

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

        dps = BatchRequest(random_seed=seed)

        if any([points in request for points in self.points]):
            dps[self.point_source] = request.points_specs.get(
                self.points[0], request[self.points[1]]
            )
        elif any([array in request for array in self.arrays]):
            dps[self.point_source] = PointsSpec(
                roi=request.array_specs.get(self.arrays[0], request[self.arrays[1]]).roi
            )
        elif any([labels in request for labels in self.labels]):
            dps[self.points_source] = PointsSpec(
                roi=request.array_specs.get(self.labels[0], request[self.labels[1]]).roi
            )
        else:
            raise ValueError(
                "One of the following must be requested: {}, {}, {}".format(
                    self.points, self.arrays, self.labels
                )
            )

        dps[self.point_source].roi = dps[self.point_source].roi.grow(growth, growth)
        dps.place_holders[self.array_source] = copy.deepcopy(request[self.arrays[0]])
        dps.place_holders[self.array_source].roi = dps.place_holders[
            self.array_source
        ].roi.grow(growth, growth)
        dps.place_holders[self.label_source] = copy.deepcopy(request[self.labels[0]])
        dps.place_holders[self.label_source].roi = dps.place_holders[
            self.label_source
        ].roi.grow(growth, growth)

        return dps, seed

    def prepare(self, request: BatchRequest, seed: int) -> Tuple[BatchRequest, int]:
        """
        Only request everything with the given seed
        """
        growth = self._get_growth()

        dps = BatchRequest(random_seed=seed)

        if any([points in request for points in self.points]):
            dps[self.point_source] = copy.deepcopy(request[self.points[0]])
            dps[self.point_source].roi = dps[self.point_source].roi.grow(growth, growth)
        if any([array in request for array in self.arrays]):
            dps[self.array_source] = copy.deepcopy(request[self.arrays[0]])
            dps[self.array_source].roi = dps[self.array_source].roi.grow(growth, growth)
        if any([labels in request for labels in self.labels]):
            dps[self.label_source] = copy.deepcopy(request[self.labels[0]])
            dps[self.label_source].roi = dps[self.label_source].roi.grow(growth, growth)

        return dps

    def provide(self, request: BatchRequest) -> Batch:
        """
        First request points with specific seeds, then request the rest if
        valid points are found.
        """

        base_seed, add_seed, direction, prepare_profiling_stats = self.get_valid_seeds(
            request
        )

        timing_prepare = Timing(self, "prepare")
        timing_prepare.start()

        request_base = self.prepare(request, base_seed)
        request_add = self.prepare(request, add_seed)

        timing_prepare.stop()

        base = self.upstream_provider.request_batch(request_base)
        add = self.upstream_provider.request_batch(request_add)

        timing_process = Timing(self, "process")
        timing_process.start()

        base = self.process(base, direction, request=request, batch_index=0)
        add = self.process(add, -direction, request=request, batch_index=1)

        batch = self.merge_batches(base, add)

        logger.debug("get neuron pair got {}".format(batch))

        timing_process.stop()
        batch.profiling_stats.merge_with(prepare_profiling_stats)
        batch.profiling_stats.add(timing_prepare)
        batch.profiling_stats.add(timing_process)

        return batch

    def merge_batches(self, base: Batch, add: Batch) -> Batch:
        combined = Batch()
        combined.profiling_stats.merge_with(base.profiling_stats)
        combined.profiling_stats.merge_with(add.profiling_stats)

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

        profiling_stats = ProfilingStats()

        k = 0
        while True:
            k += 1
            if k > 10:
                raise ValueError("Failed to retrieve a pair of neurons!")

            points_request_base, base_seed = self.prepare_points(request)

            base_batch = self.upstream_provider.request_batch(points_request_base)
            profiling_stats.merge_with(base_batch.profiling_stats)

            timing_prepare_base = Timing("prepare base")
            timing_prepare_base.start()

            distance_transform, gradients = self.prepare_base(
                base_batch[self.point_source]
            )

            for _ in range(self.request_attempts):
                points_request_add, add_seed = self.prepare_points(request)
                direction = self.check_valid_requests(
                    distance_transform,
                    gradients,
                    points_request_add,
                    request,
                    profiling_stats,
                )
                if direction is not None:
                    return base_seed, add_seed, direction, profiling_stats
                else:
                    continue

    def prepare_base(self, points_base):
        logger.debug("preparing base dist and gradients")
        base_roi = points_base.spec.roi
        voxel_size = self.spec.get_lcm_voxel_size()
        distances = np.ones(base_roi.get_shape() / voxel_size)
        for point_id, point in points_base.data.items():
            distances[
                (Coordinate(point.location) - base_roi.get_begin()) / voxel_size
            ] = 0
        distances = distance_transform_edt(distances, sampling=voxel_size)
        gradients = np.gradient(distances, *voxel_size)
        gradients = [
            gaussian_filter(gradient, sigma=10 / np.array(voxel_size))
            for gradient in gradients
        ]
        gradients = np.stack(gradients, axis=-1)
        logger.debug("finished preparing base dist and gradients")

        return distances, gradients

    def check_valid_requests(
        self,
        distances: np.ndarray,
        gradients: np.ndarray,
        points_request_add: BatchRequest,
        request: BatchRequest,
        profiling_stats: ProfilingStats,
    ) -> Optional[Coordinate]:
        """
        check if the points returned by these requests are seperable.
        If they are return the direction vector used to seperate them.
        """
        points_add = self.upstream_provider.request_batch(points_request_add)
        voxel_size = self.spec.get_lcm_voxel_size()

        profiling_stats.merge_with(points_add.profiling_stats)

        timing_process_points = Timing("process points")
        timing_process_points.start()
        logger.debug(
            "Points add has {} points".format(len(points_add[self.point_source].data))
        )
        logger.debug(
            "gradients shape: {}, accessing: {}".format(
                gradients.shape, (np.array(gradients.shape[0:3]) // 2)
            )
        )
        gradient = gradients[tuple(np.array(gradients.shape[0:3]) // 2)]
        gradient = gradient / np.linalg.norm(gradient)
        logger.debug("moving along gradient: {}".format(gradient))

        for i in range(self.shift_attempts):
            start = self.seperate_by[0] / self.seperate_by[1]
            slider = i / (self.shift_attempts - 1)
            fraction = start + slider * (1 - start)

            logger.debug("attempting shift {} of {}".format(i, self.shift_attempts))
            direction = Coordinate(np.array(self._get_growth()) * gradient * fraction)
            points_add_shifted = self.process(
                points_add, -direction, request=request, batch_index=1, inplace=False
            )
            return_roi = self._return_roi(request)
            center = (Coordinate(gradients.shape[0:3]) // 2) * voxel_size
            radius = (return_roi.get_shape() // voxel_size) // 2 * voxel_size
            pad = (
                Coordinate(
                    (np.array((return_roi.get_shape() // voxel_size)) % 2).tolist()
                )
                * voxel_size
            )
            slices = tuple(
                map(
                    slice,
                    (center - radius + direction) // voxel_size,
                    (center + radius + direction + voxel_size + pad) // voxel_size,
                )
            )
            logger.debug("slices: {}".format(slices))
            if self._valid_pair(
                points_add_shifted[self.point_source].graph, distances[slices]
            ):
                logger.info("Valid shift found: {}".format(direction))

                timing_process_points.stop()
                profiling_stats.add(timing_process_points)
                return direction
        logger.info("Request failed. New Request!")

        timing_process_points.stop()
        profiling_stats.add(timing_process_points)
        return None

    def random_direction(self) -> Coordinate:
        # there should be a better way of doing this. If distances are small,
        # the rounding may make a big difference. Rounding (1.3, 1.3, 1.3) would
        # give a euclidean distance of ~1.7 instead of ~2.25
        # A better solution might be to do a distance transform on an array with a centered
        # dot and then find all potential moves that have a distance equal to delta +- 0.5
        voxel_size = np.array(self.spec[self.array_source].voxel_size)  # should be lcm
        random_direction = np.random.randn(len(voxel_size))
        random_direction /= np.linalg.norm(random_direction)  # unit vector
        random_direction *= np.max(self.seperate_by)  # physical units
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
        inplace=True,
    ) -> Batch:
        if not inplace:
            batch = copy.deepcopy(batch)

        logger.debug("processing")
        points = batch.points.get(self.point_source, None)
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
        if points is not None:
            batch.points[self.point_source] = points
        if array is not None:
            batch.arrays[self.array_source] = array
        if label is not None:
            batch.arrays[self.label_source] = label

        return batch

    def _shift_and_crop(
        self,
        points: Optional[Points],
        array: Optional[Array],
        labels: Optional[Array],
        direction: Coordinate,
        request: BatchRequest,
        batch_index: int,
    ) -> Tuple[Points, Optional[Array], Optional[Array]]:

        roi = self._return_roi(request)
        if points is not None:
            points = self._shift_and_crop_points(points, direction, roi)
        if array is not None:
            array = self._shift_and_crop_array(array, direction, roi)
        if labels is not None:
            labels = self._shift_and_crop_array(labels, direction, roi)

        return points, array, labels

    def _return_roi(self, request):
        potential_requests = list(
            itertools.chain(self.points, self.arrays, self.labels)
        )
        contained = [p for p in potential_requests if p in request]
        rois = [request[r].roi for r in contained]
        for i, roi_a in enumerate(rois):
            for j, roi_b in enumerate(rois[i + 1 :]):
                assert roi_a == roi_b, (
                    "points, arrays, and labels should all have "
                    + "the same roi, but we see {} and {}"
                ).format(roi_a, roi_b)
        return rois[0]

    def _extract_roi(self, larger, smaller, direction):
        center = larger.get_offset() + larger.get_shape() // 2
        centered_offset = center - smaller.get_shape() // 2
        centered_smaller_roi = Roi(centered_offset, smaller.get_shape())
        return centered_smaller_roi.shift(direction)

    def _shift_and_crop_array(self, array, direction, output_roi):
        # Shift and crop the array
        shifted_smaller_roi = self._extract_roi(array.spec.roi, output_roi, direction)

        array = array.crop(shifted_smaller_roi)
        array.spec.roi = (
            output_roi
            if output_roi is not None
            else Roi((0,) * len * array.data.shape, array.data.shape)
        )
        return array

    def _shift_and_crop_points(self, points, direction, output_roi):
        # Shift and crop the array
        shifted_smaller_roi = self._extract_roi(points.spec.roi, output_roi, direction)

        point_graph = points.graph
        point_graph.crop(shifted_smaller_roi)
        point_graph.shift(-shifted_smaller_roi.get_offset())
        output_roi = (
            output_roi if output_roi is not None else Roi((None,) * 3, (None,) * 3)
        )
        points = GraphPoints._from_graph(point_graph, PointsSpec(roi=output_roi))
        return points

    def _get_growth(self):
        """
        The amount by which the volumes need to be expanded to accomodate
        cropping the two volumes such that the center voxels of each volume
        are a certain distance apart.
        """
        voxel_size = np.array(self.spec[self.array_source].voxel_size)
        distance = np.array((np.max(self.seperate_by),) * len(voxel_size))
        # voxel shift is rounded up to the nearest voxel in each axis
        voxel_shift = (distance + voxel_size - 1) // voxel_size
        # expand positive and negative sides enough to contain any desired shift
        half_shift = (voxel_shift + 1) // 2
        return Coordinate(half_shift * 2 * voxel_size)

    def _valid_pair(self, add_graph, distances):
        """
        Simply checks for every pair of points, is the distance between them
        greater than the desired seperation criterion.
        """
        min_dist = float("inf")
        voxel_size = self.spec.get_lcm_voxel_size()
        for point_id, point_attrs in add_graph.nodes.items():
            if distances[Coordinate(point_attrs["location"] // voxel_size)] < min_dist:
                min_dist = distances[Coordinate(point_attrs["location"] // voxel_size)]
        if min_dist < self.seperate_by[0] or min_dist > self.seperate_by[1]:
            logger.debug(
                (
                    "expected a minimum distance between the two neurons "
                    + "to be in the range ({}, {}), however saw a distance of {}"
                ).format(self.seperate_by[0], self.seperate_by[1], min_dist)
            )
            return False
        else:
            logger.info(f"Saw min dist of {min_dist}!")
            return True
