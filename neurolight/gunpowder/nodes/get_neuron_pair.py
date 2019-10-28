import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import cKDTree
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
from typing import Tuple, Optional, Dict, Union
import logging
import time
import copy
import itertools

import pickle
from pathlib import Path

DataKey = Union[PointsKey, ArrayKey]

logger = logging.getLogger(__name__)


class GetNeuronPair(BatchProvider):
    """
        GetNeuronPair is given a point_source, array_source, and label_source. From
        Two requests are made to each source to get two copies of each input.

        Since we need to be able to specify a certain distance between the two neurons
        this node also handles this need efficiently by first requesting only the points
        in a larger field of view and then finding a subregions of those arrays such that
        the minimum distance between the two sets of points is within a user defined
        acceptable range.

        Args:

            point_source:

                A Key containing a set of points. This key will be queried twice
                and stored using the keys in `point_keys`

            array_source:

                A key containing the voxel data counterpart to the point_source.
                Will also be queried twice and its values split between the keys
                in `array_keys`.

            label_source:

                A Key containing the voxel wise labels corresponding to the values
                in `array_source`. Will Also be queried twice.

            points:

                A pair of PointsKeys into which two sets of points will be stored.

            arrays:

                A pair of ArrayKeys into which two sets of images will be stored.

            labels:

                A pair of ArrayKeys into which two sets of labels will be stored.

            nonempty_placeholder:

                The need for a nonempty placeholder arises from the following problem:

                We first request larger rois to hopefully guarantee we can obtain subsets
                that contain points sufficiently far away from each other.

                Roi A: |------------------------------------------------------------|
                Roi B: |------------------------------------------------------------|

                Disregarding the location of points for now, assume we found valid sub rois:

                Roi a: |------------------|
                Roi b:                                   |------------------|

                Now we could simply request the whole Roi A and crop once we have it,
                or we could make a smaller request. However we assume there is a random
                location node upstream of us.
                We generally want to ensure nonempty on the the points, which means the
                randomness from the random location comes from randomly picking a point,
                and adding the sufficient shift to bring the requested roi to that point.
                Thus, regardless of what shift you apply to your roi of interest, the
                RandomLocation node would pick the same point, and provide a shift negating
                any change you make. The solution is to provide an placeholder, which
                will be assigned Roi A. This means that random location will pick the same
                point and the same shift to return you to the same location, but now you
                can request any smaller roi relative to the larger and it will be
                appropriately shifted.

            seperate_by:

                A range of acceptable minimum distances between the two sets of points.
                Note:
                1) It is only guaranteed that the points in nonempty_placeholder are
                seperated by this distance if provided. This can greatly save on time
                if processing points is expensive, but may result in slightly different
                distances.
                2) Distances may not be exactly replicable. Elastic augment for example
                will create different transformations for different roi sizes, not only
                different random seeds. This means that the smaller request rois may result
                in different transforms applied on the first pass checking the distance,
                and the second pass that returns all of your data.

            shift_attempts:

                The number of attempts to shift the fov's in the larger rois to get smaller
                rois that are seperated. It may be beneficial to use some multiple of the
                seperate distance since each shift has a step size constraint which might
                make it difficult to seperate by a sufficient distance given too few steps

            request_attempts:

                It is possible that two point sets cannot be seperated by the desired distance.
                In this case, we will keep roi A, but generate a new roi B and try again.
                request_attempts gives the maximum number of times a new request will be made.

            spec

        """

    def __init__(
        self,
        point_source: PointsKey,
        array_source: ArrayKey,
        label_source: ArrayKey,
        points: Tuple[PointsKey, PointsKey],
        arrays: Tuple[ArrayKey, ArrayKey],
        labels: Tuple[ArrayKey, ArrayKey],
        nonempty_placeholder: Optional[PointsKey] = None,
        seperate_by: Tuple[float, float] = (0.0, 1.0),
        shift_attempts: int = 50,
        request_attempts: int = 3,
        extra_keys: Optional[Dict[DataKey, Tuple[DataKey, DataKey]]] = {},
    ):
        self.point_source = point_source
        self.nonempty_placeholder = nonempty_placeholder
        self.array_source = array_source
        self.label_source = label_source
        self.points = points
        self.arrays = arrays
        self.labels = labels
        self.extra_keys = extra_keys
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
        for source, targets in self.extra_keys.items():
            self.provides(targets[0], self.spec[source])
            self.provides(targets[1], self.spec[source])

    def prepare_points(self, request: BatchRequest) -> Tuple[BatchRequest, int]:
        """
        Request either point_source or nonempty_placeholder
        """

        growth = self._get_growth()
        seed = self.seed

        dps = BatchRequest(random_seed=seed)

        point_key = (
            self.point_source
            if self.nonempty_placeholder is None
            else self.nonempty_placeholder
        )

        if any([points in request for points in self.points]):
            dps[point_key] = request.points_specs.get(
                self.points[0], request[self.points[1]]
            )
        elif any([array in request for array in self.arrays]):
            dps[point_key] = PointsSpec(
                roi=request.array_specs.get(self.arrays[0], request[self.arrays[1]]).roi
            )
        elif any([labels in request for labels in self.labels]):
            dps[point_key] = PointsSpec(
                roi=request.array_specs.get(self.labels[0], request[self.labels[1]]).roi
            )
        else:
            raise ValueError(
                "One of the following must be requested: {}, {}, {}".format(
                    self.points, self.arrays, self.labels
                )
            )

        dps[point_key].roi = dps[point_key].roi.grow(growth, growth)

        dps.place_holders[self.array_source] = copy.deepcopy(request[self.arrays[0]])
        dps.place_holders[self.array_source].roi = dps.place_holders[
            self.array_source
        ].roi.grow(growth, growth)
        dps.place_holders[self.label_source] = copy.deepcopy(request[self.labels[0]])
        dps.place_holders[self.label_source].roi = dps.place_holders[
            self.label_source
        ].roi.grow(growth, growth)

        return dps, seed

    def prepare(
        self, request: BatchRequest, seed: int, direction: Coordinate
    ) -> Tuple[BatchRequest, int]:
        """
        Only request everything with the given seed
        """
        dps = BatchRequest(random_seed=seed)

        if self.nonempty_placeholder is not None:
            # Handle nonempty placeholder
            growth = self._get_growth()

            if any([points in request for points in self.points]):
                dps.place_holders[self.nonempty_placeholder] = copy.deepcopy(
                    request.points_specs.get(self.points[0], request[self.points[1]])
                )
            elif any([array in request for array in self.arrays]):
                dps.place_holders[self.nonempty_placeholder] = copy.deepcopy(
                    PointsSpec(
                        roi=request.array_specs.get(
                            self.arrays[0], request[self.arrays[1]]
                        ).roi
                    )
                )
            elif any([labels in request for labels in self.labels]):
                dps.place_holders[self.nonempty_placeholder] = copy.deepcopy(
                    PointsSpec(
                        roi=request.array_specs.get(
                            self.labels[0], request[self.labels[1]]
                        ).roi
                    )
                )
            else:
                raise ValueError(
                    "One of the following must be requested: {}, {}, {}".format(
                        self.points, self.arrays, self.labels
                    )
                )

            dps.place_holders[self.nonempty_placeholder].roi = dps.place_holders[
                self.nonempty_placeholder
            ].roi.grow(growth, growth)

        # handle smaller requests
        voxel_size = request.get_lcm_voxel_size()
        direction += Coordinate(np.array(direction) % np.array(voxel_size))

        if any([points in request for points in self.points]):
            dps[self.point_source] = copy.deepcopy(request[self.points[0]])
            dps[self.point_source].roi = (
                dps[self.point_source]
                .roi.shift(direction)
                .snap_to_grid(voxel_size, mode="closest")
            )
        if any([array in request for array in self.arrays]):
            dps[self.array_source] = copy.deepcopy(request[self.arrays[0]])
            dps[self.array_source].roi = (
                dps[self.array_source]
                .roi.shift(direction)
                .snap_to_grid(voxel_size, mode="closest")
            )

        if any([labels in request for labels in self.labels]):
            dps[self.label_source] = copy.deepcopy(request[self.labels[0]])
            dps[self.label_source].roi = (
                dps[self.label_source]
                .roi.shift(direction)
                .snap_to_grid(voxel_size, mode="closest")
            )

        for source, targets in self.extra_keys.items():
            dps[source] = copy.deepcopy(request[targets[0]])
            dps[source].roi = (
                dps[source]
                .roi.shift(direction)
                .snap_to_grid(voxel_size, mode="closest")
            )

        return dps

    def provide(self, request: BatchRequest) -> Batch:
        """
        First request points with specific seeds, then request the rest if
        valid points are found.
        """

        logger.info(f"growing request by {self._get_growth()}")

        base_seed, add_seed, direction, prepare_profiling_stats = self.get_valid_seeds(
            request
        )

        timing_prepare = Timing(self, "prepare")
        timing_prepare.start()

        request_base = self.prepare(request, base_seed, -direction)
        logger.debug(f"base_request shrank to {request_base[self.point_source].roi}")
        request_add = self.prepare(request, add_seed, direction)
        logger.debug(f"add_request shrank to {request_add[self.point_source].roi}")

        timing_prepare.stop()

        base = self.upstream_provider.request_batch(request_base)
        add = self.upstream_provider.request_batch(request_add)

        timing_process = Timing(self, "process")
        timing_process.start()

        base = self.process(base, Coordinate([0, 0, 0]), request=request, batch_index=0)
        add = self.process(add, -Coordinate([0, 0, 0]), request=request, batch_index=1)

        if (
            len(add[self.point_source].graph.nodes) > 1
            and len(base[self.point_source].graph.nodes) > 1
        ):
            min_dist = min(
                [
                    np.linalg.norm(a["location"] - b["location"])
                    for a, b in itertools.product(
                        base[self.point_source].graph.nodes.values(),
                        add[self.point_source].graph.nodes.values(),
                    )
                ]
            )
            logger.info(f"Got a final min dist of {min_dist}")
        else:
            logger.warning("Got a final min dist of inf")

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
            **{k: v[0] for k, v in self.extra_keys.items()},
        }
        for key, value in base.items():
            combined[base_map.get(key, key)] = value

        add_map = {
            self.point_source: self.points[1],
            self.array_source: self.arrays[1],
            self.label_source: self.labels[1],
            **{k: v[1] for k, v in self.extra_keys.items()},
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

            while True:
                points_request_base, base_seed = self.prepare_points(request)
                base_batch = self.upstream_provider.request_batch(points_request_base)
                profiling_stats.merge_with(base_batch.profiling_stats)
                if len(base_batch[self.nonempty_placeholder].graph.nodes) > 0:
                    break
            logger.info("Got base batch")

            timing_prepare_base = Timing("prepare base")
            timing_prepare_base.start()

            # distance_transform, gradients = self.prepare_base(
            #     base_batch[self.nonempty_placeholder]
            # )

            for i in range(self.request_attempts):
                points_request_add, add_seed = self.prepare_points(request)
                output_roi = self._return_roi(request)
                add_batch = self.upstream_provider.request_batch(points_request_add)
                direction = self.seperate_using_kdtrees(
                    base_batch,
                    add_batch,
                    output_roi,
                    final=(i == self.request_attempts - 1),
                )
                if direction is not None:
                    return base_seed, add_seed, direction, profiling_stats
                else:
                    continue

    def seperate_using_kdtrees(
        self, base_batch: Batch, add_batch: Batch, output_roi: Roi, final=False
    ):
        points_add = add_batch.points.get(
            self.point_source, add_batch.points.get(self.nonempty_placeholder, None)
        )
        points_base = base_batch.points.get(
            self.point_source, base_batch.points.get(self.nonempty_placeholder, None)
        )

        if len(points_add.graph.nodes) < 1 or len(points_base.graph.nodes) < 1:
            return Coordinate([0, 0, 0])

        # shift add points to start at [0,0,0]
        add_locations = np.array(
            [
                point["location"] - points_add.spec.roi.get_begin()
                for point in points_add.graph.nodes.values()
            ]
        )
        add_tree = cKDTree(add_locations)
        # shift base points to start at [0,0,0]
        base_locations = np.array(
            [
                point["location"] - points_base.spec.roi.get_begin()
                for point in points_base.graph.nodes.values()
            ]
        )
        base_tree = cKDTree(base_locations)

        input_shape = points_base.spec.roi.get_shape()
        output_shape = output_roi.get_shape()
        input_radius = input_shape / 2
        output_radius = output_shape / 2

        current_shift = np.array([0, 0, 0], dtype=float)  # in voxels

        radius = max(output_radius)

        max_shift = input_radius - output_radius - Coordinate([1, 1, 1])

        for i in range(self.shift_attempts * 10):
            shift_attempt = Coordinate(current_shift)
            add_clipped_roi = Roi(
                input_radius - output_radius + shift_attempt, output_shape
            )
            base_clipped_roi = Roi(
                input_radius - output_radius - shift_attempt, output_shape
            )

            # query points in trees below certain distance from shifted center
            clipped_add_points = add_tree.query_ball_point(
                input_radius + shift_attempt, radius, p=float("inf")
            )
            clipped_base_points = base_tree.query_ball_point(
                input_radius - shift_attempt, radius, p=float("inf")
            )

            # if queried points are empty, skip
            if len(clipped_add_points) < 1 and len(clipped_base_points) < 1:
                logger.info(f"no points in centered roi!")
                continue

            # apply twice the shift to add points
            current_check = add_locations - shift_attempt * 2

            # get all points in base that are close to shifted add points
            points_too_close = base_tree.query_ball_point(
                current_check, np.mean(self.seperate_by)
            )

            # calculate next shift
            direction = np.zeros([3])
            min_dist = float("inf")
            count = 0
            for node_a, neighbors in enumerate(points_too_close):
                if node_a not in clipped_add_points or not add_clipped_roi.contains(
                    add_locations[node_a, :]
                ):
                    continue
                for neighbor in neighbors:
                    if (
                        neighbor not in clipped_base_points
                        or not base_clipped_roi.contains(base_locations[neighbor, :])
                    ):
                        continue

                    vector = (
                        add_locations[node_a, :] - add_clipped_roi.get_begin()
                    ) - (base_locations[neighbor, :] - base_clipped_roi.get_begin())
                    mag = np.linalg.norm(vector)
                    min_dist = min(min_dist, mag)
                    unit_vector = vector / (mag + 1)
                    # want to move at most n units if mag is 0, or 0 units if mag is n
                    direction += (np.max(self.seperate_by) - mag) * unit_vector
                    count += 1

            if count == 0 or self.seperate_by[0] <= min_dist <= self.seperate_by[1]:
                return shift_attempt

            logger.debug(
                f"min dist {min_dist} not in {self.seperate_by} "
                f"with shift: {current_shift}"
            )
            direction /= count
            if np.linalg.norm(direction) < 1e-2:
                logger.warning(f"Moving too slow. Probably stuck!")
                return None
            current_shift += direction + (np.random.random(3) - 0.5)
            np.clip(current_shift, -max_shift, max_shift, out=current_shift)

        logger.info(f"Request failed at {current_shift}. New Request!")
        if Path("test_output").exists():
            if not Path("test_output", "distances.obj").exists():
                pickle.dump(
                    base_batch, open(Path("test_output", "batch_base.obj"), "wb")
                )
            if not Path("test_output", "distances.obj").exists():
                pickle.dump(add_batch, open(Path("test_output", "batch_add.obj"), "wb"))
        if final:
            return current_shift
        else:
            return None

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

        # Shift and crop points and array
        return_roi = self._return_roi(request)
        for source_key, point_set in batch.points.items():
            point_set = self._shift_and_crop_points(point_set, direction, return_roi)
            batch.points[source_key] = point_set

        for source_key, array_set in batch.arrays.items():
            array_set = self._shift_and_crop_array(array_set, direction, return_roi)
            batch.arrays[source_key] = array_set

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
        distance = np.array((np.mean(self.seperate_by),) * len(voxel_size))
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
        if distances.min() > 0:
            logger.debug("No base points in this shift!")
            return False
        min_dist = float("inf")
        voxel_size = self.spec.get_lcm_voxel_size()
        for point_id, point_attrs in add_graph.nodes.items():
            if distances[Coordinate(point_attrs["location"] // voxel_size)] < min_dist:
                min_dist = distances[Coordinate(point_attrs["location"] // voxel_size)]
        if min_dist < self.seperate_by[0] or min_dist > self.seperate_by[1]:
            logger.info(f"min dist {min_dist} not in range {self.seperate_by}")
            return False
        else:
            logger.info(f"Saw min dist of {min_dist}!")
            return True
