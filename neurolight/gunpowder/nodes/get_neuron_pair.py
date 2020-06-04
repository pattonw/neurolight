import numpy as np
import networkx as nx
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.ndimage.filters import gaussian_filter
from gunpowder import (
    ProviderSpec,
    BatchProvider,
    BatchRequest,
    Batch,
    Roi,
    GraphSpec,
    Graph,
    Array,
    ArrayKey,
    Coordinate,
    GraphKey,
)
from gunpowder.profiling import Timing, ProfilingStats

from typing import Tuple, Optional, Dict, Union
import logging
import time
import copy
import itertools
import pickle
from pathlib import Path
import random

DataKey = Union[GraphKey, ArrayKey]

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
        point_source: GraphKey,
        array_source: ArrayKey,
        label_source: ArrayKey,
        points: Tuple[GraphKey, GraphKey],
        arrays: Tuple[ArrayKey, ArrayKey],
        labels: Tuple[ArrayKey, ArrayKey],
        output_shape: Coordinate,
        nonempty_placeholder: Optional[GraphKey] = None,
        seperate_by: Tuple[float, float] = (0.0, 1.0),
        shift_attempts: int = 50,
        request_attempts: int = 3,
        extra_keys: Optional[Dict[DataKey, Tuple[DataKey, DataKey]]] = {},
    ):
        self.point_source = point_source
        self.nonempty_placeholder = nonempty_placeholder
        self.array_source = array_source
        self.label_source = label_source
        self.output_shape = output_shape
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
            dps[point_key] = GraphSpec(
                roi=request.array_specs.get(self.arrays[0], request[self.arrays[1]]).roi
            )
        elif any([labels in request for labels in self.labels]):
            dps[point_key] = GraphSpec(
                roi=request.array_specs.get(self.labels[0], request[self.labels[1]]).roi
            )
        else:
            raise ValueError(
                "One of the following must be requested: {}, {}, {}".format(
                    self.points, self.arrays, self.labels
                )
            )

        dps[point_key].roi = dps[point_key].roi.grow(growth, growth)

        dps[self.array_source] = copy.deepcopy(request[self.arrays[0]])
        dps[self.array_source].roi = dps[self.array_source].roi.grow(growth, growth)
        dps[self.array_source].placeholder = True
        dps[self.label_source] = copy.deepcopy(request[self.labels[0]])
        dps[self.label_source].roi = dps[self.label_source].roi.grow(growth, growth)
        dps[self.label_source].placeholder = True

        return dps, seed

    def prepare(
        self, request: BatchRequest, seed: int, direction: Coordinate
    ) -> Tuple[BatchRequest, int]:
        """
        Only request everything with the given seed
        """
        dps = BatchRequest(random_seed=seed)

        if self.nonempty_placeholder is not None:
            # request nonempty placeholder of size request total roi
            # grow such that it can be cropped down to two different locations
            growth = self._get_growth()

            total_roi = request.get_total_roi()
            grown_roi = total_roi.grow(growth, growth)
            dps[self.nonempty_placeholder] = GraphSpec(roi=grown_roi, placeholder=True)

        # handle smaller requests
        array_keys = list(request.array_specs.keys())
        voxel_size = self.spec.get_lcm_voxel_size(array_keys)
        direction = Coordinate(direction)
        direction -= Coordinate(tuple(np.array(direction) % np.array(voxel_size)))

        if any([points in request for points in self.points]):
            dps[self.point_source] = copy.deepcopy(request[self.points[0]])
            dps[self.point_source].roi = dps[self.point_source].roi.shift(direction)
        if any([array in request for array in self.arrays]):
            dps[self.array_source] = copy.deepcopy(request[self.arrays[0]])
            dps[self.array_source].roi = dps[self.array_source].roi.shift(direction)
        if any([labels in request for labels in self.labels]):
            dps[self.label_source] = copy.deepcopy(request[self.labels[0]])
            dps[self.label_source].roi = dps[self.label_source].roi.shift(direction)

        for source, targets in self.extra_keys.items():
            if targets[0] in request:
                dps[source] = copy.deepcopy(request[targets[0]])
                dps[source].roi = dps[source].roi.shift(direction)

        return dps

    def provide(self, request: BatchRequest) -> Batch:
        """
        First request points with specific seeds, then request the rest if
        valid points are found.
        """

        logger.debug(f"growing request by {self._get_growth()}")

        has_component = False
        while not has_component:

            base_seed, add_seed, direction, prepare_profiling_stats = self.get_valid_seeds(
                request
            )

            timing_prepare = Timing(self, "prepare")
            timing_prepare.start()

            request_base = self.prepare(request, base_seed, -direction)
            if add_seed is not None:
                request_add = self.prepare(request, add_seed, direction)
            else:
                request_add = None
                logger.debug(f"No add_request needed!")

            timing_prepare.stop()

            base = self.upstream_provider.request_batch(request_base)
            if request_add is not None:
                add = self.upstream_provider.request_batch(request_add)
            else:
                add = self._empty_copy(base)

            has_component = True

            timing_process = Timing(self, "process")
            timing_process.start()

            base = self.process(base, Coordinate([0, 0, 0]), request=request)
            add = self.process(add, -Coordinate([0, 0, 0]), request=request)

            batch = self.merge_batches(base, add)

            timing_process.stop()
            batch.profiling_stats.merge_with(prepare_profiling_stats)
            batch.profiling_stats.add(timing_prepare)
            batch.profiling_stats.add(timing_process)

        return batch

    def _empty_copy(self, base: Batch):
        add = Batch()
        for key, array in base.arrays.items():
            add[key] = Array(np.zeros_like(array.data), spec=copy.deepcopy(array.spec))
        for key, points in base.points.items():
            add[key] = Graph([], [], spec=copy.deepcopy(points.spec))
        return add

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
            if k > self.request_attempts:
                raise ValueError("Failed to retrieve a pair of neurons!")

            while True:
                points_request_base, base_seed = self.prepare_points(request)
                base_batch = self.upstream_provider.request_batch(points_request_base)
                profiling_stats.merge_with(base_batch.profiling_stats)
                if len(list(base_batch[self.nonempty_placeholder].nodes)) > 0:
                    # Can happen if there are processing steps between random location and here.
                    break
            logger.debug("Got base batch")

            wccs = list(
                base_batch[self.nonempty_placeholder]
                .crop(
                    roi=self._centered_output_roi(
                        base_batch[self.nonempty_placeholder].spec.roi
                    ),
                )
                .trim(
                    roi=self._centered_output_roi(
                        base_batch[self.nonempty_placeholder].spec.roi
                    )
                )
                .connected_components
            )
            if len(wccs) > 2:
                logger.debug(
                    f"Skipping add batch since we see {len(list(wccs))} connected components"
                )
                return base_seed, None, Coordinate([0, 0, 0]), profiling_stats

            for i in range(self.request_attempts):
                points_request_add, add_seed = self.prepare_points(request)
                output_roi = self._return_roi(request)
                add_batch = self.upstream_provider.request_batch(points_request_add)
                direction = self.seperate_using_kdtrees(
                    base_batch,
                    add_batch,
                    output_roi,
                    final=(i == self.request_attempts - 1),
                    goal=random.random() * (self.seperate_by[1] - self.seperate_by[0])
                    + self.seperate_by[0],
                )
                if direction is not None:
                    logger.debug("Got add batch")
                    return base_seed, add_seed, direction, profiling_stats
                else:
                    continue

    def seperate_using_kdtrees(
        self,
        base_batch: Batch,
        add_batch: Batch,
        output_roi: Roi,
        final=False,
        goal: float = 0,
        epsilon: float = 0.1,
    ):
        points_add = add_batch.graphs.get(
            self.point_source, add_batch.graphs.get(self.nonempty_placeholder, None)
        )
        points_base = base_batch.graphs.get(
            self.point_source, base_batch.graphs.get(self.nonempty_placeholder, None)
        )

        if len(list(points_add.nodes)) < 1 or len(list(points_base.nodes)) < 1:
            return Coordinate([0, 0, 0])

        # shift add points to start at [0,0,0]
        add_locations = np.array(
            [
                point.location - points_add.spec.roi.get_begin()
                for point in points_add.nodes
            ]
        )
        add_tree = cKDTree(add_locations)
        # shift base points to start at [0,0,0]
        base_locations = np.array(
            [
                point.location - points_base.spec.roi.get_begin()
                for point in points_base.nodes
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
                logger.debug(f"no points in centered roi!")
                continue

            # apply twice the shift to add points
            current_check = add_locations - shift_attempt * 2

            # get all points in base that are close to shifted add points
            points_too_close = base_tree.query_ball_point(current_check, goal)

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
                        base_locations[neighbor, :] - base_clipped_roi.get_begin()
                    ) - (add_locations[node_a, :] - add_clipped_roi.get_begin())
                    mag = np.linalg.norm(vector)
                    min_dist = min(min_dist, mag)
                    unit_vector = vector / (mag + 1)
                    # want to move at most n units if mag is 0, or 0 units if mag is n
                    direction += (goal - mag) * unit_vector
                    count += 1

            if (
                count == 0
                or goal - goal * epsilon - epsilon
                <= min_dist
                <= goal + goal * epsilon + epsilon
            ):
                logger.debug(
                    f"shift: {shift_attempt} worked with {min_dist} and {count}"
                )
                return shift_attempt

            logger.debug(
                f"min dist {min_dist} not in {goal - goal*epsilon, goal + goal*epsilon} "
                f"with shift: {current_shift}"
            )
            direction /= count
            if np.linalg.norm(direction) < 1e-2:
                logger.debug(f"Moving too slow. Probably stuck!")
                return None
            current_shift += direction + (np.random.random(3) - 0.5)
            np.clip(current_shift, -max_shift, max_shift, out=current_shift)

        logger.debug(f"Request failed at {current_shift}. New Request!")
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
        self, batch: Batch, direction: Coordinate, request: BatchRequest, inplace=True
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

    def _return_roi(self, request):
        """
        Get the required output roi. Fails if multiple different roi are requested
        TODO: this assumption doesn't seem necessary
        """
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

    def _centered_output_roi(self, input_roi):
        start = input_roi.get_center() - self.output_shape / 2
        return Roi(start, self.output_shape)

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
        # Extract portion of points.spec.roi to be kept. i.e. center of size output_roi,
        # shifted by direction
        shifted_smaller_roi = self._extract_roi(points.spec.roi, output_roi, direction)

        # Crop out points to keep
        cropped = points.crop(shifted_smaller_roi).trim(shifted_smaller_roi)

        # Shift them into position relative to output_roi
        direction = -shifted_smaller_roi.get_offset() + output_roi.get_offset()
        cropped.shift(direction)
        cropped.spec.roi = output_roi

        return cropped

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
