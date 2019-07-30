import numpy as np
import copy
from gunpowder import (
    RandomLocation,
    BatchFilter,
    BatchRequest,
    Roi,
    Points,
    Array,
    ArrayKey,
    ArraySpec,
    Coordinate,
    PointsSpec,
    PointsKey,
)
from neurolight.gunpowder.swc_file_source import SwcPoint
from gunpowder.profiling import Timing
from typing import Tuple
import logging

from .swc_nx_graph import (
    points_to_graph,
    crop_graph,
    shift_graph,
    graph_to_swc_points,
    relabel_connected_components,
    interpolate_points,
)

logger = logging.getLogger(__name__)


class SeperatePoints(RandomLocation):
    """Given two input point keys, obtain rois such that the points can
    be overlaid directly on top of each other, and be seperated by some
    minimum distance. 

        Args:

            point_keys (:class:``Tuple[PointsKey]``):
                
                A pair of PointsKeys from which two sets of points will be taken.
                If a single PointsKey is provided, both sets of points will be
                taken from the same dataset

            seperate_by (:class:``float``, optional):
                Distance in world units between the two center nodes.
                It is guaranteed that no other points in two point sets will be closer than
                this distance.
                This distance will be approximated since the volumetric data will be shifted
                on its grid.

        """

    def __init__(
        self,
        *args,
        ensure_seperated: Tuple[PointsKey, PointsKey],
        seperate_by: float = 10.0,
        shift_attempts: int = 300,
        request_attempts: int = 30,
        **kwargs
    ):
        self.super().__init__(args, kwargs)
        self.ensure_seperated = ensure_seperated
        self.seperate_by = seperate_by

        self.shift_attempts = shift_attempts
        self.request_attempts = request_attempts

    def setup(self):
        """
        TODO: mutates the rois of points, should this be indicated here?
        """
        pass

    def prepare(self, request):
        """
        grow the rois such that assuming there exists a centered point,
        the two rois can be cropped such that the center points are a minimum
        distance from each other
        """
        assert (
            self.points[0] in request and self.points[1] in request
        ), "Both {} and {} must be requested!".format(self.points[0], self.points[1])
        assert all(
            request[self.points[0]].roi.get_shape()
            == request[self.points[1]].roi.get_shape()
        ), (
            "{} requested roi of shape {} but {} requested roi "
            + "of shape {}. These should be the same."
        ).format(
            self.points[0],
            request[self.points[0]].roi.get_shape(),
            self.points[1],
            request[self.points[1]].roi.get_shape,
        )

        growth = self._get_growth()
        request[self.points[0]].roi = request[self.points[0]].roi.grow(growth, growth)
        request[self.points[1]].roi = request[self.points[1]].roi.grow(growth, growth)

    def provide(self, request):
        """
        Instead of making multiple requests in the process phase,
        this provide method should be overwritten to get two copies of
        the required points and arrays. 
        """

        upstream_request = copy.deepcopy(request)

        # skip = super().__can_skip(request)
        skip = False
        valid_pair = False

        timing_prepare = Timing(self, "prepare")
        timing_prepare.start()

        if not skip:
            upstream_request = self.prepare(upstream_request)
            self.remove_provided(upstream_request)

        request_attempts = 0
        while not valid_pair:

            batch = self.get_upstream_provider().request_batch(upstream_request)
            self.process(batch)
            request_attempts += 1

            if not skip:
                for _ in range(self.shift_attempts):
                    if self._valid_pair(batch):
                        valid_pair = True
                        break

            if request_attempts >= self.request_attempts:
                raise Exception(
                    "COULD NOT OBTAIN A PAIR OF NEURONS SEPERATED BY {}!".format(
                        self.seperate_by
                    )
                )

        timing_prepare.stop()

        timing_process = Timing(self, "process")
        timing_process.start()
        timing_process.stop()

        batch.profiling_stats.add(timing_process)

        return batch

    def process(self, batch, request):
        points_a = batch.points[self.points[0]]
        points_b = batch.arrays[self.points[1]]

        points_a, points_b = self.__seperate(points_a, points_b, request)

        batch.points[self.points[0]] = points_a
        batch.points[self.points[1]] = points_b

        return batch

    def __seperate(self, points_a, points_b, request):

        random_direction = self.__get_random_shift_direction()

        # Shift and crop points and array
        points_a = self._shift_and_crop(
            points_a, direction=random_direction, output_roi=request[points_a].roi
        )
        points_b = self._shift_and_crop(
            points_b, direction=-random_direction, output_roi=request[points_b].roi
        )
        return points_a, points_b

    def __get_random_shift_direction(self):
        # there should be a better way of doing this. If distances are small,
        # the rounding may make a big difference. Rounding (1.3, 1.3, 1.3) would
        # give a euclidean distance of ~1.7 instead of ~2.25
        # A better solution might be to do a distance transform on an array with a centered
        # dot and then find all potential moves that have a distance equal to delta +- 0.5
        voxel_size = np.array(self.spec[self.array_source].voxel_size)
        random_direction = np.random.randn(3)
        random_direction /= np.linalg.norm(random_direction)  # unit vector
        random_direction *= self.seperate_by  # physical units
        random_direction = (
            (np.round(random_direction / voxel_size) + 1) // 2
        ) * voxel_size  # physical units rounded to nearest voxel size
        random_direction = Coordinate(random_direction)
        return random_direction

    def _shift_and_crop(
        self,
        points: Points,
        array: Array,
        labels: Array,
        direction: Coordinate,
        output_roi: Roi,
    ):
        # Shift and crop the points
        center = points.spec.roi.get_offset() + points.spec.roi.get_shape() // 2

        new_center = center + direction
        new_offset = new_center - output_roi.get_shape() // 2
        new_roi = Roi(new_offset, output_roi.get_shape())

        g = points_to_graph(points.data)
        g = crop_graph(g, new_roi)
        g = shift_graph(g, np.array(output_roi.get_begin() - new_offset, dtype=float))
        g, _ = relabel_connected_components(g)

        new_points_data = graph_to_swc_points(g)

        points = Points(new_points_data, points.spec.copy())
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

    def _valid_pair(self, batch):
        """
        Simply checks for every pair of points, is the distance between them
        greater than the desired seperation criterion.
        """
        voxel_size = np.array(self.spec[self.array_source].voxel_size)
        points_base = batch[self.points[0]].data
        points_add = batch[self.points[1]].data

        # add interpolated points:
        base_graph = points_to_graph(points_base)
        base_graph = interpolate_points(base_graph)
        add_graph = points_to_graph(points_add)
        add_graph = interpolate_points(add_graph)

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
                    self.seperate_by - voxel_size.max() * 2,
                    self.seperate_by + voxel_size.max() * 2,
                    min_dist,
                )
            )
            return False

    def __shift_request(self, request, shift):

        # shift request ROIs
        for specs_type in [request.array_specs, request.points_specs]:
            for (key, spec) in specs_type.items():
                roi = spec.roi.shift(shift)
                specs_type[key].roi = roi
