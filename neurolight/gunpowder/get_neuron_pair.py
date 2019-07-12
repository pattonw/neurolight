import numpy as np
import copy
from gunpowder import (
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
from typing import List, Optional, Tuple, Union
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class GetNeuronPair(BatchFilter):
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
        points: Tuple[PointsKey, PointsKey],
        arrays: Tuple[ArrayKey, ArrayKey],
        seperate_by: float = 1.0,
        shift_attempts: int = 3,
        request_attempts: int = 3,
    ):
        self.point_source = point_source
        self.array_source = array_source
        self.points = points
        self.arrays = arrays
        self.seperate_by = seperate_by

        self.shift_attempts = shift_attempts
        self.request_attempts = request_attempts

    def setup(self):
        """
        Provide the two copies of the upstream points and images
        """
        for point_key, array_key in zip(self.points, self.arrays):
            self.provides(point_key, self.spec[self.point_source])
            self.provides(array_key, self.spec[self.array_source])

    def prepare(self, request):
        """
        The request from upstream not ask for the points or arrays provided
        by this node. Instead it should just add a request to the point_source
        and image_source which will then be retrieved twice in provide.
        """
        growth = self._get_growth()
        points_roi = request[self.points[0]].roi
        points_roi = points_roi.grow(growth, growth)
        arrays_roi = request[self.arrays[0]].roi
        arrays_roi = arrays_roi.grow(growth, growth)

        # prepare requests
        # if point_key is not requested request it
        if self.point_source not in request:
            request.add(self.point_source, points_roi.get_shape())
        else:
            request[self.point_source].roi = points_roi

        if self.array_source not in request:
            request.add(
                self.array_source,
                arrays_roi.get_shape(),
                voxel_size=self.spec[self.array_source].voxel_size,
            )
        else:
            request[self.array_source].roi = arrays_roi

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
            self.prepare(upstream_request)
            self.remove_provided(upstream_request)

        request_attempts = 0
        while not valid_pair:

            timing_prepare.stop()

            batch_base = self.get_upstream_provider().request_batch(upstream_request)
            batch_add = self.get_upstream_provider().request_batch(upstream_request)
            request_attempts += 1

            timing_process = Timing(self, "process")
            timing_process.start()

            if not skip:
                shift_attempts = 0
                while not valid_pair:
                    shift_attempts += 1
                    batch_base = self.process(batch_base, batch_add, request)
                    if self._valid_pair(batch_base):
                        valid_pair = True
                    if shift_attempts >= self.shift_attempts:
                        break

            if request_attempts >= self.request_attempts:
                raise Exception(
                    "COULD NOT OBTAIN A PAIR OF NEURONS SEPERATED BY {}!".format(
                        self.seperate_by
                    )
                )

        timing_process.stop()

        batch_base.profiling_stats.add(timing_prepare)
        batch_base.profiling_stats.add(timing_process)

        return batch_base

    def process(self, batch_base, batch_add, request):
        points_base = batch_base.points[self.point_source]
        array_base = batch_base.arrays[self.array_source]
        points_add = batch_add.points[self.point_source]
        array_add = batch_add.arrays[self.array_source]

        # there should be a better way of doing this. If distances are small,
        # the rounding may make a big difference. Rounding (1.3, 1.3, 1.3) would
        # give a euclidean distance of ~1.7 instead of ~2.25
        # A better solution might be to do a distance transform on an array with a centered
        # dot and then find all potential moves that have a distance equal to delta +- 0.5
        random_direction = np.random.randn(3)
        random_direction /= np.linalg.norm(random_direction)
        random_direction *= (self.seperate_by + 1) // 2
        random_direction = Coordinate(np.round(random_direction))

        # Shift and crop points and array
        points_base, array_base = self._shift_and_crop(
            points_base,
            array_base,
            direction=random_direction,
            output_roi=request[self.points[0]].roi,
        )
        points_add, array_add = self._shift_and_crop(
            points_add,
            array_add,
            direction=-random_direction,
            output_roi=request[self.points[0]].roi,
        )

        batch_base.points[self.points[0]] = points_base
        batch_base.arrays[self.arrays[0]] = array_base
        batch_base.points[self.points[1]] = points_add
        batch_base.arrays[self.arrays[1]] = array_add

        return batch_base

    def _shift_and_crop(
        self, points: Points, array: Array, direction: Coordinate, output_roi: Roi
    ):
        # Shift and crop the array
        center = array.spec.roi.get_offset() + array.spec.roi.get_shape() // 2
        new_center = center + direction
        new_offset = new_center - output_roi.get_shape() // 2
        new_roi = Roi(new_offset, output_roi.get_shape())
        array = array.crop(new_roi)
        array.spec.roi = output_roi

        new_points_data = {}
        new_points_spec = points.spec
        new_points_spec.roi = new_roi
        new_points_graph = nx.DiGraph()

        # shift points and add them to a graph
        for point_id, point in points.data.items():
            if new_roi.contains(point.location):
                new_point = point.copy()
                new_point.location = (
                    point.location - new_offset + output_roi.get_begin()
                )
                new_points_graph.add_node(
                    new_point.point_id,
                    point_id=new_point.point_id,
                    parent_id=new_point.parent_id,
                    location=new_point.location,
                    label_id=new_point.label_id,
                    radius=new_point.radius,
                    point_type=new_point.point_type,
                )
                if points.data.get(new_point.parent_id, False) and new_roi.contains(
                    points.data[new_point.parent_id].location
                ):
                    new_points_graph.add_edge(new_point.parent_id, new_point.point_id)

        # relabel connected components
        for i, connected_component in enumerate(
            nx.weakly_connected_components(new_points_graph)
        ):
            for node in connected_component:
                new_points_graph.nodes[node]["label_id"] = i

        # store new graph data in points
        new_points_data = {
            point_id: SwcPoint(
                point_id=point["point_id"],
                point_type=point["point_type"],
                location=point["location"],
                radius=point["radius"],
                parent_id=point["parent_id"],
                label_id=point["label_id"],
            )
            for point_id, point in new_points_graph.nodes.items()
        }
        points = Points(new_points_data, new_points_spec)
        points.spec.roi = output_roi
        return points, array

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
        return Coordinate(half_shift * 2)

    def _valid_pair(self, batch):
        """
        Simply checks for every pair of points, is the distance between them
        greater than the desired seperation criterion.
        """
        points_base = batch[self.points[0]]
        points_add = batch[self.points[1]]
        min_dist = self.seperate_by + 2
        for point_id_base, point_base in points_base.data.items():
            for point_id_add, point_add in points_add.data.items():
                min_dist = min(
                    np.linalg.norm(point_base.location - point_add.location), min_dist
                )
        if self.seperate_by - 1 <= min_dist <= self.seperate_by + 1:
            print(("Got a min distance of {}").format(min_dist))
            return True
        else:
            print(
                (
                    "expected a minimum distance between the two neurons"
                    + "to be in the range ({}, {}), however saw a min distance of {}"
                ).format(self.seperate_by - 1, self.seperate_by + 1, min_dist)
            )
            return False

