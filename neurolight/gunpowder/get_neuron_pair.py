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
        label_source: ArrayKey,
        points: Tuple[PointsKey, PointsKey],
        arrays: Tuple[ArrayKey, ArrayKey],
        labels: Tuple[ArrayKey, ArrayKey],
        seperate_by: float = 1.0,
        shift_attempts: int = 3,
        request_attempts: int = 3,
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

    def prepare(self, request):
        """
        The request from upstream not ask for the points or arrays provided
        by this node. Instead it should just add a request to the point_source
        and image_source which will then be retrieved twice in provide.


        TODO:
        How to handle strange cases:
        Requesting one or two out of the 3 pieces (raw, labels, points)
        Requesting varying sized rois
        Requestiong multiple copies (points in output_size and input_size)
        """
        growth = self._get_growth()

        upstream_request = BatchRequest()

        # prepare requests
        # if point_key is not requested request it
        points_roi = request[self.points[0]].roi
        points_roi = points_roi.grow(growth, growth)
        if self.point_source not in request:
            upstream_request.add(self.point_source, points_roi.get_shape())
        else:
            upstream_request[self.point_source] = request[self.point_source]
            upstream_request[self.point_source].roi = points_roi

        arrays_roi = request[self.arrays[0]].roi
        arrays_roi = arrays_roi.grow(growth, growth)
        if self.array_source not in request:
            upstream_request.add(
                self.array_source,
                arrays_roi.get_shape(),
                voxel_size=self.spec[self.array_source].voxel_size,
            )
        else:
            upstream_request[self.array_source] = request[self.array_source]
            upstream_request[self.array_source].roi = arrays_roi

        labels_roi = request[self.labels[0]].roi
        labels_roi = labels_roi.grow(growth, growth)
        if self.label_source not in request:
            upstream_request.add(
                self.label_source,
                labels_roi.get_shape(),
                voxel_size=self.spec[self.label_source].voxel_size,
            )
        else:
            upstream_request[self.label_source] = request[self.label_source]
            upstream_request[self.label_source].roi = labels_roi

        return upstream_request

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
        label_base = batch_base.arrays[self.label_source]
        points_add = batch_add.points[self.point_source]
        array_add = batch_add.arrays[self.array_source]
        label_add = batch_add.arrays[self.label_source]

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

        # Shift and crop points and array
        points_base, array_base, label_base = self._shift_and_crop(
            points_base,
            array_base,
            label_base,
            direction=random_direction,
            request=request,
            k=0,
        )
        points_add, array_add, label_add = self._shift_and_crop(
            points_add,
            array_add,
            label_add,
            direction=-random_direction,
            request=request,
            k=1,
        )

        batch_base.points[self.points[0]] = points_base
        batch_base.arrays[self.arrays[0]] = array_base
        batch_base.arrays[self.labels[0]] = label_base
        batch_base.points[self.points[1]] = points_add
        batch_base.arrays[self.arrays[1]] = array_add
        batch_base.arrays[self.labels[1]] = label_add

        return batch_base

    def _shift_and_crop(
        self,
        points: Points,
        array: Array,
        labels: Array,
        direction: Coordinate,
        request: BatchRequest,
        k: int,
    ):

        # Shift and crop the array
        output_roi = request[self.arrays[k]].roi
        center = array.spec.roi.get_offset() + array.spec.roi.get_shape() // 2
        new_center = center + direction
        new_offset = new_center - output_roi.get_shape() // 2
        new_roi = Roi(new_offset, output_roi.get_shape())

        array = array.crop(new_roi)
        array.spec.roi = output_roi

        # Shift and crop the labels
        output_roi = request[self.labels[k]].roi
        center = labels.spec.roi.get_offset() + labels.spec.roi.get_shape() // 2
        new_center = center + direction
        new_offset = new_center - output_roi.get_shape() // 2
        new_roi = Roi(new_offset, output_roi.get_shape())

        labels = labels.crop(new_roi)
        labels.spec.roi = output_roi

        # Shift and crop the points
        output_roi = request[self.points[k]].roi
        center = points.spec.roi.get_offset() + points.spec.roi.get_shape() // 2
        new_center = center + direction
        new_offset = new_center - output_roi.get_shape() // 2
        new_roi = Roi(new_offset, output_roi.get_shape())

        print("original roi: {}".format(points.spec.roi))
        print("new roi: {}".format(new_roi))
        print("output roi: {}".format(output_roi))

        g = points_to_graph(points.data)
        g = crop_graph(g, new_roi)
        g = shift_graph(g, -np.array(new_offset, dtype=float))
        g, _ = relabel_connected_components(g)

        new_points_data = graph_to_swc_points(g)

        points = Points(new_points_data, points.spec.copy())
        points.spec.roi = output_roi

        return points, array, labels

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

