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


class Recenter(BatchFilter):
    """Ensures there is point in the center of the roi.

        Args:

            point_source (:class:``PointsKey``):
                The Points from which to ensure a node is centered

            array_source (:class:``Tuple[ArraySpec]``):
                The Array to crops

            max_offset (:class:``float``, optional):
                Maximum distance the center point could have moved due to upstream processing

        """

    def __init__(
        self, point_source: PointsKey, array_source: ArrayKey, max_offset: float = 1.0
    ):
        self.point_source = point_source
        self.array_source = array_source
        self.max_offset = max_offset

    def prepare(self, request):
        """
        The request from upstream not ask for the points or arrays provided
        by this node. Instead it should just add a request to the point_source
        and image_source which will then be retrieved twice in provide.
        """
        growth = self._get_growth()
        points_roi = request[self.point_source].roi
        points_roi = points_roi.grow(growth, growth)
        arrays_roi = request[self.array_source].roi
        arrays_roi = arrays_roi.grow(growth, growth)

        request[self.point_source].roi = points_roi
        request[self.array_source].roi = arrays_roi

    def process(self, batch, request):
        points = batch.points[self.point_source]
        array = batch.arrays[self.array_source]

        center = array.spec.roi.get_begin() + array.spec.roi.get_shape() / 2
        closest = None
        for point in points.data.values():
            closest = (
                closest
                if closest is not None
                and np.linalg.norm(center - closest)
                < np.linalg.norm(center - Coordinate(point.location))
                else Coordinate(point.location)
            )
        print(center)
        direction = closest - center
        print(direction)

        # Shift and crop points and array
        points, array = self._shift_and_crop(
            points,
            array,
            direction=direction,
            output_roi=request[self.point_source].roi,
        )

        batch.points[self.point_source] = points
        batch.arrays[self.array_source] = array

        return batch

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
        distance = np.array((self.max_offset,) * 3)
        # voxel shift is rounded up to the nearest voxel in each axis
        voxel_shift = (distance + voxel_size - 1) // voxel_size
        # expand positive and negative sides enough to contain any desired shift
        half_shift = (voxel_shift + 1) // 2
        return Coordinate(half_shift * 2)

