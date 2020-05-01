import numpy as np
import networkx as nx
from gunpowder import (
    BatchFilter,
    BatchRequest,
    Roi,
    Array,
    ArraySpec,
    Coordinate,
    GraphSpec,
    Graph,
)
import logging

logger = logging.getLogger(__name__)


class RasterizeSkeleton(BatchFilter):
    """Draw skeleton into a binary array given a swc.

        Args:

            array (:class:``ArrayKey``):
                The key of the binary array to create.

            array_spec (:class:``ArraySpec``, optional):

                The spec of the array to create. Use this to set the datatype and
                voxel size.

            points (:class:``PointsKey``):
                The key of the points to form the skeleton.
        """

    def __init__(
        self, points, array, array_spec, connected_component_labeling: bool = True
    ):

        self.points = points
        self.array = array
        self.array_spec = array_spec
        self.connected_component_labeling = connected_component_labeling

    def setup(self):
        self.enable_autoskip()

        # todo: ensure that both rois are the same
        points_roi = self.spec[self.points].roi
        if self.array_spec is None:
            self.array_spec = ArraySpec(
                roi=points_roi.copy(),
                voxel_size=Coordinate((1,) * points_roi.dims()),
                interpolatable=False,
                dtype=np.uint64,
            )

        if self.array_spec.roi is None:
            self.array_spec.roi = points_roi.copy()

        self.provides(self.array, self.array_spec)

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.points] = GraphSpec(roi=request[self.array].roi)
        return deps

    def process(self, batch, request):

        points = batch.points[self.points]
        # assert len(points.data.items()) > 0, "No Swc Points in enlarged Roi."
        assert isinstance(points, Graph), "Rasterize skeleton needs a Graph."

        voxel_size = self.array_spec.voxel_size

        # get roi used for creating the new array (points_roi does not
        # necessarily align with voxel size)
        enlarged_array_roi = points.spec.roi.snap_to_grid(voxel_size)
        offset = enlarged_array_roi.get_begin() / voxel_size
        shape = enlarged_array_roi.get_shape() / voxel_size
        array_roi = Roi(offset, shape)
        array_data = np.zeros(shape, dtype=self.array_spec.dtype)

        graph = points

        if self.connected_component_labeling:
            # graph.relabel_connected_components()
            binarized = {}
            for e in graph.edges:
                cc = graph.node(e.u).attrs["component"]
                bin_component = binarized.setdefault(
                    cc, np.zeros_like(array_data, dtype=bool)
                )
                p1 = (graph.node(e.u).location / voxel_size - offset).astype(int)
                p2 = (graph.node(e.v).location / voxel_size - offset).astype(int)
                self._rasterize_line_segment(p1, p2, bin_component)
            for i, bined in binarized.items():
                overlap = np.logical_and(
                    np.logical_and(array_data > 0, array_data != i + 1), bined
                )
                array_data[bined] = i + 1
                array_data[overlap] = -1
        else:
            binarized = np.zeros_like(array_data, dtype=np.bool)
            for e in graph.edges:
                p1 = (graph.node(e.u).location / voxel_size - offset).astype(int)
                p2 = (graph.node(e.v).location / voxel_size - offset).astype(int)
                binarized = self._rasterize_line_segment(p1, p2, binarized)

            overlap = np.logical_and(
                np.logical_and(array_data > 0, array_data != 1), binarized
            )
            array_data[binarized] = 1

        logger.debug(f"Input graph had {len(list(graph.nodes))} nodes!")
        logger.debug(f"Output array contains {sum(array_data != 0)} non_empty pixels")

        array = Array(
            data=array_data,
            spec=ArraySpec(
                roi=array_roi * voxel_size,
                voxel_size=voxel_size,
                interpolatable=False,
                dtype=self.array_spec.dtype,
            ),
        )

        array = array.crop(request[self.array].roi)
        batch.arrays[self.array] = array

        return batch

    def _bresenhamline_nslope(self, slope):

        scale = np.amax(np.abs(slope))
        normalizedslope = slope / float(scale) if scale != 0 else slope

        return normalizedslope

    def _bresenhamline(self, start_voxel, end_voxel, max_iter=5):

        if max_iter == -1:
            max_iter = np.amax(np.abs(end_voxel - start_voxel))
        dim = start_voxel.shape[0]
        nslope = self._bresenhamline_nslope(end_voxel - start_voxel)

        # steps to iterate on
        stepseq = np.arange(1, max_iter + 1)
        stepmat = np.tile(stepseq, (dim, 1)).T

        # some hacks for broadcasting properly
        bline = start_voxel[np.newaxis, :] + nslope[np.newaxis, :] * stepmat

        # Approximate to nearest int
        return np.array(np.rint(bline), dtype=start_voxel.dtype)

    def _rasterize_line_segment(self, point, parent, skeletonized):
        point = np.clip(
            np.floor(point), np.zeros_like(point), np.array(skeletonized.shape) - 1
        )
        parent = np.clip(
            np.floor(parent), np.zeros_like(parent), np.array(skeletonized.shape) - 1
        )

        # use Bresenham's line algorithm based on:
        # http://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
        line_segment_points = self._bresenhamline(point, parent, max_iter=-1)

        if line_segment_points.shape[0] > 0:
            idx = np.transpose(line_segment_points.astype(int))
            if any(
                [np.max(idx) >= shape for idx, shape in zip(idx, skeletonized.shape)]
            ):
                logger.warning(
                    "Got max index: {}, but shape is only: {}".format(
                        np.max(idx, axis=0), skeletonized.shape
                    )
                )
            skeletonized[tuple(idx)] = True
        skeletonized[tuple(point.astype(int))] = True

        return skeletonized
