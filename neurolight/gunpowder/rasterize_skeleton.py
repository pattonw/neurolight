import numpy as np
from gunpowder import BatchFilter, Roi, Array, ArraySpec, Coordinate, PointsSpec


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

    def __init__(self, points, array, array_spec, radius=1.0):

        self.points = points
        self.array = array
        self.array_spec = array_spec
        self.radius = radius

    def setup(self):

        # todo: ensure that both rois are the same
        points_roi = self.spec[self.points].roi
        if self.array_spec is None:
            self.array_spec = ArraySpec(
                roi=points_roi.copy(),
                voxel_size=Coordinate((1,) * points_roi.dims()),
                interpolatable=False,
                dtype=np.int32,
            )

        if self.array_spec.roi is None:
            self.array_spec.roi = points_roi.copy()

        self.provides(self.array, self.array_spec)

    def prepare(self, request):
        self.array_spec.roi = request[self.array].roi
        request[self.points] = PointsSpec(roi=self.array_spec.roi)

    def process(self, batch, request):

        points = batch.points[self.points]
        assert len(points.data.items()) > 0, "No Swc Points in enlarged Roi."

        voxel_size = self.array_spec.voxel_size

        # get roi used for creating the new array (points_roi does not
        # necessarily align with voxel size)
        enlarged_array_roi = points.spec.roi.snap_to_grid(voxel_size)
        offset = enlarged_array_roi.get_begin() / voxel_size
        shape = enlarged_array_roi.get_shape() / voxel_size
        array_roi = Roi(offset, shape)
        array_data = np.zeros(shape, dtype=self.array_spec.dtype)

        # iterate through swc points
        labels = np.unique([p.label_id for p in points.data.values()])
        for label in labels:
            binarized = np.zeros_like(array_data, dtype=np.bool)
            for p in points.data.values():
                if p.label_id == label:
                    if p.parent_id in points.data.keys():
                        p1 = (p.location / voxel_size - offset).astype(int)
                        p2 = (points.data[p.parent_id].location / voxel_size).astype(
                            int
                        ) - offset
                        binarized = self._rasterize_line_segment(p1, p2, binarized)

            array_data[binarized] = label

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

        # use Bresenham's line algorithm based on:
        # http://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
        line_segment_points = self._bresenhamline(point, parent, max_iter=-1)

        if line_segment_points.shape[0] > 0:
            idx = np.transpose(line_segment_points.astype(int))
            if (
                np.max(idx[0]) >= skeletonized.shape[0]
                or np.max(idx[1]) >= skeletonized.shape[1]
                or np.max(idx[2]) >= skeletonized.shape[2]
            ):
                print(np.max(idx, axis=0), skeletonized.shape)
            skeletonized[idx[0], idx[1], idx[2]] = True
        skeletonized[point[0], point[1], point[2]] = True

        return skeletonized
