from gunpowder import *
from gunpowder.nodes import BatchFilter
import numpy as np
from scipy import ndimage


class RasterizeSwcSkeleton(BatchFilter):
    '''Draw skeleton into a binary array given a swc.

        Args:

            points (:class:``PointsKeys``):
                The key of the points to form the skeleton.

            array (:class:``ArrayKey``):
                The key of the binary array to create.

            array_spec (:class:``ArraySpec``, optional):

                The spec of the array to create. Use this to set the datatype and
                voxel size.

            distance (``float`` or ``tuple`` of ``floats``, optional):

                The distance between two points in the swc. Required to grow the ROI in order to get neighboring
                points to complete the skeleton at the border of the request. E.g. mouselight: 10um

            iteration (``float``, optional):

                The number of iterations to apply binary dilation to the skeleton.
        '''

    def __init__(self, points, array, array_spec, distance=10, iteration=1):

        self.points = points
        self.array = array
        self.array_spec = array_spec
        self.distance = distance
        self.iteration = iteration

    def setup(self):

        points_roi = self.spec[self.points].roi

        if self.array_spec.voxel_size is None:
            self.array_spec.voxel_size = Coordinate((1,) * points_roi.dims())

        if self.array_spec.dtype is None:
            self.array_spec.dtype = np.uint8

        if self.array_spec.interpolatable is None:
            self.array_spec.interpolatable = False

        self.array_spec.roi = points_roi.copy()
        self.provides(self.array, self.array_spec)

    def prepare(self, request):

        dims = self.array_spec.roi.dims()

        # grow region to add neighboring points
        if len(self.distance) == 1:
            context = self.distance.repeat(dims)
        else:
            context = self.distance

        points_roi = request[self.points].roi.grow(Coordinate(context), Coordinate(context))

        request[self.points] = PointsSpec(roi=points_roi)

    def process(self, batch, request):

        points = batch.points[self.points]
        points_spec = batch.points[self.points].spec.copy()
        voxel_size = self.array_spec.voxel_size

        # get roi used for creating the new array (points_roi does not
        # necessarily align with voxel size)
        enlarged_vol_roi = points.spec.roi.snap_to_grid(voxel_size)
        offset = enlarged_vol_roi.get_begin() / voxel_size
        shape = enlarged_vol_roi.get_shape() / voxel_size
        skeletonized_roi = Roi(offset, shape)

        skeletonized = np.zeros(shape, dtype=self.array_spec.dtype)

        if len(points.data.items()) == 0:
            print('No Swc Points in enlarged Roi.')

        # iterate through swc points
        for point in points.data.items():
            if point[1].parent_id in points.data:
                p1 = (point[1].location / voxel_size - offset).astype(int)
                p2 = (points.data[point[1].parent_id].location / voxel_size).astype(int) - offset
                skeletonized = self.__rasterize_line_segment(p1, p2, skeletonized)

        if self.iteration >= 1:
            skeletonized = ndimage.binary_dilation(skeletonized, structure=np.ones((3,3,3)),
                                                   iterations=self.iteration)

        skeleton = Array(data=skeletonized, spec=ArraySpec(dtype=self.array_spec.dtype, roi=skeletonized_roi * voxel_size,
                                                           interpolatable=False, voxel_size=voxel_size))
        skeleton = skeleton.crop(request[self.array].roi)
        batch.arrays[self.array] = skeleton

        # restore requested ROI of points
        if self.points in request:
            request_roi = request[self.points].roi
            points.spec.roi = request_roi
            delete_points = []
            for i, p in points.data.items():
                if not request_roi.contains(p.location):
                    delete_points.append(i)

            for i in delete_points:
                del points.data[i]

        print('sum > 0: ', np.sum(skeleton.data > 0), ', num points in roi: ', len(points.data.items()))

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


    def __rasterize_line_segment(self, point, parent, skeletonized):

        # use Bresenham's line algorithm
        # based on http://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
        line_segment_points = self._bresenhamline(point, parent, max_iter=-1)
        idx = np.transpose(line_segment_points.astype(int))
        skeletonized[idx[0], idx[1], idx[2]] = True
        skeletonized[point[0], point[1], point[2]] = True

        return skeletonized