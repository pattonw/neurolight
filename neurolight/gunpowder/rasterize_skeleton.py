import numpy as np
from gunpowder import *
from scipy import ndimage


class RasterizeSkeleton(BatchFilter):
    """Draw skeleton into a binary array given a swc.

        Args:

            points (:class:``PointsKey``):
                The key of the points to form the skeleton.

            array (:class:``ArrayKey``):
                The key of the binary array to create.

            array_spec (:class:``ArraySpec``, optional):

                The spec of the array to create. Use this to set the datatype and
                voxel size.

            iteration (``float``, optional):

                The number of iterations to apply binary dilation to the skeleton.
        """

    def __init__(self, points, array, array_spec=None, points_env=None, iteration=1):

        self.points = points
        self.array = array
        self.array_spec = array_spec
        self.points_env = points_env
        self.iteration = iteration

    def setup(self):

        points_roi = self.spec[self.points].roi

        if self.array_spec is None:
            self.array_spec = ArraySpec(roi=points_roi.copy(),
                                        voxel_size=Coordinate((1,) * points_roi.dims()),
                                        interpolatable=False,
                                        dtype=np.int32
                                        )

        if self.array_spec.roi is None:
            self.array_spec.roi = points_roi.copy()

        self.provides(self.array, self.array_spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        points = batch.points[self.points]

        if self.points_env is not None:
            if self.points_env in batch.points:
                points_env = batch.points[self.points_env]

                # exchange environment data without specifying size in request
                # apply random shift from random location to swc env
                random_shift = None
                for k, p in points.data.items():
                    if k in points_env.data.keys():
                        random_shift = points_env.data[k].location - p.location
                        break

                if random_shift is not None:
                    for k, p in points_env.data.items():
                        p.location -= random_shift

                    points_env.spec.roi.set_offset(points_env.spec.roi.get_offset() - tuple(random_shift.astype(int)))

                points = points_env

        assert len(points.data.items()) > 0, 'No Swc Points in enlarged Roi.'

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
                        p2 = (points.data[p.parent_id].location / voxel_size).astype(int) - offset
                        binarized = self._rasterize_line_segment(p1, p2, binarized)

            if self.iteration >= 1:
                binarized = ndimage.binary_dilation(binarized, iterations=self.iteration)

            array_data[binarized] = label

        print('sum > 0: ', np.sum(array_data > 0), ', num points in roi: ', len(points.data.items()))

        array_spec = ArraySpec(roi=array_roi * voxel_size,
                               voxel_size=voxel_size,
                               interpolatable=False,
                               dtype=self.array_spec.dtype
                               )
        array = Array(data=array_data, spec=array_spec)

        array = array.crop(request[self.array].roi)
        batch.arrays[self.array] = array

        # get rid of all env point keys which are not in request
        delete_point_keys = []
        for points_key in batch.points:
            if points_key not in request:
                delete_point_keys.append(points_key)
        for points_key in delete_point_keys:
            del batch.points[points_key]

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

        # use Bresenham's line algorithm
        # based on http://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
        line_segment_points = self._bresenhamline(point, parent, max_iter=-1)

        if line_segment_points.shape[0] > 0:
            idx = np.transpose(line_segment_points.astype(int))
            if np.max(idx[0]) >= skeletonized.shape[0] or np.max(idx[1]) >= skeletonized.shape[1] \
                    or np.max(idx[2]) >= skeletonized.shape[2]:
                print(np.max(idx, axis=0), skeletonized.shape)
            skeletonized[idx[0], idx[1], idx[2]] = True
        skeletonized[point[0], point[1], point[2]] = True

        return skeletonized
