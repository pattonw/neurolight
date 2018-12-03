import logging
from gunpowder import CsvPointsSource, Point
import numpy as np
import h5py
import pdb

logger = logging.getLogger(__name__)

class SwcPoint(Point):

    def __init__(self, location, point_id, parent_id):

        super(SwcPoint, self).__init__(location)

        self.thaw()
        self.point_id = point_id
        self.parent_id = parent_id
        self.freeze()


class SwcSource(CsvPointsSource):
    '''Read points of a skeleton from a hdf dataset.
    --> todo: should also be possible to read from swc file directly with considering offset and resolution

    Each line in the file represents one point as::

        point_id, structure identifier (soma, axon, ...), x, y, z, radius, parent_id

    where ``parent_id`` can be -1 to indicate no parent.

    Args:

        filename (``string``):

            The HDF5 file.

        dataset (``string``):

            Array key to dataset names that this source offers.

        points (:class:`PointsKey`):

            The key of the points set to create.

        points_spec (:class:`PointsSpec`, optional):

            An optional :class:`PointsSpec` to overwrite the points specs
            automatically determined from the CSV file. This is useful to set
            the :class:`Roi` manually, for example.

        scale (scalar or array-like):

            An optional scaling to apply to the coordinates of the points.
            This is useful if the points refer to voxel positions to convert them to world units.
    '''
    def __init__(self, filename, dataset, points, points_spec=None, scale=None):

        super(SwcSource, self).__init__(filename, points, points_spec, scale)
        self.dataset = dataset
        self.ndims = 3

    def _open_file(self, filename):
        return h5py.File(filename, 'r')

    def _get_points(self, point_filter):

        filtered = self.data[point_filter]
        return {
            int(p[self.ndims]): SwcPoint(
                p[:self.ndims],
                int(p[self.ndims]),
                int(p[self.ndims + 1]),
            )
            for p in filtered
        }

    def _read_points(self):

        with self._open_file(self.filename) as data_file:

            if self.dataset not in data_file:
                raise RuntimeError("%s not in %s" % (self.dataset, self.filename))

            points = data_file[self.dataset]

            # data = [x, y, z, point_id, parent_id]
            self.data = np.transpose(np.array([points[:, 2], points[:, 3], points[:, 4], points[:, 0], points[:, 6]]))

            resolution = None
            if data_file[self.dataset].attrs.__contains__('resolution'):
                resolution = data_file[self.dataset].attrs.get('resolution')

            if self.scale is not None:
                self.data[:, :self.ndims] *= self.scale
                if resolution is not None:
                    if resolution != self.scale:
                        logger.warning("WARNING: File %s contains resolution information "
                                       "for %s (dataset %s). However, voxel size has been set to scale factor %s." 
                                       "This might not be what you want.",
                                       self.filename, points, self.dataset, self.scale)
            elif resolution is not None:
                self.data[:, :self.ndims] *= resolution
            else:
                logger.warning("WARNING: No scaling factor or resolution information in file %s"
                               "for %s (dataset %s). So points refer to voxel positions, "
                               "this might not be what you want.",
                               self.filename, points, self.dataset)



