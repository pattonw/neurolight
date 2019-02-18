import numpy as np
from gunpowder import *
from gunpowder.profiling import Timing
import h5py

logger = logging.getLogger(__name__)


class SwcPoint(Point):

    def __init__(self, location, point_id, parent_id, label_id):

        super(SwcPoint, self).__init__(location)

        self.thaw()
        self.point_id = point_id
        self.parent_id = parent_id
        self.label_id = label_id
        self.freeze()

    def copy(self):
        return SwcPoint(self.location, self.point_id, self.parent_id, self.label_id)


class SwcSource(BatchProvider):
    """Read points of a skeleton from a hdf dataset.
    --> todo: should also be possible to read from swc file directly with considering offset and resolution

    Each line in the file represents one point as::

        point_id, structure identifier (soma, axon, ...), x, y, z, radius, parent_id

    where ``parent_id`` can be -1 to indicate no parent.

    Args:

        filename (``string``):

            The HDF5 file.

        dataset (``string``):

            Array key to dataset names that this source offers.

        points (``tuple`` of :class:`PointsKey`):

            The key of the points set to create.

        point_specs (``dict``, :class:`PointsKey`, optional):

            An optional dictionary of point keys to point specs to overwrite
            the array specs automatically determined from the data file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.

        return_env (``bool``, optional):

            If parent and children nodes should be returned.

        scale (scalar or array-like, optional):

            An optional scaling to apply to the coordinates of the points.
            This is useful if the points refer to voxel positions to convert them to world units.
    """

    def __init__(self, filename, dataset, points, point_specs=None, return_env=False, scale=None):

        self.filename = filename
        self.dataset = dataset
        self.points = points
        self.point_specs = point_specs
        self.return_env = return_env
        self.scale = scale

        # variables to keep track of swc skeleton graphs
        self.ndims = 3
        self.data = None
        self.child_to_parent = None
        self.parent_to_children = None
        self.sources = None

    def setup(self):

        self._read_points()

        if self.point_specs is not None:

            assert len(self.point_specs) == len(self.points), 'Number of point keys and point specs differ!'

            for points_key, points_spec in zip(self.points, self.point_specs):
                self.provides(points_key, points_spec)

        else:

            min_bb = Coordinate(np.floor(np.amin(self.data[:, :self.ndims], 0)))
            max_bb = Coordinate(np.ceil(np.amax(self.data[:, :self.ndims], 0)) + 1)
            roi = Roi(min_bb, max_bb - min_bb)

            for points_key in self.points:
                self.provides(points_key, PointsSpec(roi=roi))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for points_key in self.points:

            if points_key not in request:
                continue

            # get points for output size / center region
            min_bb = request[points_key].roi.get_begin()
            max_bb = request[points_key].roi.get_end()

            logger.debug(
                "SWC points source got request for %s",
                request[points_key].roi)

            point_filter = np.ones((self.data.shape[0],), dtype=np.bool)
            for d in range(self.ndims):
                point_filter = np.logical_and(point_filter, self.data[:, d] >= min_bb[d])
                point_filter = np.logical_and(point_filter, self.data[:, d] < max_bb[d])

            points_data = self._get_points(point_filter)
            points_spec = PointsSpec(roi=request[points_key].roi.copy())

            # get neighboring points in order to draw skeleton correctly

            if len(points_data) < self.data.shape[0] and self.return_env:

                points_env = PointsKey(points_key.identifier + '_ENV')
                points_env_data = {}
                points_env_spec = points_spec.copy()

                # get children and parent of center points and include them to points_env
                for p in points_data:
                    if p not in self.sources:
                        parent_id = self.child_to_parent[p]
                        parent = self._get_point(self.data[:, 3] == parent_id)
                        points_env_data[int(parent_id)] = parent

                    if p in self.parent_to_children.keys():
                        for child_id in self.parent_to_children[p]:
                            child = self._get_point(self.data[:, 3] == child_id)
                            points_env_data[int(child_id)] = child

                    # add also original points to points env
                    points_env_data[p] = points_data[p].copy()

                locs = np.asarray([p.location for p in points_env_data.values()])
                neg_grow = np.round(np.maximum(np.zeros((3,)),
                                               np.asarray(points_spec.roi.get_begin()) - np.min(locs, axis=0)))
                pos_grow = np.round(np.maximum(np.zeros((3,)),
                                               np.max(locs, axis=0) - np.asarray(points_spec.roi.get_end())))

                neg_grow += np.asarray(self.scale) - np.mod(neg_grow, self.scale)
                pos_grow += np.asarray(self.scale) - np.mod(pos_grow, self.scale)

                points_env_spec.roi = points_env_spec.roi.grow(Coordinate(neg_grow), Coordinate(pos_grow))

                batch.points[points_env] = Points(points_env_data, points_env_spec)

            batch.points[points_key] = Points(points_data, points_spec)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _open_file(self, filename):
        return h5py.File(filename, 'r')

    def _get_points(self, point_filter):

        filtered = self.data[point_filter]
        return {
            int(p[self.ndims]): SwcPoint(
                p[:self.ndims],
                int(p[self.ndims]),
                int(p[self.ndims + 1]),
                int(p[self.ndims + 2])
            )
            for p in filtered
        }

    def _get_point(self, point_filter):

        filtered = self.data[point_filter][0]
        return SwcPoint(
            filtered[:self.ndims],
            int(filtered[self.ndims]),
            int(filtered[self.ndims + 1]),
            int(filtered[self.ndims + 2])
        )

    def _label_skeleton(self, p, label_id):
        self.data[self.data[:, 3] == p, 5] = label_id
        if p in self.parent_to_children.keys():
            for child in self.parent_to_children[p]:
                self._label_skeleton(child, label_id)

    def _label_skeletons(self):

        self.child_to_parent = {}
        self.parent_to_children = {}
        self.sources = []

        # data = [x, y, z, point_id, parent_id, label_id]
        # indices = 0: x, 1: y, 2:z, 3: point_id, 4: parent_id, 5: label_id
        for p in self.data:

            if p[3] == p[4]:
                self.sources.append(int(p[3]))
            else:
                self.child_to_parent[p[3]] = p[4]

                if p[4] in self.parent_to_children:
                    self.parent_to_children[p[4]].append(p[3])
                else:
                    self.parent_to_children[p[4]] = [p[3]]

        label_id = 1
        for source in self.sources:
            self._label_skeleton(source, label_id)
            label_id += 1

    def _read_points(self):

        with self._open_file(self.filename) as data_file:

            if self.dataset not in data_file:
                raise RuntimeError("%s not in %s" % (self.dataset, self.filename))

            points = data_file[self.dataset]

            # data = [x, y, z, point_id, parent_id, label_id]
            self.data = np.transpose(np.array([points[:, 2], points[:, 3], points[:, 4], points[:, 0], points[:, 6]]))
            self.data = np.concatenate((self.data, np.zeros((self.data.shape[0], 1), dtype=self.data.dtype)), axis=1)

            # separate skeletons and assign labels
            self._label_skeletons()

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
