from __future__ import print_function
import numpy as np
import h5py
from gunpowder.roi import Roi
from gunpowder.nodes.hdf5like_source_base import Hdf5LikeSource
from gunpowder.profiling import Timing
from gunpowder.batch import Batch
from gunpowder.array import Array
from gunpowder.coordinate import Coordinate
from gunpowder.array_spec import ArraySpec


class Hdf5ChannelSource(Hdf5LikeSource):
    '''An HDF5 data source with channels

    Args:

        filename (``string``):

            The HDF5 file.

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to dataset names that this source offers.

        channel_ids (``dict``, :class:`ArrayKey` -> ``int``):

            Dictionary of array keys to dataset channel index for this source.

        data_format (``string``):

            String in which dimension channel is stored, ``channels_first`` if channels are in the first dimension
            (default) or ``channels_last`` in the last one.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the data file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.
    '''

    def __init__(
            self,
            filename,
            datasets,
            channel_ids,
            data_format='channels_first',
            array_specs=None):

        super(Hdf5ChannelSource, self).__init__(filename, datasets, array_specs)
        self.channel_ids = channel_ids
        self.data_format = data_format

    def _open_file(self, filename):
        return h5py.File(filename, 'r')

    def setup(self):
        with self._open_file(self.filename) as data_file:
            for (array_key, ds_name) in self.datasets.items():

                if ds_name not in data_file:
                    raise RuntimeError("%s not in %s" % (ds_name, self.filename))
                spec = self.__read_spec(array_key, data_file, ds_name)

                self.provides(array_key, spec)

    def __read(self, data_file, ds_name, roi, channel_id):
        c = len(data_file[ds_name].shape) - self.ndims - 1
        if self.data_format == 'channels_first':
            return np.asarray(data_file[ds_name][(slice(channel_id, channel_id + 1),) + (slice(None),)*c + roi.to_slices()])
        if self.data_format == 'channels_last':
            return np.reshape(np.asarray(data_file[ds_name][(slice(None),)*c + roi.to_slices() +
                                                            (slice(channel_id, channel_id + 1),)]),roi.get_shape())

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        with self._open_file(self.filename) as data_file:
            for (array_key, request_spec) in request.array_specs.items():

                voxel_size = self.spec[array_key].voxel_size

                # scale request roi to voxel units
                dataset_roi = request_spec.roi / voxel_size

                # shift request roi into dataset
                dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size

                # create array spec
                array_spec = self.spec[array_key].copy()
                array_spec.roi = request_spec.roi

                # add array to batch
                batch.arrays[array_key] = Array(
                    self.__read(data_file, self.datasets[array_key], dataset_roi, self.channel_ids[array_key]),
                    array_spec)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_spec(self, array_key, data_file, ds_name):

        dataset = data_file[ds_name]

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = self._get_voxel_size(dataset)
            if voxel_size is None:
                voxel_size = Coordinate((1,) * len(dataset.shape))
            spec.voxel_size = voxel_size

        self.ndims = len(spec.voxel_size)

        if spec.roi is None:
            offset = self._get_offset(dataset)
            if offset is None:
                offset = Coordinate((0,) * self.ndims)

            if self.data_format == 'channels_first':
                shape = Coordinate(dataset.shape[-self.ndims:])
            if self.data_format == 'channels_last':
                shape = Coordinate(dataset.shape[0:self.ndims])
            spec.roi = Roi(offset, shape * spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, ("dtype %s provided in array_specs for %s, "
                                                 "but differs from dataset %s dtype %s" %
                                                 (self.array_specs[array_key].dtype,
                                                  array_key, ds_name, dataset.dtype))
        else:
            spec.dtype = dataset.dtype

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8  # assuming this is not used for labels
            ]

        return spec