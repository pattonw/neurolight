from mclahe import mclahe
import numpy as np

from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch
from gunpowder.array import Array

from gunpowder import BatchFilter

import itertools


class mCLAHE(BatchFilter):
    """Node utilizing mclahe CLAHE implementation:
    see https://github.com/VincentStimper/mclahe for documentation

    Args:

        arrays (List, :class:`ArrayKey`):

            The arrays to modify.

        kernel_size (int or array_like :int:):

            See mclahe documentation

        clip_limit (float):

            See mclahe documentation

        nbins (int):

            See mclahe documentation

        slice_wise (bool):
        
            whether to expand kernel to array dimensions with prepended 1's,
            or to sice over the first m dims such that the remaining
            dimensions match the kernel dimensions.

        adaptive_hist_range (bool):

            see mclahe documentation
    """

    def __init__(
        self,
        arrays,
        kernel_size,
        clip_limit=0.01,
        nbins=256,
        slice_wise=False,
        adaptive_hist_range=False,
        output_arrays=None,
        context=None,
    ):
        self.arrays = arrays
        self.kernel_size = np.array(kernel_size)
        self.clip_limit = clip_limit
        self.nbins = nbins
        self.slice_wise = slice_wise
        self.adaptive_hist_range = adaptive_hist_range
        self.output_arrays = output_arrays
        if self.output_arrays is not None:
            assert len(output_arrays) == len(arrays)
        else:
            self.output_arrays = self.arrays
        self.context = context

    def setup(self):
        self.enable_autoskip()
        for in_key, out_key in zip(self.arrays, self.output_arrays):
            if out_key == in_key:
                self.updates(out_key, self.spec[in_key])
            else:
                self.provides(out_key, self.spec[in_key])

    def prepare(self, request):
        deps = BatchRequest()
        for in_key, out_key in zip(self.arrays, self.output_arrays):
            spec = request[out_key].copy()
            if self.context is not None:
                spec.roi = spec.roi.grow(self.context, self.context)
            deps[in_key] = spec
        return deps

    def process(self, batch, request):
        output = Batch()

        for in_key, out_key in zip(self.arrays, self.output_arrays):
            array = batch[in_key]
            data = array.data
            shape = data.shape
            data_dims = len(shape)
            kernel_dims = len(self.kernel_size)
            extra_dims = data_dims - kernel_dims
            if self.slice_wise:
                for index in itertools.product(*[range(s) for s in shape[:extra_dims]]):
                    data[index] = mclahe(
                        data[index],
                        kernel_size=self.kernel_size,
                        clip_limit=self.clip_limit,
                        n_bins=self.nbins,
                        use_gpu=False,
                        adaptive_hist_range=self.adaptive_hist_range,
                    )
            else:
                full_kernel = np.array(
                    (1,) * extra_dims + tuple(self.kernel_size), dtype=int
                )
                data = mclahe(
                    data,
                    kernel_size=full_kernel,
                    clip_limit=self.clip_limit,
                    n_bins=self.nbins,
                    # use_gpu=False,
                ).astype(self.spec[out_key].dtype)
            output[out_key] = Array(data, array.spec).crop(request[out_key].roi)
        return output
