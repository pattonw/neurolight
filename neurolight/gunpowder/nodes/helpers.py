import gunpowder as gp
import numpy as np

import copy


class BinarizeGt(gp.BatchFilter):
    def __init__(self, gt, gt_binary):

        self.gt = gt
        self.gt_binary = gt_binary

    def setup(self):

        spec = self.spec[self.gt].copy()
        spec.dtype = np.uint8
        self.provides(self.gt_binary, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        spec.dtype = np.int32

        binarized = gp.Array(data=(batch[self.gt].data > 0).astype(np.int32), spec=spec)

        batch[self.gt_binary] = binarized


class Crop(gp.BatchFilter):
    def __init__(self, input_array: gp.ArrayKey, output_array: gp.ArrayKey):

        self.input_array = input_array
        self.output_array = output_array

    def setup(self):

        spec = self.spec[self.input_array].copy()
        self.provides(self.output_array, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):
        input_data = batch[self.input_array].data
        input_spec = batch[self.input_array].spec
        input_roi = input_spec.roi
        output_roi_shape = request[self.output_array].roi.get_shape()
        shift = (input_roi.get_shape() - output_roi_shape) / 2
        output_roi = gp.Roi(shift, output_roi_shape)
        output_data = input_data[
            tuple(
                map(
                    slice,
                    output_roi.get_begin() / input_spec.voxel_size,
                    output_roi.get_end() / input_spec.voxel_size,
                )
            )
        ]
        output_spec = copy.deepcopy(input_spec)
        output_spec.roi = output_roi

        output_array = gp.Array(output_data, output_spec)

        batch[self.output_array] = output_array


class RejectIfEmpty(gp.BatchFilter):
    def __init__(self, ensure_nonempty: gp.ArrayKey, request_limit: int = 100):
        self.ensure_nonempty = ensure_nonempty
        self.request_limit = request_limit

    def setup(self):
        self.enable_autoskip()
        self.updates(
            self.ensure_nonempty, copy.deepcopy(self.spec[self.ensure_nonempty])
        )

    def prepare(self, request):
        return copy.deepcopy(request)

    def process(self, batch, request):

        k = 0
        found_valid = False
        while (
            not found_valid
            and k < self.request_limit
            and self.ensure_nonempty in request
        ):
            batch = self.get_upstream_provider().request_batch(request)
            if len(batch[self.ensure_nonempty].graph.nodes) > 0:
                found_valid = True
            k += 1

        return batch
