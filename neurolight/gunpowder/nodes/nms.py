import torch
import numpy as np

from neurolight.networks.pytorch import NMS
from gunpowder import BatchFilter, ArrayKey, BatchRequest, Array, Coordinate

import logging

logger = logging.getLogger(__name__)


class NonMaxSuppression(BatchFilter):
    """Run non max suppression on an input array.

        Args:

            array (:class:``ArrayKey``):

                The array to run nms on.

            nms (:class:``ArrayKey``):

                The array to store the nms in.

            threshold (``float``, optional):

                The minimum value be considered for a local maxima.
        """

    def __init__(
        self, array: ArrayKey, nms: ArrayKey, window_size: Coordinate, threshold: float = None
    ):

        self.array = array
        self.nms = nms
        self.window_size = window_size
        self.threshold = threshold

    def setup(self):
        self.enable_autoskip()
        provided_spec = self.spec[self.array].copy()
        provided_spec.dtype = bool
        self.provides(self.nms, provided_spec)

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        deps[self.array] = request[self.nms].copy()

        return deps

    def process(self, batch, request: BatchRequest):
        data = batch[self.array].data
        voxel_size = batch[self.array].spec.voxel_size
        window_size = self.window_size / voxel_size

        if self.threshold is None:
            data_min = np.min(data)
            data_med = np.median(data)
            threshold = (data_min + data_med) / 2
        else:
            threshold = self.threshold

        nms_op = NMS(window_size, threshold)
        nms_input = torch.from_numpy(data)
        maxima = nms_op(nms_input)

        spec = batch[self.array].spec.copy()
        spec.dtype = bool

        batch.arrays[self.nms] = Array(maxima.numpy(), spec)
