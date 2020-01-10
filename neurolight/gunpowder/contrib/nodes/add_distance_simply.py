import numpy as np
from gunpowder import BatchFilter, ArrayKey, BatchRequest, Array
from scipy.ndimage.morphology import distance_transform_edt

from typing import List
import logging
import collections

logger = logging.getLogger(__name__)


class AddDistance(BatchFilter):
    def __init__(
        self,
        label_array_key,
        distance_array_key,
        distance_array_spec=None,
        label_id=None,
    ):

        self.label_array_key = label_array_key
        self.distance_array_key = distance_array_key
        self.distance_array_spec = distance_array_spec
        if not isinstance(label_id, collections.Iterable) and label_id is not None:
            label_id = (label_id,)
        self.label_id = label_id

    def setup(self):
        self.enable_autoskip()

        spec = self.spec[self.label_array_key].copy()
        spec.dtype = (
            self.distance_array_spec.dtype
            if self.distance_array_spec is not None
            else np.float32
        )
        self.dtype = spec.dtype

        self.provides(self.distance_array_key, spec)

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        deps[self.label_array_key] = request[self.distance_array_key].copy()

        return deps

    def process(self, batch, request: BatchRequest):
        labels = batch[self.label_array_key].data
        spec = batch[self.label_array_key].spec.copy()
        spec.dtype = self.dtype

        binarized = labels != 0
        dt = -distance_transform_edt(
            np.logical_not(binarized), sampling=spec.voxel_size
        ).astype(self.dtype)

        expanded = Array(data=dt, spec=spec)

        batch.arrays[self.distance_array_key] = expanded
