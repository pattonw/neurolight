import numpy as np
from gunpowder import BatchFilter, BatchRequest, Array, ArrayKey
from typing import Optional


class BinarizeLabels(BatchFilter):
    def __init__(self, labels, labels_binary: Optional[ArrayKey] = None):

        self.labels = labels
        self.labels_binary = labels_binary

    def setup(self):
        self.enable_autoskip()

        spec = self.spec[self.labels].copy()
        spec.dtype = np.uint8
        if self.labels_binary is not None:
            self.provides(self.labels_binary, spec)
        else:
            self.updates(self.labels, spec)

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.labels] = (
            request[self.labels_binary]
            if self.labels_binary is not None
            else request[self.labels]
        )
        return deps

    def process(self, batch, request):

        spec = batch[self.labels].spec.copy()
        spec.dtype = np.uint8

        binarized = Array(
            data=(batch[self.labels].data > 0).astype(np.uint8), spec=spec
        )

        if self.labels_binary is not None:
            batch[self.labels_binary] = binarized
        else:
            batch[self.labels] = binarized
