import numpy as np
from gunpowder import *


class BinarizeLabels(BatchFilter):
    def __init__(self, labels, labels_binary):

        self.labels = labels
        self.labels_binary = labels_binary

    def setup(self):

        spec = self.spec[self.labels].copy()
        spec.dtype = np.uint8
        self.provides(self.labels_binary, spec)

    def prepare(self, request):
        request[self.labels] = request[self.labels_binary]
        pass

    def process(self, batch, request):

        spec = batch[self.labels].spec.copy()
        spec.dtype = np.uint8

        binarized = Array(
            data=(batch[self.labels].data > 0).astype(np.uint8), spec=spec
        )

        batch[self.labels_binary] = binarized
