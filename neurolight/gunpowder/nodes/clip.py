import numpy as np
from gunpowder import *


class Clip(BatchFilter):

    def __init__(self, array, min=None, max=None):

        self.array = array
        self.min = min
        self.max = max

    def process(self, batch, request):

        np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]

        if self.min is None:
            self.min = np.min(array.data)
        if self.max is None:
            self.max = np.max(array.data)

        for c in range(array.data.shape[0]):
            hist = np.histogram(array.data[c], range(4096), density=True)
            #print('histogram density at max: ', hist[0][self.max])

        array.data = np.clip(array.data, self.min, self.max).astype(array.spec.dtype)