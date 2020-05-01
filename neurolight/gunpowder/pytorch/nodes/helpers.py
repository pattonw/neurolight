import gunpowder as gp
import torch

import copy
from typing import List


class UnSqueeze(gp.BatchFilter):
    def __init__(self, arrays: List[gp.ArrayKey]):
        self.arrays = arrays

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        for array in self.arrays:
            deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        for array in self.arrays:
            outputs[array] = copy.deepcopy(batch[array])
            outputs[array].data = (
                torch.from_numpy(batch[array].data).unsqueeze(0).numpy()
            )
        return outputs


class Squeeze(gp.BatchFilter):
    def __init__(self, arrays: List[gp.ArrayKey]):
        self.arrays = arrays

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        for array in self.arrays:
            if array in request:
                deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        for array in self.arrays:
            if array in batch:
                outputs[array] = copy.deepcopy(batch[array])
                outputs[array].data = torch.from_numpy(batch[array].data).squeeze(0).numpy()
        return outputs
