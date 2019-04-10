import numpy as np
from gunpowder import *


class CountOverlap(BatchFilter):

    def __init__(self, gt, gt_overlap, maxnuminst=None):

        self.gt = gt
        self.gt_overlap = gt_overlap
        self.maxnuminst = maxnuminst
        self.dims = None

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        spec = self.spec[self.gt].copy()
        self.provides(self.gt_overlap, spec)

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        array = batch[self.gt].data

        num_channels = len(array.shape) - self.dims
        assert num_channels <= 1, 'Sorry, dont know what to do with more than one channel dimension.'

        overlap = np.sum((array > 0).astype('uint16'), axis=0)
        print('overlap: ', np.sum(overlap), np.unique(overlap))
        if self.maxnuminst is not None:
            overlap = np.clip(overlap, 0, self.maxnuminst)

        spec.dtype = np.int32

        batch[self.gt_overlap] = Array(data=(overlap).astype(np.int32),
                spec=spec)
