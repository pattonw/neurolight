import gunpowder as gp
from gunpowder import Roi
import numpy as np


#todo: list of channels to be merged
class MergeChannel(gp.BatchFilter):

    def __init__(self, fg, bg, raw):
        self.fg = fg
        self.bg = bg
        self.raw = raw

    def setup(self):
        spec = self.spec[self.fg].copy()
        spec.roi = Roi((0,) + spec.roi.get_offset(), (3,) + spec.roi.get_shape())
        self.provides(self.raw, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = self.spec[self.fg].copy()
        voxel_size = (1,) + spec.voxel_size
        merged = np.stack([batch[self.fg].data, batch[self.bg].data], axis=0)

        batch[self.raw] = gp.Array(data=merged.astype(spec.dtype),
                                   spec=gp.ArraySpec(dtype=spec.dtype,
                                                     roi=Roi((0, 0, 0, 0), merged.shape) * voxel_size,
                                                     interpolatable=True,
                                                     voxel_size=voxel_size))
