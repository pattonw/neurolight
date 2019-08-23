import numpy as np
from gunpowder import *


class ConvertRgbToHls(BatchFilter):

    def __init__(self, rgb, hls):
        self.rgb = rgb
        self.hls = hls

    def setup(self):
        spec = self.spec[self.rgb].copy()
        spec.dtype = np.float32
        self.provides(self.hls, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = batch[self.rgb].spec.copy()
        spec.dtype = np.float32

        rgb = batch[self.rgb].data
        hls = getHls(rgb)

        batch[self.hls] = Array(data=hls.astype(np.float32), spec=spec)


class ConvertRgbToHlsVector(BatchFilter):

    def __init__(self, rgb, hls_vector):
        self.rgb = rgb
        self.hls_vector = hls_vector

    def setup(self):
        spec = self.spec[self.rgb].copy()
        spec.dtype = np.float32
        self.provides(self.hls_vector, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = batch[self.rgb].spec.copy()
        spec.dtype = np.float32

        rgb = batch[self.rgb].data
        hls = getHls(rgb)

        cos = hls[2] * np.cos(hls[0] * 2 * np.pi)
        sin = hls[2] * np.sin(hls[0] * 2 * np.pi)

        hls_vector = np.stack([cos, sin, hls[1]], axis=0)

        batch[self.hls_vector] = Array(data=hls_vector.astype(np.float32), spec=spec)


def getHls(rgb):

    # c, z, y, x
    # convert rgb to hls, assume that rgb is normalized and float [0,1]
    maxc = np.max(rgb, axis=0)
    minc = np.min(rgb, axis=0)
    dst = np.zeros_like(rgb, dtype=np.float32)

    dst[1] = (minc + maxc) / 2.0

    not_white = minc != maxc
    minus = maxc - minc
    plus = maxc + minc

    with np.errstate(divide='ignore', invalid='ignore'):

        dark = dst[1] < 0.5
        idx = np.logical_and(dark, not_white)
        dst[2, idx] = np.divide(minus[idx], plus[idx])

        idx = np.logical_and(np.logical_not(dark), not_white)
        dst[2, idx] = np.divide(minus[idx], (2.0 - plus)[idx])

        rc = np.divide((maxc - rgb[0]), minus)
        gc = np.divide((maxc - rgb[1]), minus)
        bc = np.divide((maxc - rgb[2]), minus)

        idx = np.logical_and(rgb[0] == maxc, not_white)
        dst[0, idx] = bc[idx] - gc[idx]
        idx = np.logical_and(rgb[1] == maxc, not_white)
        dst[0, idx] = 2.0 + rc[idx] - bc[idx]
        idx = np.logical_and(rgb[2] == maxc, not_white)
        dst[0, idx] = (4.0 + gc - rc)[idx]

        dst[0] = (dst[0] / 6.0) % 1.0

    return dst
