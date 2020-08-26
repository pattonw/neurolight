from gunpowder import BatchFilter, BatchRequest, Batch, ArrayKey, Array
from gunpowder.graph import Graph

from scipy.ndimage import maximum_filter
from skimage.morphology import skeletonize_3d as skeletonize
import numpy as np
import networkx as nx

import time
import random
import math
import logging

logger = logging.getLogger(__name__)


class SinSample(BatchFilter):
    """
    Get local maxima allong a skeleton via sin wave local maxima

        Args:

            array (:class:``ArrayKey``):

                The binary skeletonized array.

            maxima (:class:``ArrayKey``):

                The array to store the maxima in.

            sample_distance (:class:``Coordinate``):

                skeletonizing will produce a tree. That tree will then be downsampled
                to have points approx sample_dist appart. Note branch points in the
                skeletonization are guaranteed to be contained, so edge lengths may be
                shorter, but should never be longer than the provided length.
    """

    def __init__(
        self,
        skeleton,
        candidates,
        sample_distance: float,
        deterministic_sins: bool = False,
    ):
        self.skeleton = skeleton
        self.candidates = candidates
        self.sample_distance = sample_distance
        self.deterministic_sins = deterministic_sins

    def setup(self):
        self.enable_autoskip()
        provided_spec = self.spec[self.skeleton].copy()
        self.provides(self.candidates, provided_spec)

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        deps[self.skeleton] = request[self.candidates].copy()

        return deps

    def process(self, batch, request):

        skeleton = batch[self.skeleton].data

        spec = batch[self.skeleton].spec.copy()
        spec.dtype = bool

        skeleton_array = Array(skeleton, spec)

        skeleton_array.data = self.candidates_via_sins(skeleton_array)
        candidates = skeleton_array

        batch.arrays[self.candidates] = candidates

    def candidates_via_sins(self, array):

        voxel_shape = array.spec.roi.get_shape() / array.spec.voxel_size
        voxel_size = array.spec.voxel_size
        sphere_radius = self.sample_distance

        offset = array.spec.roi.get_offset()

        if self.deterministic_sins:
            shifts = tuple(0 for _ in range(len(voxel_shape)))
        else:
            shifts = tuple(
                random.random() * 2 * math.pi for _ in range(len(voxel_shape))
            )

        ys = [
            np.sin(
                np.linspace(
                    shifts[i] + (offset[i] * 2 * math.pi / sphere_radius),
                    shifts[i]
                    + (
                        (offset[i] + voxel_shape[i] * voxel_size[i])
                        * 2
                        * math.pi
                        / sphere_radius
                    ),
                    voxel_shape[i],
                )
            ).reshape(
                tuple(1 if j != i else voxel_shape[i] for j in range(len(voxel_shape)))
            )
            for i in range(len(voxel_shape))
        ]
        x = np.sum(ys)

        weighted_skel = x * array.data

        candidates = np.logical_and(
            np.equal(
                maximum_filter(
                    weighted_skel, size=(3 for _ in voxel_size), mode="nearest"
                ),
                weighted_skel,
            ),
            array.data,
        )

        return candidates


class Skeletonize(BatchFilter):
    """ Skeletonize an intensity array in range 0-1

        Args:

            array (:class:``ArrayKey``):

                The array to calculate skeleton from.

            skeleton (:class:``ArrayKey``):

                The array to store the skeletonization in.

            threshold (float):
            
                The threshold to binarize the array.
        """

    def __init__(
        self, array: ArrayKey, skeletonization: ArrayKey, threshold: float = None
    ):

        self.array = array
        self.skeletonization = skeletonization
        self.threshold = threshold

    def setup(self):
        self.enable_autoskip()
        provided_spec = self.spec[self.array].copy()
        provided_spec.dtype = bool
        self.provides(self.skeletonization, provided_spec)

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        deps[self.array] = request[self.skeletonization].copy()

        return deps

    def process(self, batch, request: BatchRequest):
        outputs = Batch()
        data = batch[self.array].data

        if self.threshold is None:
            data_min = np.min(data)
            data_med = np.median(data)
            threshold = (data_min + data_med) / 2
        else:
            threshold = self.threshold

        thresholded = data > threshold
        skeleton = np.squeeze(thresholded)
        skeleton = skeletonize(skeleton)
        skeleton = skeleton > 0

        spec = batch[self.array].spec.copy()
        spec.dtype = bool

        outputs.arrays[self.skeletonization] = Array(skeleton, spec)
        return outputs
