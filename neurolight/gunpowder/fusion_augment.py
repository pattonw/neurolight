import numpy as np
from gunpowder import *
from scipy import ndimage


class FusionAugment(BatchFilter):
    """Combine foreground of one or more volumes with another using soft mask and convex combination.
        Args:
            raw_base (:class:``ArrayKey``):

                The intensity array for "base" volume.

            raw_add (:class:``ArrayKey``):

                The intensity array for "add" volume.

            labels_base (:class:``ArrayKey``):

                The labeled array for "base" volume.

            labels_add (:class:``ArrayKey``):

                The labeled array for "add" volume.

            raw_fused (:class:``ArrayKey``):

                The intensity array for "fused" volume.

            labels_fused (:class:``ArrayKey``):

                The labeled array for "fused" volume.

            blend_mode(``string``, optional):

                One of "labels_mask" or "intensities". If "labels_mask" (the default), alpha blending is applied
                to the labels mask of "add" volume. If "intensities", raw intensities of "add" volume are used.

            blend_smoothness (``float``, optional):

                Set sigma for gaussian smoothing of labels mask of "add" volume.

            num_blended_objects (``int``):

                The number of objects which should be used from "add" volume to copy it into "base" volume.
                Use 0 to copy all objects. Can only be applied to blend mode "labels_mask".
    """

    def __init__(self, raw_base, raw_add, labels_base, labels_add, raw_fused, labels_fused,
                      blend_mode='labels_mask', blend_smoothness=3, num_blended_objects=0):

        self.raw_base = raw_base
        self.raw_add = raw_add
        self.labels_base = labels_base
        self.labels_add = labels_add
        self.raw_fused = raw_fused
        self.labels_fused = labels_fused
        self.blend_mode = blend_mode
        self.blend_smoothness = blend_smoothness
        self.num_blended_objects = num_blended_objects

        assert self.blend_mode in ['intensity', 'labels_mask'], (
                "Unknown blend mode %s." % self.blend_mode)

    def setup(self):

        self.provides(self.raw_fused, self.spec[self.raw_base].copy())
        self.provides(self.labels_fused, self.spec[self.labels_base].copy())

    def prepare(self, request):

        # add "base" and "add" volume to request
        request[self.raw_base] = request[self.raw_fused].copy()
        request[self.raw_add] = request[self.raw_fused].copy()

        # enlarge roi for labels to be the same size as the raw data for mask generation
        request[self.labels_base] = request[self.raw_fused].copy()
        request[self.labels_add] = request[self.raw_fused].copy()

    def process(self, batch, request):

        # copy "base" volume to "fused"
        raw_fused_array = batch[self.raw_base].data.copy()
        raw_fused_spec = batch[self.raw_base].spec.copy()
        labels_fused_array = batch[self.labels_base].data.copy()
        labels_fused_spec = request[self.labels_fused].copy()

        raw_add_array = batch[self.raw_add].data
        labels_add_array = batch[self.labels_add].data
        labels_add_spec = batch[self.labels_add].spec.copy()

        # fuse labels, create labels_mask of "add" volume
        labels = np.unique(labels_add_array)
        if 0 in labels:
            labels = np.delete(labels, 0)

        if 0 < self.num_blended_objects < len(labels):
            labels = np.random.choice(labels, self.num_blended_objects)

        labels_fused_array = self._relabel(labels_fused_array.astype(np.int32))
        labels_fused_mask = labels_fused_array > 0
        mask = np.zeros_like(labels_fused_array, dtype=bool)
        cnt = np.max(labels_fused_array) + 1

        for label in labels:
            label_mask = labels_add_array == label
            overlap = np.logical_and(labels_fused_mask, label_mask)
            mask[label_mask] = True
            labels_fused_array[label_mask] = cnt    # todo: position object randomly or with specified overlap/distance
            labels_fused_array[overlap] = 0     # set label 0 for overlapping neurons
            cnt += 1

        # fuse raw
        if self.blend_mode == 'intensity':

            add_mask = raw_add_array.astype(np.float32) / np.max(raw_add_array)
            raw_fused_array = add_mask * raw_add_array + (1 - add_mask) * raw_fused_array

        elif self.blend_mode == 'labels_mask':

            # create soft mask
            soft_mask = np.zeros_like(mask, dtype='float32')
            ndimage.gaussian_filter(mask.astype('float32'), sigma=self.blend_smoothness, output=soft_mask,
                                    mode='nearest')
            soft_mask /= np.max(soft_mask)
            soft_mask = np.clip((soft_mask * 2), 0, 1)

            raw_fused_array = soft_mask * raw_add_array + (1 - soft_mask) * raw_fused_array

        else:
            raise NotImplementedError("Unknown blend mode %s." % self.blend_mode)

        # return raw and labels for "fused" volume
        batch.arrays[self.raw_fused] = Array(data=raw_fused_array.astype(raw_fused_spec.dtype), spec=raw_fused_spec)
        batch.arrays[self.labels_fused] = Array(data=labels_fused_array.astype(labels_fused_spec.dtype),
                                                spec=labels_add_spec).crop(labels_fused_spec.roi)

        return batch

    def _relabel(self, a):

        labels = list(np.unique(a))
        if 0 in labels:
            labels.remove(0)

        old_values = np.asarray(labels, dtype=np.int32)
        new_values = np.arange(1, len(labels) + 1, dtype=np.int32)

        values_map = np.arange(int(a.max() + 1), dtype=new_values.dtype)
        values_map[old_values] = new_values

        return values_map[a]
