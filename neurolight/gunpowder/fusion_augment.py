import numpy as np
from gunpowder import *
from scipy import ndimage

import logging

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        raw_base,
        raw_add,
        labels_base,
        labels_add,
        raw_fused,
        labels_fused,
        blend_mode="labels_mask",
        blend_smoothness=3,
        num_blended_objects=0,
    ):

        self.raw_base = raw_base
        self.raw_add = raw_add
        self.labels_base = labels_base
        self.labels_add = labels_add
        self.raw_fused = raw_fused
        self.labels_fused = labels_fused
        self.blend_mode = blend_mode
        self.blend_smoothness = blend_smoothness
        self.num_blended_objects = num_blended_objects

        assert self.blend_mode in ["intensity", "labels_mask"], (
            "Unknown blend mode %s." % self.blend_mode
        )

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

        # Get base arrays
        raw_base_array = batch[self.raw_base].data.copy()
        labels_base_array = batch[self.labels_base].data.copy()

        # Get add arrays
        raw_add_array = batch[self.raw_add].data
        labels_add_array = batch[self.labels_add].data

        # base labels are relabelled from 1 to num_labels + 1
        labels_base_array = self._relabel(labels_base_array.astype(np.int32))
        # boolean mask that keeps track of all masks. Used for finding label overlap
        all_labels_mask = labels_base_array > 0

        # Handle add arrays
        raw_add_array = batch[self.raw_add].data
        labels_add_array = batch[self.labels_add].data
        # mask to store added labels
        add_mask = np.zeros_like(labels_base_array, dtype=bool)
        # keep track of new label ids
        next_label_id = np.max(labels_base_array) + 1

        # get labels from add_array to fuse onto base
        labels_to_fuse = np.unique(labels_add_array)
        if 0 in labels_to_fuse:
            labels_to_fuse = np.delete(labels_to_fuse, 0)
        if 0 < self.num_blended_objects < len(labels_to_fuse):
            # why do we do this? If it is to avoid overcrowding of a volume, wouldn't
            # we need to subsample labels in the base volume as well?
            labels_to_fuse = np.random.choice(labels_to_fuse, self.num_blended_objects)

        # labels corresponds to add_labels
        for label in labels_to_fuse:
            # get add mask for this label
            label_mask = labels_add_array == label
            # get overlap with base and previous add labels
            overlap = np.logical_and(all_labels_mask, label_mask)
            # set add mask to true where necessary
            add_mask[label_mask] = True
            # set label in fused array at mask to next label
            labels_base_array[label_mask] = next_label_id
            # todo: position object randomly or with specified overlap/distance
            labels_base_array[overlap] = 0  # set label 0 for overlapping neurons
            next_label_id += 1

        # fuse raw
        if self.blend_mode == "intensity":

            add_mask = raw_add_array.astype(np.float32) / np.max(raw_add_array)
            raw_fused_array = add_mask * raw_add_array + (1 - add_mask) * raw_base_array

        elif self.blend_mode == "labels_mask":

            # create soft mask
            soft_mask = np.zeros_like(add_mask, dtype="float32")
            ndimage.gaussian_filter(
                add_mask.astype("float32"),
                sigma=self.blend_smoothness,
                output=soft_mask,
                mode="nearest",
            )
            soft_mask /= np.max(soft_mask)
            soft_mask = np.clip((soft_mask * 2), 0, 1)

            raw_fused_array = soft_mask * raw_add_array + raw_base_array

        else:
            raise NotImplementedError("Unknown blend mode %s." % self.blend_mode)

        # load specs
        labels_add_spec = batch[self.labels_add].spec.copy()
        labels_fused_spec = request[self.labels_fused].copy()
        raw_base_spec = batch[self.raw_base].spec.copy()

        # return raw and labels for "fused" volume
        batch.arrays[self.raw_fused] = Array(
            data=raw_fused_array.astype(raw_base_spec.dtype), spec=raw_base_spec
        )
        batch.arrays[self.labels_fused] = Array(
            data=labels_base_array.astype(labels_fused_spec.dtype), spec=labels_add_spec
        ).crop(labels_fused_spec.roi)

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
