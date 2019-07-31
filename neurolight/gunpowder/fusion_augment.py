import numpy as np
from gunpowder import Array, BatchFilter, PointsSpec
from scipy import ndimage

import logging

logger = logging.getLogger(__name__)


class FusionAugment(BatchFilter):
    """Combine foreground of two volumes.
        
        Fusion process details:
        The whole "base" volume is kept un-modified, we simply add the
        "add" volume to it. To avoid changing the noise profile of the background
        and to minimize the impact of our fusion on the true data, we use
        the "add" labels which are assumed to be rough estimates of the signal
        location, covering a relatively large area surrounding the desired signal.
        Denote this smoothed labelling as alpha, then the output fused array is
        simply base + alpha * add.
        We end up with a fused volume with a smooth transitions between true image
        data and fused image data.
        Note: if the true "base" signal exactly overlaps with
        the true "add" signal, there will be excessively bright voxels, thus it
        is important to guarantee that the signals do not exactly overlap. This can
        be achieved by using the "GetNeuronPair" node.
        Note: if the "base" labels overlap with the "add" labels which will often
        occur if you want the true signals to be close, the overlapping areas will
        be given a label of -1 in the fused_labels array.

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

                One of "labels_mask" or "intensities". If "labels_mask" (the default),
                alpha blending is applied to the labels mask of "add" volume. If
                "intensities", raw intensities of "add" volume are used.

            blend_smoothness (``float``, optional):

                Set sigma for gaussian smoothing of labels mask of "add" volume.
    """

    def __init__(
        self,
        raw_base,
        raw_add,
        labels_base,
        labels_add,
        points_base,
        points_add,
        raw_fused,
        labels_fused,
        points_fused,
        blend_mode="labels_mask",
        blend_smoothness=3,
        num_blended_objects=0,
    ):

        self.raw_base = raw_base
        self.raw_add = raw_add
        self.labels_base = labels_base
        self.labels_add = labels_add
        self.points_base = points_base
        self.points_add = points_add
        self.raw_fused = raw_fused
        self.labels_fused = labels_fused
        self.points_fused = points_fused
        self.blend_mode = blend_mode
        self.blend_smoothness = blend_smoothness
        self.num_blended_objects = num_blended_objects

        assert self.blend_mode in ["intensity", "labels_mask"], (
            "Unknown blend mode %s." % self.blend_mode
        )

    def setup(self):

        self.provides(self.raw_fused, self.spec[self.raw_base].copy())
        self.provides(self.labels_fused, self.spec[self.labels_base].copy())
        self.provides(self.points_fused, self.spec[self.points_base].copy())

    def prepare(self, request):
        # add "base" and "add" volume to request
        request[self.raw_base] = request[self.raw_fused].copy()
        request[self.raw_add] = request[self.raw_fused].copy()

        # enlarge roi for labels to be the same size as the raw data for mask generation
        request[self.labels_base] = request[self.raw_fused].copy()
        request[self.labels_add] = request[self.raw_fused].copy()

        # enlarge roi for points to be the same as the raw data
        request[self.points_base] = PointsSpec(roi=request[self.raw_fused].roi)
        request[self.points_add] = PointsSpec(roi=request[self.raw_fused].roi)

    def process(self, batch, request):

        # Get base arrays
        raw_base_array = batch[self.raw_base].data
        labels_base_array = batch[self.labels_base].data

        # Get add arrays
        raw_add_array = batch[self.raw_add].data
        labels_add_array = batch[self.labels_add].data

        # fuse labels
        fused_labels_array = self._relabel(labels_base_array.astype(np.int32))
        next_label_id = np.max(fused_labels_array) + 1

        add_mask = np.zeros_like(fused_labels_array, dtype=bool)
        for label in np.unique(labels_add_array):
            if label == 0:
                continue
            label_mask = labels_add_array == label

            # handle overlap
            overlap = np.logical_and(fused_labels_array, label_mask)
            fused_labels_array[overlap] = -1

            # assign new label
            add_mask[label_mask] = True
            fused_labels_array[label_mask] = next_label_id
            next_label_id += 1

        # fuse raw
        if self.blend_mode == "intensity":

            add_mask = raw_add_array.astype(np.float32) / np.max(raw_add_array)
            raw_fused_array = add_mask * raw_add_array + (1 - add_mask) * raw_base_array

        elif self.blend_mode == "labels_mask":

            soft_mask = np.zeros_like(add_mask, dtype="float32")
            ndimage.gaussian_filter(
                add_mask.astype("float32"),
                sigma=self.blend_smoothness,
                output=soft_mask,
                mode="nearest",
            )
            soft_mask /= np.max(soft_mask)
            soft_mask = np.clip((soft_mask * 2), 0, 1)

            raw_fused_array = np.maximum(soft_mask * raw_add_array, raw_base_array)

        else:
            raise NotImplementedError("Unknown blend mode %s." % self.blend_mode)

        # load specs
        labels_add_spec = batch[self.labels_add].spec.copy()
        labels_fused_spec = request[self.labels_fused].copy()
        raw_base_spec = batch[self.raw_base].spec.copy()
        points_base_spec = batch[self.points_base].spec.copy()

        # return raw and labels for "fused" volume
        # raw_fused_array.astype(raw_base_spec.dtype)
        batch.arrays[self.raw_fused] = Array(data=raw_fused_array, spec=raw_base_spec)
        batch.arrays[self.labels_fused] = Array(
            data=fused_labels_array.astype(labels_fused_spec.dtype),
            spec=labels_add_spec,
        ).crop(labels_fused_spec.roi)

        # fuse points:
        batch.points[self.points_fused] = batch[self.points_base].merge(
            batch[self.points_add]
        )

        return batch

    def _relabel(self, a):

        labels = list(np.unique(a))
        if 0 in labels:
            labels.remove(0)

        old_values = np.asarray(labels, dtype=np.int32)
        new_values = np.arange(1, len(labels) + 1, dtype=np.int32)

        values_map = np.arange(int(a.max() + 1), dtype=new_values.dtype)
        values_map[old_values] = new_values

        return values_map[a.copy()]
