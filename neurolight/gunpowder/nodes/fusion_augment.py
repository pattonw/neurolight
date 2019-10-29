import numpy as np
from gunpowder import Array, BatchFilter, BatchRequest, PointsSpec
from scipy import ndimage

import logging
from copy import deepcopy

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
        scale_add_volume=True,
        soft_mask=None,
        masked_base=None,
        masked_add=None,
        mask_maxed=None,
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
        self.scale_add_volume = scale_add_volume
        self.soft_mask = soft_mask
        self.masked_base = masked_base
        self.masked_add = masked_add
        self.mask_maxed = mask_maxed

        assert self.blend_mode in ["intensity", "labels_mask"], (
            "Unknown blend mode %s." % self.blend_mode
        )

    def setup(self):

        self.provides(self.raw_fused, self.spec[self.raw_base].copy())
        self.provides(self.labels_fused, self.spec[self.labels_base].copy())
        self.provides(self.points_fused, self.spec[self.points_base].copy())
        if self.soft_mask is not None:
            self.provides(self.soft_mask, self.spec[self.raw_base].copy())
        if self.masked_base is not None:
            self.provides(self.masked_base, self.spec[self.raw_base].copy())
        if self.masked_add is not None:
            self.provides(self.masked_add, self.spec[self.raw_base].copy())
        if self.mask_maxed is not None:
            self.provides(self.mask_maxed, self.spec[self.raw_base].copy())

    def prepare(self, request):
        # add "base" and "add" volume to request
        deps = BatchRequest()
        deps[self.raw_base] = request[self.raw_fused]
        deps[self.raw_add] = request[self.raw_fused]

        # enlarge roi for labels to be the same size as the raw data for mask generation
        deps[self.labels_base] = request[self.raw_fused]
        deps[self.labels_add] = request[self.raw_fused]

        # make points optional
        if self.points_fused in request:
            deps[self.points_base] = PointsSpec(roi=request[self.raw_fused].roi)
            deps[self.points_add] = PointsSpec(roi=request[self.raw_fused].roi)

        return deps

    def process(self, batch, request):
        raw_base_spec = batch[self.raw_base].spec.copy()

        # Get base arrays
        raw_base_array = batch[self.raw_base].data
        labels_base_array = batch[self.labels_base].data

        # Get add arrays
        raw_add_array = batch[self.raw_add].data
        labels_add_array = batch[self.labels_add].data

        if self.scale_add_volume:
            raw_base_median = np.median(raw_base_array)
            raw_add_median = np.median(raw_add_array)
            diff = raw_base_median - raw_add_median
            raw_add_array = raw_add_array + diff

        # fuse labels
        fused_labels_array = self._relabel(labels_base_array)
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
                sigma=self.blend_smoothness / np.array(raw_base_spec.voxel_size),
                output=soft_mask,
                mode="nearest",
            )
            soft_mask /= np.clip(np.max(soft_mask), 1e-5, float("inf"))
            soft_mask = np.clip((soft_mask * 2), 0, 1)
            if self.soft_mask is not None:
                batch.arrays[self.soft_mask] = Array(soft_mask, spec=raw_base_spec)
            if self.masked_base is not None:
                batch.arrays[self.masked_base] = Array(
                    raw_base_array * (soft_mask > 0.25), spec=raw_base_spec
                )
            if self.masked_add is not None:
                batch.arrays[self.masked_add] = Array(
                    raw_add_array * soft_mask, spec=raw_base_spec
                )
            if self.mask_maxed is not None:
                batch.arrays[self.mask_maxed] = Array(
                    np.maximum(
                        raw_base_array * (soft_mask > 0.25), raw_add_array * soft_mask
                    ),
                    spec=raw_base_spec,
                )

            raw_fused_array = np.maximum(soft_mask * raw_add_array, raw_base_array)

        else:
            raise NotImplementedError("Unknown blend mode %s." % self.blend_mode)

        # load specs
        labels_add_spec = batch[self.labels_add].spec.copy()
        labels_fused_spec = request[self.labels_fused].copy()
        raw_base_spec = batch[self.raw_base].spec.copy()

        # return raw and labels for "fused" volume
        # raw_fused_array.astype(raw_base_spec.dtype)
        batch.arrays[self.raw_fused] = Array(
            data=raw_fused_array.astype(raw_base_spec.dtype), spec=raw_base_spec
        )
        batch.arrays[self.labels_fused] = Array(
            data=fused_labels_array, spec=labels_add_spec
        )

        # fuse points:
        if self.points_fused in request:
            batch.points[self.points_fused] = deepcopy(batch[self.points_base])
            batch.points[self.points_fused].graph.merge(batch[self.points_add].graph)

        return batch

    def _relabel(self, a):
        """
        map n labels from arbitrary values to range 1-n
        """

        labels = list(np.unique(a))
        if 0 in labels:
            labels.remove(0)

        if len(labels) == 0:
            return a.copy()
        old_values = np.asarray(labels)
        new_values = np.arange(1, len(labels) + 1, dtype=old_values.dtype)

        values_map = np.arange(int(a.max() + 1), dtype=new_values.dtype)
        values_map[old_values] = new_values

        return values_map[a.copy()]
