import numpy as np
from gunpowder import *
import copy
from scipy import ndimage


class FusionAugment(BatchProvider):
    """Combine foreground of one or more volumes with another using soft mask and convex combination.
        Args:
            fg (:class:``ArrayKey``):

                The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

            bg (:class:``ArrayKey``):

                The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

            gt (:class:``ArrayKey``):

                The labeled array to use as mask to cut out foreground objects.

            smoothness (``float``, optional):

                Set sigma for gaussian smoothing as size of soft mask around foreground object.

            return_intermediate (``bool``, optional):

                If intermediate results should be returned (fg and bg of the two merged volumes + soft mask)
    """

    def __init__(self, fg, bg, gt, smoothness=3, return_intermediate=False):

        self.fg = fg
        self.bg = bg
        self.gt = gt
        self.gt_spec = None
        self.fg_spec = None
        self.smoothness = smoothness

        self.intermediates = None
        if return_intermediate:

            self.intermediates = (
                ArrayKey('A_' + self.fg.identifier),
                ArrayKey('A_' + self.bg.identifier),
                ArrayKey('B_' + self.fg.identifier),
                ArrayKey('B_' + self.bg.identifier),
                ArrayKey('SOFT_MASK')
            )

    def setup(self):

        assert len(
            self.get_upstream_providers()) > 1, "at least two batch provider needs to be added to the FusionAugment"

        common_spec = None

        # advertise outputs only if all upstream providers have them
        for provider in self.get_upstream_providers():

            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for key, spec in provider.spec.items():
                    if key not in common_spec:
                        del common_spec[key]

        for key, spec in common_spec.items():
            self.provides(key, spec)

        if self.intermediates is not None:
            for intermediate in self.intermediates:
                self.provides(intermediate, common_spec[self.fg])

    def provide(self, request):

        batch = Batch()

        # prepare: enlarge gt requested roi to be the same as the raw data
        self.gt_spec = request[self.gt].roi.copy()
        self.fg_spec = request[self.fg].roi.copy()
        request[self.gt].roi = request[self.fg].roi.copy()

        if self.intermediates is not None:
            # copy requested intermediates
            intermediate_request = {}
            for intermediate in self.intermediates:
                intermediate_request[intermediate] = request[intermediate].copy()
                del request[intermediate]

        # take one volume as base to cut the other objects into
        base_request = np.random.choice(self.get_upstream_providers(), replace=False).request_batch(request)
        fg = base_request[self.fg].data
        bg = base_request[self.bg].data
        gt = base_request[self.gt].data

        # get object from another volume
        cutout_request = np.random.choice(self.get_upstream_providers(), replace=False).request_batch(request)
        cutout_fg = cutout_request[self.fg].data
        cutout_bg = cutout_request[self.bg].data
        cutout_gt = cutout_request[self.gt].data

        if self.intermediates is not None:
            batch.arrays[ArrayKey('A_' + self.fg.identifier)] = Array(data=base_request[self.fg].data,
                                                                      spec=base_request[self.fg].spec.copy())
            batch.arrays[ArrayKey('A_' + self.bg.identifier)] = Array(data=base_request[self.bg].data,
                                                                      spec=base_request[self.bg].spec.copy())
            batch.arrays[ArrayKey('B_' + self.fg.identifier)] = Array(data=cutout_request[self.fg].data,
                                                                      spec=cutout_request[self.fg].spec.copy())
            batch.arrays[ArrayKey('B_' + self.bg.identifier)] = Array(data=cutout_request[self.bg].data,
                                                                      spec=cutout_request[self.bg].spec.copy())

        labels, counts = np.unique(cutout_gt, return_counts=True)
        labels = np.delete(labels, 0)
        counts = np.delete(counts, 0)
        label = labels[np.argmax(counts)]
        mask = cutout_gt == label

        # position object
        mask = self._position_object(gt, mask)

        # create soft mask
        soft_mask = np.zeros_like(mask, dtype='float32')
        ndimage.gaussian_filter(mask.astype('float32'), sigma=self.smoothness, output=soft_mask, mode='nearest')
        soft_mask /= np.max(soft_mask)

        if self.intermediates is not None:
            soft_mask_spec = base_request[self.fg].spec.copy()
            soft_mask_spec.dtype = np.float32
            batch.arrays[ArrayKey('SOFT_MASK')] = Array(data=soft_mask, spec=soft_mask_spec)

        # paste object into base using convex combination
        fg = soft_mask * cutout_fg + (1 - soft_mask) * fg
        bg = soft_mask * cutout_bg + (1 - soft_mask) * bg
        gt[mask] = np.max(gt) + 1

        gt = self._relabel(gt.astype(np.int32))

        gt_spec = self.spec[self.gt].copy()
        gt_spec.roi = request[self.gt].roi.copy()

        fg_spec = self.spec[self.fg].copy()
        fg_spec.roi = request[self.fg].roi.copy()

        bg_spec = self.spec[self.bg].copy()
        bg_spec.roi = request[self.bg].roi.copy()

        skeleton = Array(data=gt.astype(gt_spec.dtype), spec=gt_spec)
        skeleton = skeleton.crop(self.gt_spec)

        # return augmented raw and gt volume
        batch.arrays[self.fg] = Array(data=fg.astype(fg_spec.dtype), spec=fg_spec)
        batch.arrays[self.bg] = Array(data=bg.astype(bg_spec.dtype), spec=bg_spec)
        batch.arrays[self.gt] = skeleton

        for points_key in base_request.points:
            batch.points[points_key] = base_request.points[points_key]

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


    def _get_overlap(self, a, b):

        return np.sum(np.logical_and(a > 0, b > 0))

    def _get_distance_to_next_object(self, a, b):

        fg = a > 0
        dt_next_object = np.zeros_like(a, dtype='float32')
        object_mask = b > 0
        dt_next_object[object_mask] = ndimage.distance_transform_edt(np.logical_not(fg))[object_mask]

        return dt_next_object

    def _position_object(self, gt, mask):

        # todo: position object randomly or with certain overlap/min_distance
        overlap = self._get_overlap(gt, mask)
        min_distance, counts = np.unique(self._get_distance_to_next_object(gt, mask), return_counts=True)
        print('overlap: ', overlap, ' min_distance: ', np.min(np.delete(min_distance, 0)))

        return mask
