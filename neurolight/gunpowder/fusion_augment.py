import numpy as np
from gunpowder import *
import copy
from scipy import ndimage


class FusionAugment(BatchProvider): #todo: use Fusion Augment as BatchFilter
    """Combine foreground of one or more volumes with another using soft mask and convex combination.
        Args:
            ch1 (:class:``ArrayKey``):

                The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

            ch2 (:class:``ArrayKey``):

                The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

            gt (:class:``ArrayKey``):

                The labeled array to use as mask to cut out foreground objects.

            smoothness (``float``, optional):

                Set sigma for gaussian smoothing as size of soft mask around foreground object.

            return_intermediate (``bool``, optional):

                If intermediate results should be returned (ch1 and ch2 of the two merged volumes (A and B) + soft mask)
    """

    def __init__(self, ch1, ch2, gt, smoothness=3, return_intermediate=False):

        self.ch1 = ch1
        self.ch2 = ch2
        self.gt = gt
        self.gt_spec = None
        self.ch1_spec = None
        self.smoothness = smoothness

        self.intermediates = None
        if return_intermediate:

            self.intermediates = (
                ArrayKey('A_' + self.ch1.identifier),
                ArrayKey('A_' + self.ch2.identifier),
                ArrayKey('B_' + self.ch1.identifier),
                ArrayKey('B_' + self.ch2.identifier),
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
                self.provides(intermediate, common_spec[self.ch1])

    def provide(self, request):

        batch = Batch()

        # prepare: enlarge gt requested roi to be the same as the raw data
        self.gt_spec = request[self.gt].roi.copy()
        self.ch1_spec = request[self.ch1].roi.copy()
        request[self.gt].roi = request[self.ch1].roi.copy()

        if self.intermediates is not None:
            # copy requested intermediates
            intermediate_request = {}
            for intermediate in self.intermediates:
                intermediate_request[intermediate] = request[intermediate].copy()
                del request[intermediate]

        # get volume A
        a_request = np.random.choice(self.get_upstream_providers(), replace=False).request_batch(request)
        a_ch1 = a_request[self.ch1].data
        a_ch2 = a_request[self.ch2].data
        a_gt = a_request[self.gt].data

        # get object from another volume
        b_request = np.random.choice(self.get_upstream_providers(), replace=False).request_batch(request)
        b_ch1 = b_request[self.ch1].data
        b_ch2 = b_request[self.ch2].data
        b_gt = b_request[self.gt].data

        if self.intermediates is not None:
            batch.arrays[ArrayKey('A_' + self.ch1.identifier)] = Array(
                data=a_request[self.ch1].data, spec=a_request[self.ch1].spec.copy())
            batch.arrays[ArrayKey('A_' + self.ch2.identifier)] = Array(
                data=a_request[self.ch2].data, spec=a_request[self.ch2].spec.copy())
            batch.arrays[ArrayKey('B_' + self.ch1.identifier)] = Array(
                data=b_request[self.ch1].data, spec=b_request[self.ch1].spec.copy())
            batch.arrays[ArrayKey('B_' + self.ch2.identifier)] = Array(
                data=b_request[self.ch2].data, spec=b_request[self.ch2].spec.copy())

        labels, counts = np.unique(b_gt, return_counts=True)
        labels = np.delete(labels, 0)
        counts = np.delete(counts, 0)
        label = labels[np.argmax(counts)]
        mask = b_gt == label

        # position object
        mask = self._position_object(a_gt, mask)

        # create soft mask
        soft_mask = np.zeros_like(mask, dtype='float32')
        ndimage.gaussian_filter(mask.astype('float32'), sigma=self.smoothness, output=soft_mask, mode='nearest')
        soft_mask /= np.max(soft_mask)
        soft_mask = np.clip((soft_mask*1.2), 0, 1)

        if self.intermediates is not None:
            soft_mask_spec = a_request[self.ch1].spec.copy()
            soft_mask_spec.dtype = np.float32
            batch.arrays[ArrayKey('SOFT_MASK')] = Array(data=soft_mask, spec=soft_mask_spec)

        # paste object from b to a using convex combination
        a_ch1 = soft_mask * b_ch1 + (1 - soft_mask) * a_ch1
        a_ch2 = soft_mask * b_ch2 + (1 - soft_mask) * a_ch2
        a_gt[mask] = np.max(a_gt) + 1

        a_gt = self._relabel(a_gt.astype(np.int32))

        gt_spec = self.spec[self.gt].copy()
        gt_spec.roi = request[self.gt].roi.copy()

        ch1_spec = self.spec[self.ch1].copy()
        ch1_spec.roi = request[self.ch1].roi.copy()

        ch2_spec = self.spec[self.ch2].copy()
        ch2_spec.roi = request[self.ch2].roi.copy()

        # return augmented raw and gt volume
        batch.arrays[self.ch1] = Array(data=a_ch1.astype(ch1_spec.dtype), spec=ch1_spec)
        batch.arrays[self.ch2] = Array(data=a_ch2.astype(ch2_spec.dtype), spec=ch2_spec)
        batch.arrays[self.gt] = Array(data=a_gt.astype(gt_spec.dtype), spec=gt_spec).crop(self.gt_spec)

        for points_key in a_request.points:
            batch.points[points_key] = a_request.points[points_key]

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
