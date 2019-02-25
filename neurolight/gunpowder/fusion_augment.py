import numpy as np
from gunpowder import *
from scipy import ndimage


class FusionAugment(BatchFilter):
    """Combine foreground of one or more volumes with another using soft mask and convex combination.
        Args:
            arrays (``list`` of :class:``ArrayKey``):

                The intensity arrays to modify.

            gt (:class:``ArrayKey``):

                The labeled array to use as mask to cut out foreground objects.

            alpha_blending_ref(``string``, optional):

                Determines if alpha blending should be applied to intensities of volume B or gt mask of it.

            num_blended_obj (``int``):

                The number of objects which should be used from volume B. Use 0 to copy all objects. Can only be \
                applied to alpha_blending_ref gt.

            smoothness (``float``, optional):

                Set sigma for gaussian smoothing as size of soft mask around foreground object.

            return_intermediate (``bool``, optional):

                If intermediate results should be returned (ch1 and ch2 of the two merged volumes (A and B) + soft mask)
    """

    def __init__(self, arrays, gt, alpha_blending_ref='intensity', num_blended_obj=0, smoothness=3,
                 return_intermediate=False):

        if type(arrays) in [list, tuple]:
            self.arrays = list(arrays)
        else:
            self.arrays = [arrays, ]

        self.gt = gt
        self.gt_roi = None
        self.array_roi = None
        self.alpha_blending_ref = alpha_blending_ref
        self.num_blended_obj = num_blended_obj
        self.smoothness = smoothness
        self.intermediates = None

        if return_intermediate:

            self.intermediates = []

            for array in self.arrays:
                self.intermediates += [
                    ArrayKey('A_' + array.identifier),
                    ArrayKey('B_' + array.identifier),
                ]
            if self.alpha_blending_ref == 'gt':
                self.intermediates += [ArrayKey('SOFT_MASK'),]

        assert self.alpha_blending_ref in ['intensity', 'gt'], (
                "Unknown alpha blending reference %s." % self.alpha_blending_ref)

    def setup(self):

        if len(self.arrays) > 1:
            for i in range(1, len(self.arrays)):
                assert(self.spec[self.arrays[0]] == self.spec[self.arrays[i]], (
                    "Specs for %s and %s are not the same. Please check!"
                    %(self.arrays[0].identifier, self.arrays[i].identifier)))

        if self.intermediates is not None:

            spec = self.spec[self.arrays[0]].copy()
            for intermediate in self.intermediates:
                self.provides(intermediate, spec)

    def _handle_intermediates(self, request):

        # remove intermediates from request
        for intermediate in self.intermediates:
            if intermediate in request:
                del request[intermediate]

    def _enlarge_gt(self, request):

        # enlarge roi for gt to be the same size as the raw data for mask generation
        request[self.gt].roi = request[self.arrays[0]].roi.copy()

    def prepare(self, request):

        # save originally requested specs
        self.gt_roi = request[self.gt].roi.copy()
        self.array_roi = request[self.arrays[0]].roi.copy()

        self._enlarge_gt(request)

        if self.intermediates is not None:
            self._handle_intermediates(request)

    def process(self, batch, request):

        array_spec = self.spec[self.arrays[0]].copy()
        array_spec.roi = request[self.arrays[0]].roi.copy()

        # volume A is provided batch
        a_arrays = [batch[array].data for array in self.arrays]
        a_gt = batch[self.gt].data

        # get volume B with additional request
        if self.intermediates is not None:
            self._handle_intermediates(request)

        self._enlarge_gt(request)
        b_batch = self.get_upstream_provider().request_batch(request)
        b_arrays = [b_batch[array].data for array in self.arrays]
        b_gt = b_batch[self.gt].data

        # add original data of A and B to batch
        if self.intermediates is not None:
            for i, array in enumerate(self.arrays):
                batch.arrays[ArrayKey('A_' + array.identifier)] = Array(data=a_arrays[i], spec=array_spec.copy())
                batch.arrays[ArrayKey('B_' + array.identifier)] = Array(data=b_arrays[i], spec=array_spec.copy())

        # create fused gt
        fused_gt = a_gt.copy()
        labels = np.unique(b_gt[b_gt > 0])

        if 0 < self.num_blended_obj < len(labels):
            labels = np.random.sample(labels, self.num_blended_obj)

        fused_gt = self._relabel(fused_gt.astype(np.int32))
        cnt = np.max(fused_gt) + 1
        for label in labels:
            fused_gt[b_gt == label] = cnt
            cnt += 1
        mask = fused_gt > (cnt - len(labels) - 1)

        # create fused raw
        fused = []
        if self.alpha_blending_ref == 'intensity':

            for i in range(len(self.arrays)):

                b_mask = b_arrays[i].astype(np.float32) / np.max(b_arrays[i])
                fused += [b_arrays[i] * b_mask + a_arrays[i] * (1 - b_mask), ]

        elif self.alpha_blending_ref == 'gt':

            # generate gt soft mask
            mask = self._position_object(a_gt, mask)

            # create soft mask
            soft_mask = np.zeros_like(mask, dtype='float32')
            ndimage.gaussian_filter(mask.astype('float32'), sigma=self.smoothness, output=soft_mask, mode='nearest')
            soft_mask /= np.max(soft_mask)
            soft_mask = np.clip((soft_mask * 2), 0, 1)

            if self.intermediates is not None:
                soft_mask_spec = array_spec.copy()
                soft_mask_spec.dtype = np.float32
                batch.arrays[ArrayKey('SOFT_MASK')] = Array(data=soft_mask, spec=soft_mask_spec)

            for i in range(len(self.arrays)):

                fused += [soft_mask * b_arrays[i] + (1 - soft_mask) * a_arrays[i], ]

        else:
            raise NotImplementedError("Unknown alpha blending reference %s." % self.alpha_blending_ref)

        gt_spec = self.spec[self.gt].copy()
        gt_spec.roi = request[self.gt].roi.copy()

        # return fused raw and gt volume
        for i in range(len(self.arrays)):
            batch.arrays[self.arrays[i]] = Array(data=fused[i].astype(array_spec.dtype), spec=array_spec)

        batch.arrays[self.gt] = Array(data=fused_gt.astype(gt_spec.dtype), spec=gt_spec).crop(self.gt_roi)

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
