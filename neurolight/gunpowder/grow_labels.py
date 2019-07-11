import numpy as np
from gunpowder import BatchFilter, ArrayKey, BatchRequest, Array
from scipy.ndimage.morphology import distance_transform_edt


class GrowLabels(BatchFilter):
    """Expand labels by a given radius.

        Args:

            labels (:class:``ArrayKey``):
                The label array to expand.

            labels_spec (:class:``ArraySpec``, optional):

                The spec of the labels array.

            radius (``float``, optional):

                The radius to expand labels in world units.
        """

    def __init__(self, array: ArrayKey, overlap_value: int = -1, radius: float = 1.0):

        self.array = array
        self.overlap_value = overlap_value
        self.radius = radius

    def setup(self):
        pass

    def prepare(self, request: BatchRequest):
        pass

    def process(self, batch, request: BatchRequest):
        """
        currently doesnt handle multiple different labels, simply works with a mask.
        It would probably be more efficient if we could grow multiple labels at the
        same time, keeping track of which one is closest, and labelling overlap
        """
        labels = batch[self.array].data
        spec = batch[self.array].spec.copy()

        expanded = np.zeros_like(labels)
        for label in np.unique(labels):
            if label == 0:
                continue
            label_mask = labels == label
            dt = distance_transform_edt(
                np.logical_not(label_mask), sampling=spec.voxel_size
            )
            binarized = dt <= self.radius
            overlap = np.logical_and(expanded, binarized)
            expanded[binarized] = label
            expanded[overlap] = self.overlap_value

        expanded = Array(data=expanded, spec=spec)

        batch.arrays[self.array] = expanded
