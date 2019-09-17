import numpy as np
from gunpowder import BatchFilter, ArrayKey, BatchRequest, Array
from scipy.ndimage.morphology import distance_transform_edt

from typing import List
import logging

logger = logging.getLogger(__name__)


class GrowLabels(BatchFilter):
    """Expand labels by a given radius.

        Args:

            labels (:class:``ArrayKey``):
                The label array to expand.

            labels_spec (:class:``ArraySpec``, optional):

                The spec of the labels array.

            radii (``list``->``float``, optional):

                The radius to expand labels in world units.
        """

    def __init__(
        self,
        array: ArrayKey,
        overlap_value: int = -1,
        radii: List[float] = None,
        radius=None,
    ):

        self.array = array
        self.overlap_value = overlap_value
        if radii is not None:
            self.radii = radii
            if radius is not None:
                logger.debug(
                    "Since both radius and radii are defined, default behavior is to use radii"
                )
        elif radius is not None:
            self.radii = [radius]
        else:
            self.radii = [1]

    def setup(self):
        self.enable_autoskip()

        self.updates(self.array, self.spec[self.array])

    def prepare(self, request: BatchRequest):
        deps = BatchRequest()

        deps[self.array] = request[self.array]

        return deps

    def process(self, batch, request: BatchRequest):
        labels = batch[self.array].data
        spec = batch[self.array].spec.copy()

        expanded = np.zeros_like(labels)
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            if label == 0:
                continue
            label_mask = labels == label
            dt = distance_transform_edt(
                np.logical_not(label_mask), sampling=spec.voxel_size
            )
            binarized = dt <= self.radii[i % len(self.radii)]
            overlap = np.logical_and(expanded, binarized)
            expanded[binarized] = label
            expanded[overlap] = self.overlap_value

        expanded = Array(data=expanded, spec=spec)

        batch.arrays[self.array] = expanded
