import logging
import numpy as np

from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class TanhSaturate(BatchFilter):
    """Saturate the values of an array to be floats between 0 and 1 by applying
    1 + 1 * tanh function on negative inputs
    
    Args:
        array (:class:`ArrayKey`):
            The key of the array to modify.
        factor (scalar, optional):
            The factor to divide by before applying the tanh, controls how quickly the values
            saturate to -1, 1.
    """

    def __init__(self, array, scale=1):

        self.array = array
        self.scale = scale

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]

        array.data = 1 + np.tanh(array.data / self.scale)

