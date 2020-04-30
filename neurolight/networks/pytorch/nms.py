import torch

from typing import List


class NMS(torch.nn.Module):
    """
    A non max suppression module.

    Notes:
    - Given an tie for maxima in a block, it will pick the first seen
    example along each axis.
    - Local maxima may be adjacent due to the blockwise nature of this
    nms.
    """

    def __init__(self, window_size: List[int], threshold: float):
        super(NMS, self).__init__()

        self.window_size = window_size
        self.dims = len(self.window_size)
        self.threshold = threshold

        try:
            self.down_op = {
                1: torch.nn.MaxPool1d,
                2: torch.nn.MaxPool2d,
                3: torch.nn.MaxPool3d,
            }[self.dims]
            self.up_op = {
                1: torch.nn.MaxUnpool1d,
                2: torch.nn.MaxUnpool2d,
                3: torch.nn.MaxUnpool3d,
            }[self.dims]
        except KeyError:
            raise ValueError(f"NMS does not yet support {self.dims} dims!")

        self.down_1 = self.down_op(
            kernel_size=window_size, stride=window_size, return_indices=True
        )
        self.up_1 = self.up_op(kernel_size=window_size, stride=window_size)

    def forward(self, raw):
        raw, original_shape = self.__pad(raw)

        down_1, indices = self.down_1(raw)
        maxima = self.up_1(down_1, indices)
        zeros = torch.zeros(*maxima.shape, device=raw.device, dtype=maxima.dtype)
        thresholded_maxima = torch.where(maxima > self.threshold, maxima, zeros)
        maxima_mask = thresholded_maxima == raw

        return self.__crop(maxima_mask, original_shape)

    def __crop(self, maxima, original_shape):
        return maxima[tuple(map(slice, [0 for x in original_shape], original_shape))]

    def __pad(self, raw):
        original_shape = raw.shape
        padding = self.__calculate_padding(raw.shape[2:], self.window_size)
        raw = torch.nn.functional.pad(raw, padding, mode="constant", value=0)
        return raw, original_shape

    def __calculate_padding(self, input_size, window_size):
        """
        pytorch does not seem to handle "SAME" padding, so it must be manually
        calculated. See https://github.com/pytorch/pytorch/issues/3867
        
        This is an naieve helper function that assumes kernel_size == stride
        and only pads on 1 side.
        """
        pad = (torch.tensor(window_size) - torch.tensor(input_size)) % torch.tensor(
            window_size
        )
        padding = tuple(int(x) for x in pad)[::-1]
        padding = tuple(
            padding[i // 2] if i % 2 == 1 else 0 for i in range(len(padding) * 2)
        )

        return padding
