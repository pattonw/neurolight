import torch

from neurolight.networks.pytorch.nms import NMS


def test_simple():
    raw = torch.Tensor(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
    )
    nms = NMS([2, 2], 0)
    maxima = nms(raw)
    expected_maxima = torch.Tensor(
        [[[[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]]]]
    )
    assert torch.all(torch.eq(maxima, expected_maxima))


def test_threshold():
    raw = torch.Tensor(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
    )
    nms = NMS([2, 2], 10)
    maxima = nms(raw)
    expected_maxima = torch.Tensor(
        [[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1]]]]
    )
    assert torch.all(torch.eq(maxima, expected_maxima))


def test_uniform():
    raw = torch.Tensor([[[[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]]])
    nms = NMS([2, 2], 0)
    maxima = nms(raw)
    expected_maxima = torch.Tensor(
        [[[[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]]]]
    )
    assert torch.all(torch.eq(maxima, expected_maxima))


def test_coprime_window_input_size():
    raw = torch.Tensor(
        [
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                ]
            ]
        ]
    )
    nms = NMS([2, 2], 0)
    maxima = nms(raw)
    expected_maxima = torch.Tensor(
        [[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 1]]]]
    )
    print(maxima)
    assert torch.all(torch.eq(maxima, expected_maxima))
