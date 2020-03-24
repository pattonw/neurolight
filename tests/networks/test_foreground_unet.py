import torch

from neurolight.networks.pytorch.fg_unet import ForegroundUnet


def test_fg_unet():
    config = {
        "NETWORK_NAME": "fg_net",
        "INPUT_SHAPE": [80, 256, 256],
        "OUTPUT_SHAPE": [32, 120, 120],
        "NUM_FMAPS_FOREGROUND": 3,
        "FMAP_INC_FACTORS_FOREGROUND": 2,
        "DOWNSAMPLE_FACTORS": [(1, 3, 3), (2, 2, 2), (2, 2, 2)],
        "KERNEL_SIZE_UP": [
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
        ],
        "VOXEL_SIZE": [1, 1, 1],
        "ACTIVATION": "ReLU"
    }
    denoiser = ForegroundUnet(config)

    raw = torch.rand(1, 2, 80, 256, 256)
    foreground, logits = denoiser(raw)
    assert foreground.shape == (1, 1, 32, 120, 120)
    assert foreground.min() > 0
    assert foreground.max() < 1

