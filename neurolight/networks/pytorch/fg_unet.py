import torch
from funlib.learn.torch.models import UNet, ConvPass

from typing import Optional


class ForegroundUnet(torch.nn.Module):
    def __init__(self, config):
        super(ForegroundUnet, self).__init__()

        # CONFIGS
        self.input_shape = tuple(config["INPUT_SHAPE"])
        self.output_shape = tuple(config["OUTPUT_SHAPE"])
        self.voxel_size = tuple(config["VOXEL_SIZE"])
        self.num_fmaps = config["NUM_FMAPS_FOREGROUND"]
        self.fmap_inc_factors = config["FMAP_INC_FACTORS_FOREGROUND"]
        self.downsample_factors = tuple(tuple(x) for x in config["DOWNSAMPLE_FACTORS"])
        self.kernel_size_up = config["KERNEL_SIZE_UP"]
        self.activation = config["ACTIVATION"]

        # LAYERS
        self.unet = UNet(
            in_channels=2,
            num_fmaps=self.num_fmaps,
            fmap_inc_factors=self.fmap_inc_factors,
            downsample_factors=self.downsample_factors,
            kernel_size_down=self.kernel_size_up,
            kernel_size_up=self.kernel_size_up,
            activation=self.activation,
            fov=tuple(a * b for a, b in zip(self.input_shape, self.voxel_size)),
            voxel_size=self.voxel_size,
            num_heads=1,
            constant_upsample=True,
        )
        self.conv_layer = ConvPass(
            in_channels=self.num_fmaps,
            out_channels=1,
            kernel_sizes=[[1, 1, 1]],
            activation="Sigmoid",
        )

    def forward(self, raw):
        fg_logits = self.unet(raw)
        fg = self.conv_layer(fg_logits)
        return fg, fg_logits


class ForegroundDistLoss(torch.nn.Module):
    def __init__(self):
        super(ForegroundDistLoss, self).__init__()

    def forward(self, input, target, weights: Optional[torch.Tensor] = None):
        if weights is None:
            weights = torch.ones(input.shape, device=input.device)
        return torch.sum(weights * (input - target) ** 2) / (1 + torch.sum(weights > 0))


class ForegroundBinLoss(torch.nn.Module):
    def __init__(self):
        super(ForegroundBinLoss, self).__init__()

        self.bce = torch.nn.BCELoss(reduction="none")

    def forward(self, input, target, weights: Optional[torch.Tensor] = None):
        target = target.float()
        if weights is None:
            weights = torch.ones(input.shape, device=input.device)
        return torch.sum(weights * self.bce(input, target)) / (1 + torch.sum(weights > 0))
