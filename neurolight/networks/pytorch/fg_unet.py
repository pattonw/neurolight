import torch
from funlib.learn.torch.models import UNet, ConvPass


class ForegroundUnet(torch.nn.Model):
    def __init__(self, config):
        super(ForegroundUnet, self).__init__()

        # CONFIGS
        self.network_name = config["NETWORK_NAME"]
        self.input_shape = tuple(config["INPUT_SHAPE"])
        self.output_shape = tuple(config["OUTPUT_SHAPE"])
        self.num_fmaps = config["NUM_FMAPS_FOREGROUND"]
        self.fmap_inc_factors = config["FMAP_INC_FACTORS_FOREGROUND"]
        self.downsample_factors = config["DOWNSAMPLE_FACTORS"]
        self.kernel_size_up = config["KERNEL_SIZE_UP"]

        # LAYERS
        self.unet = UNet(
            in_channels=2,
            num_fmaps=self.num_fmaps,
            fmap_inc_factors=self.fmap_inc_factors,
            downsample_factors=self.downsample_factors,
            kernel_size_down=self.kernel_size_up,
            kernel_size_up=self.kernel_size_up,
            activation="ReLU",
            fov=self.input_shape * self.voxel_size,
            voxel_size=self.voxel_size,
            num_fmaps_out=1,
            num_heads=1,
            constant_upsample=True,
        )
        self.conv_layer = ConvPass(
            in_channels=self.embedding_dims,
            out_channels=self.embedding_dims,
            kernel_sizs=[[1, 1, 1]],
            activation="sigmoid",
        )

    def forward(self, raw):
        fg_logits = self.unet(raw)
        fg = self.conv_layer(fg_logits)
        return fg
