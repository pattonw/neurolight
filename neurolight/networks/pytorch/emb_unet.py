import torch
import numpy as np

from funlib.learn.torch.models import UNet, ConvPass

# from funlib.learn.torch.losses import ultrametric_loss


class EmbeddingUnet(torch.nn.Module):
    def __init__(self, config):
        super(EmbeddingUnet, self).__init__()

        # CONFIGS
        self.network_name = config["NETWORK_NAME"]
        self.input_shape = tuple(config["INPUT_SHAPE"])
        self.output_shape = tuple(config["OUTPUT_SHAPE"])
        self.embedding_dims = config["EMBEDDING_DIMS"]
        self.num_fmaps = config["NUM_FMAPS_EMBEDDING"]
        self.fmap_inc_factors = config["FMAP_INC_FACTORS_EMBEDDING"]
        self.downsample_factors = config["DOWNSAMPLE_FACTORS"]
        self.kernel_size_up = config["KERNEL_SIZE_UP"]
        self.voxel_size = config["VOXEL_SIZE"]

        # LAYERS
        self.unet = UNet(
            in_channels=2,
            num_fmaps=self.num_fmaps,
            fmap_inc_factors=self.fmap_inc_factors,
            downsample_factors=self.downsample_factors,
            kernel_size_down=self.kernel_size_up,
            kernel_size_up=self.kernel_size_up,
            activation="ReLU",
            fov=self.input_shape,
            num_fmaps_out=self.embedding_dims,
            num_heads=1,
            constant_upsample=True,
        )
        self.conv_layer = ConvPass(
            in_channels=self.embedding_dims,
            out_channels=self.embedding_dims,
            kernel_sizes=[[1 for _ in self.input_shape]],
            activation="Tanh",
        )

    def forward(self, raw):
        embedding_logits = self.unet(raw)
        embedding = self.conv_layer(embedding_logits)
        embedding_norms = embedding.norm(dim=1, keepdim=True)
        embedding_normalized = embedding / embedding_norms
        return embedding_normalized

