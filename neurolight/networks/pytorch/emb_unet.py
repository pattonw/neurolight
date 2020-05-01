import torch
import numpy as np

from funlib.learn.torch.models import UNet, ConvPass
from .nms import NMS

from funlib.learn.torch.losses import ultrametric_loss

from typing import Optional


class EmbeddingUnet(torch.nn.Module):
    def __init__(self, config):
        super(EmbeddingUnet, self).__init__()

        # CONFIGS
        self.input_shape = tuple(config["INPUT_SHAPE"])
        self.output_shape = tuple(config["OUTPUT_SHAPE"])
        self.embedding_dims = config["EMBEDDING_DIMS"]
        self.num_fmaps = config["NUM_FMAPS_EMBEDDING"]
        self.fmap_inc_factors = config["FMAP_INC_FACTORS_EMBEDDING"]
        self.downsample_factors = tuple(tuple(x) for x in config["DOWNSAMPLE_FACTORS"])
        self.kernel_size_up = config["KERNEL_SIZE_UP"]
        self.voxel_size = config["VOXEL_SIZE"]
        self.activation = config["ACTIVATION"]
        self.normalize_embeddings = config["NORMALIZE_EMBEDDINGS"]

        # LAYERS
        self.unet = UNet(
            in_channels=2,
            num_fmaps=self.num_fmaps,
            fmap_inc_factors=self.fmap_inc_factors,
            downsample_factors=self.downsample_factors,
            kernel_size_down=self.kernel_size_up,
            kernel_size_up=self.kernel_size_up,
            activation=self.activation,
            fov=self.input_shape,
            num_heads=1,
            constant_upsample=True,
        )
        self.conv_layer = ConvPass(
            in_channels=self.num_fmaps,
            out_channels=self.embedding_dims,
            kernel_sizes=[[1 for _ in self.input_shape]],
            activation="Tanh",
        )

    def forward(self, raw):
        raw.requires_grad = True
        embedding_logits = self.unet(raw)
        embedding = self.conv_layer(embedding_logits)
        if self.normalize_embeddings:
            embedding_norms = embedding.norm(dim=1, keepdim=True)
            embedding_normalized = embedding / embedding_norms
            embedding = embedding_normalized
        return embedding


class EmbeddingLoss(torch.nn.Module):
    def __init__(self, setup_config):
        super(EmbeddingLoss, self).__init__()
        self.nms_window_size = tuple(
            a // b
            for a, b in zip(setup_config["NMS_WINDOW_SIZE"], setup_config["VOXEL_SIZE"])
        )
        self.nms_threshold = setup_config["NMS_THRESHOLD"]
        self.nms = NMS(self.nms_window_size, self.nms_threshold)

        self.add_coordinates = setup_config["ADD_COORDS"]

        # coordinate scale should be defined in microns, but used in pixel space
        voxel_size = np.array(setup_config["VOXEL_SIZE"])
        micron_scale = voxel_size[0]
        self.coordinate_scale = tuple(
            voxel_size / micron_scale * setup_config["COORDINATE_SCALE"]
        )

        self.balance = setup_config["BALANCE_UM_LOSS"]
        self.quadrupel_loss = setup_config["QUADRUPLE_LOSS"]
        self.constrained_emst = setup_config["CONSTRAINED"]
        self.alpha = setup_config["ALPHA"]

    def forward(self, input, target, mask):
        # ultrametric loss is computed on cpu, and no gpu accelerated operations
        # use the outputs.
        input = input.cpu()
        target = target.cpu()
        mask = mask.cpu()
        loss, emst, edges_u, edges_v, dist, ratio_pos, ratio_neg = ultrametric_loss(
            embedding=input,
            gt_seg=target,
            mask=mask,
            alpha=self.alpha,
            add_coordinates=self.add_coordinates,
            coordinate_scale=self.coordinate_scale,
            balance=self.balance,
            quadrupel_loss=self.quadrupel_loss,
            constrained_emst=self.constrained_emst,
        )
        return loss, emst, edges_u, edges_v, dist, ratio_pos, ratio_neg
