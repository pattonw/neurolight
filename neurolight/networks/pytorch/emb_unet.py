import torch
import numpy as np

from funlib.learn.torch.models import UNet, ConvPass
from .nms import NMS

from funlib.learn.torch.losses import ultrametric_loss

from typing import Optional
import logging

logger = logging.getLogger(__file__)


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
        self.aux_task = config["AUX_TASK"]
        self.neighborhood = config["AUX_NEIGHBORHOOD"]

        # LAYERS
        # UNET
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
        # FINAL CONV LAYER
        self.conv_layer = ConvPass(
            in_channels=self.num_fmaps,
            out_channels=self.embedding_dims,
            kernel_sizes=[[1 for _ in self.input_shape]],
            activation="Tanh",
        )
        # AUX LAYER
        if self.aux_task:
            self.aux_layer = ConvPass(
                in_channels=self.num_fmaps,
                out_channels=self.neighborhood,
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
            if torch.isnan(embedding_normalized).any():
                logger.warning("ENCOUNTERED NAN IN NORMALIZED EMBEDDINGS!")
                logger.warning(f"raw: {raw}")
                logger.warning(f"nan in raw: {torch.isnan(raw).any()}")
                logger.warning(f"inf in raw: {torch.isinf(raw).any()}")
                logger.warning(f"embedding_logits: {embedding_logits}")
                # logger.warning(f"embeddings: {embedding}")
                # logger.warning(f"norms: {embedding_norms}")
                # logger.warning(f"normalized_embeddings: {embedding_normalized}")
                torch.save({"model_state_dict": self.state_dict()}, "nan_model")
            embedding = embedding_normalized
            logger.info(
                f"Unet output logits with mean: {torch.mean(embedding_logits)} "
                f"and std: {torch.std(embedding_logits)}"
            )

        if self.aux_task:
            neighborhood = self.aux_layer(embedding_logits)
            return embedding, neighborhood
        else:
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

        self.aux_task = setup_config["AUX_TASK"]

        self.mse_loss = torch.nn.MSELoss()

    def forward(
        self,
        input,
        target,
        mask,
        neighborhood=None,
        neighborhood_mask=None,
        neighborhood_target=None,
    ):
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

        if self.aux_task:
            neighborhood_loss = self.mse_loss(
                neighborhood * neighborhood_mask, neighborhood_target
            ).cpu()
            loss = neighborhood_loss * loss
            return (
                loss,
                emst,
                edges_u,
                edges_v,
                dist,
                ratio_pos,
                ratio_neg,
                neighborhood_loss,
            )

        return loss, emst, edges_u, edges_v, dist, ratio_pos, ratio_neg
