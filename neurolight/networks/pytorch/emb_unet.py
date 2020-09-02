import torch
import numpy as np

from funlib.learn.torch.models import UNet, ConvPass

from funlib.learn.torch.losses import ultrametric_loss

import logging

logger = logging.getLogger(__file__)


class EmbeddingUnet(torch.nn.Module):
    def __init__(self, config):
        super(EmbeddingUnet, self).__init__()

        # CONFIGS
        self.embedding_dims = config.emb_model.embedding_dims
        self.num_fmaps = config.emb_model.num_fmaps
        self.fmap_inc_factors = config.emb_model.fmap_inc_factors
        self.downsample_factors = tuple(
            tuple(x) for x in config.emb_model.downsample_factors
        )
        self.kernel_size_up = config.emb_model.kernel_size_up
        self.voxel_size = config.data.voxel_size
        self.activation = config.emb_model.activation
        self.normalize_embeddings = config.emb_model.normalize_embeddings
        self.aux_task = config.emb_model.aux_task.enabled
        self.neighborhood = config.emb_model.aux_task.neighborhood.value

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
            num_heads=1,
            constant_upsample=True,
        )
        # FINAL CONV LAYER
        self.conv_layer = ConvPass(
            in_channels=self.num_fmaps,
            out_channels=self.embedding_dims,
            kernel_sizes=[[1 for _ in range(len(self.downsample_factors[0]))]],
            activation="Tanh",
        )
        # AUX LAYER
        if self.aux_task:
            self.aux_layer = ConvPass(
                in_channels=self.num_fmaps,
                out_channels=self.neighborhood,
                kernel_sizes=[[1 for _ in range(len(self.downsample_factors[0]))]],
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

        self.add_coordinates = setup_config.um_loss.add_coords

        # coordinate scale should be defined in microns, but used in pixel space
        voxel_size = np.array(setup_config.data.voxel_size)
        micron_scale = voxel_size[0]
        self.coordinate_scale = tuple(
            voxel_size / micron_scale * setup_config.um_loss.coordinate_scale
        )

        self.balance = setup_config.um_loss.balance_um_loss
        self.quadrupel_loss = setup_config.um_loss.quadruple_loss
        self.constrained_emst = setup_config.um_loss.constrained
        self.alpha = setup_config.um_loss.alpha
        self.loss_mode = setup_config.um_loss.loss_mode.name.lower()

        self.aux_task = setup_config.emb_model.aux_task.enabled

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
            loss_mode=self.loss_mode,
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
