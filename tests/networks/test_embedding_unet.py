import torch

from neurolight.networks.pytorch.emb_unet import EmbeddingUnet


def test_emb_unet():
    config = {
        "NETWORK_NAME": "emb_net",
        "INPUT_SHAPE": [80, 256, 256],
        "OUTPUT_SHAPE": [32, 120, 120],
        "EMBEDDING_DIMS": 3,
        "NUM_FMAPS_EMBEDDING": 3,
        "FMAP_INC_FACTORS_EMBEDDING": 2,
        "DOWNSAMPLE_FACTORS": [(1, 3, 3), (2, 2, 2), (2, 2, 2)],
        "KERNEL_SIZE_UP": [
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
        ],
        "VOXEL_SIZE": [1, 1, 1],
        "ACTIVATION": "ReLU",
        "NORMALIZE_EMBEDDINGS": True
    }
    embedder = EmbeddingUnet(config)

    raw = torch.rand(1, 2, 80, 256, 256)
    embeddings = embedder(raw)
    assert embeddings.shape == (1, 3, 32, 120, 120)
    assert torch.allclose(
        embeddings.norm(dim=1, keepdim=True), torch.ones(1, 1, 32, 120, 120)
    )

