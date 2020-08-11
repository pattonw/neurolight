from test_sources import get_test_data_sources

from gunpowder import build, Coordinate, BatchRequest

from neurolight.pipelines import DEFAULT_CONFIG, embedding_pipeline
from neurolight.networks.pytorch import EmbeddingUnet

import torch

import logging
import pickle
import time

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    setup_config = DEFAULT_CONFIG
    setup_config["FUSION_PIPELINE"] = True
    setup_config["TRAIN_EMBEDDING"] = True
    setup_config["SNAPSHOT_EVERY"] = 0
    setup_config["SNAPSHOT_FILE_NAME"] = None
    setup_config["MATCHING_FAILURES_DIR"] = None
    setup_config["PROFILE_EVERY"] = 1
    setup_config["CLAHE"] = False
    setup_config["NUM_FMAPS_EMBEDDING"] = 20
    setup_config["FMAP_INC_FACTORS_EMBEDDING"] = 5
    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    output_size = Coordinate(setup_config["OUTPUT_SHAPE"]) * voxel_size
    input_size = Coordinate(setup_config["INPUT_SHAPE"]) * voxel_size

    model = EmbeddingUnet(setup_config).cuda()

    raw, _, target = pickle.load(open("profiling_stuff.obj", "rb"))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    raw = torch.from_numpy(raw).cuda()
    target = torch.from_numpy(target).cuda()

    mse_loss = torch.nn.MSELoss()

    for i in range(5):
        t1 = time.time()

        embeddings = model.forward(raw=raw)
        torch.cuda.synchronize()
        t2 = time.time()

        loss = mse_loss(embeddings, target)
        torch.cuda.synchronize()
        t3 = time.time()

        loss.backward()
        torch.cuda.synchronize()
        t4 = time.time()

        print(f"{i}: forward: {t2-t1}")
        print(f"{i}: loss: {t3-t2}")
        print(f"{i}: backward: {t4-t3}")