from test_sources import get_test_data_sources

from gunpowder import build, Coordinate, BatchRequest

from neurolight.pipelines import DEFAULT_CONFIG, embedding_pipeline
from neurolight.networks.pytorch import EmbeddingUnet

import logging
import pickle

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
    setup_config["NUM_FMAPS_EMBEDDING"] = 12
    setup_config["FMAP_INC_FACTORS_EMBEDDING"] = 5
    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    output_size = Coordinate(setup_config["OUTPUT_SHAPE"]) * voxel_size
    input_size = Coordinate(setup_config["INPUT_SHAPE"]) * voxel_size
    pipeline, raw, output, inputs = embedding_pipeline(setup_config, get_test_data_sources)
    request = BatchRequest()
    request.add(raw, input_size)
    request.add(output, output_size)
    for key in inputs:
        request.add(key, output_size)
    with build(pipeline):
        for i in range(1):
            batch = pipeline.request_batch(request)
            assert output in batch
            assert raw in batch
