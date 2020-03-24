from gunpowder import build, Coordinate, BatchRequest

from neurolight.pipelines import DEFAULT_CONFIG, embedding_pipeline
from test_sources import get_test_data_sources

import logging
logging.basicConfig(level=logging.WARNING)

setup_config = DEFAULT_CONFIG
setup_config["FUSION_PIPELINE"] = True
setup_config["TRAIN_EMBEDDING"] = True
setup_config["SNAPSHOT_EVERY"] = 1
setup_config["SNAPSHOT_FILE_NAME"] = "test.hdf"
setup_config["MATCHING_FAILURES_DIR"] = None
setup_config["USE_GUROBI"] = False
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
    batch = pipeline.request_batch(request)
    assert output in batch
    assert raw in batch

