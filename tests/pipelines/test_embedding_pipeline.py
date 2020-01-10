from gunpowder import build, Coordinate, BatchRequest
import pytest

from neurolight.pipelines import DEFAULT_CONFIG, embedding_pipeline
from test_sources import get_test_data_sources


@pytest.mark.slow
@pytest.mark.parametrize("snapshot_every", [0, 1])
@pytest.mark.parametrize("train_embedding", [True, False])
@pytest.mark.parametrize("fusion_pipeline", [True, False])
def test_embedding_pipeline(fusion_pipeline, train_embedding, snapshot_every):
    setup_config = DEFAULT_CONFIG
    setup_config["FUSION_PIPELINE"] = fusion_pipeline
    setup_config["TRAIN_EMBEDDING"] = train_embedding
    setup_config["SNAPSHOT_EVERY"] = snapshot_every
    setup_config["SNAPSHOT_FILE_NAME"] = None
    setup_config["MATCHING_FAILURES_DIR"] = None
    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    output_size = Coordinate(setup_config["OUTPUT_SHAPE"]) * voxel_size
    input_size = Coordinate(setup_config["INPUT_SHAPE"]) * voxel_size
    pipeline, raw, output = embedding_pipeline(setup_config, get_test_data_sources)
    request = BatchRequest()
    request.add(raw, input_size)
    request.add(output, output_size)
    with build(pipeline):
        batch = pipeline.request_batch(request)
        assert output in batch
        assert raw in batch

