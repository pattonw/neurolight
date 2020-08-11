from gunpowder import build, Coordinate, BatchRequest
import pytest

from neurolight.pipelines import DEFAULT_CONFIG, embedding_pipeline
from test_sources import get_test_data_sources


@pytest.mark.slow
@pytest.mark.parametrize("snapshot_every", [0])
@pytest.mark.parametrize("train_embedding", [True])
@pytest.mark.parametrize("fusion_pipeline", [True, False])
@pytest.mark.parametrize("blend_mode", ["add"])
@pytest.mark.parametrize("aux_task", [True, False])
def test_embedding_pipeline(
    tmpdir, aux_task, blend_mode, fusion_pipeline, train_embedding, snapshot_every
):
    setup_config = DEFAULT_CONFIG
    setup_config["FUSION_PIPELINE"] = fusion_pipeline
    setup_config["TRAIN_EMBEDDING"] = train_embedding
    setup_config["SNAPSHOT_EVERY"] = snapshot_every
    setup_config["TENSORBOARD_LOG_DIR"] = tmpdir
    setup_config["SNAPSHOT_DIR"] = tmpdir
    setup_config["SNAPSHOT_FILE_NAME"] = "test_snapshot"
    setup_config["MATCHING_FAILURES_DIR"] = None
    setup_config["BLEND_MODE"] = blend_mode
    setup_config["AUX_TASK"] = aux_task
    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    output_size = Coordinate(setup_config["OUTPUT_SHAPE"]) * voxel_size
    input_size = Coordinate(setup_config["INPUT_SHAPE"]) * voxel_size
    pipeline, raw, output, inputs = embedding_pipeline(
        setup_config, get_test_data_sources
    )
    request = BatchRequest()
    request.add(raw, input_size)
    request.add(output, output_size)
    for key in inputs:
        request.add(key, output_size)
    with build(pipeline):
        batch = pipeline.request_batch(request)
        assert output in batch
        assert raw in batch