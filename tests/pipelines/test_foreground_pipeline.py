from gunpowder import build, Coordinate, BatchRequest
import pytest

from neurolight.pipelines import DEFAULT_CONFIG, foreground_pipeline
from test_sources import get_test_data_sources


@pytest.mark.slow
@pytest.mark.parametrize("snapshot_every", [0, 1])
@pytest.mark.parametrize("distance_loss", [True, False])
@pytest.mark.parametrize("train_foreground", [True, False])
@pytest.mark.parametrize("fusion_pipeline", [True, False])
def test_foreground_pipeline(
    tmpdir, fusion_pipeline, train_foreground, distance_loss, snapshot_every
):
    setup_config = DEFAULT_CONFIG
    setup_config["FUSION_PIPELINE"] = fusion_pipeline
    setup_config["TRAIN_FOREGROUND"] = train_foreground
    setup_config["SNAPSHOT_EVERY"] = snapshot_every
    setup_config["TENSORBOARD_LOG_DIR"] = tmpdir
    setup_config["SNAPSHOT_DIR"] = tmpdir
    setup_config["SNAPSHOT_FILE_NAME"] = "test_snapshot"
    setup_config["MATCHING_FAILURES_DIR"] = None
    voxel_size = Coordinate(setup_config["VOXEL_SIZE"])
    output_size = Coordinate(setup_config["OUTPUT_SHAPE"]) * voxel_size
    input_size = Coordinate(setup_config["INPUT_SHAPE"]) * voxel_size
    pipeline, raw, output, inputs = foreground_pipeline(setup_config, get_test_data_sources)
    request = BatchRequest()
    request.add(raw, input_size)
    request.add(output, output_size)
    for key in inputs:
        request.add(key, output_size)
    with build(pipeline):
        batch = pipeline.request_batch(request)
        assert output in batch
        assert raw in batch
