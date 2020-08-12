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
    setup_config.training.profile_every = 1
    setup_config.pipeline.fusion_pipeline = fusion_pipeline
    setup_config.pipeline.train_foreground = train_foreground
    setup_config.snapshot.every = snapshot_every
    setup_config.training.tensorboard_log_dir = str(tmpdir)
    setup_config.snapshot.directory = str(tmpdir)
    setup_config.snapshot.file_name = "test_snapshot"
    pipeline, requests = foreground_pipeline(
        setup_config, get_test_data_sources
    )
    request = BatchRequest()
    for key, shape in requests:
        request.add(key, shape)
    with build(pipeline):
        pipeline.request_batch(request)
