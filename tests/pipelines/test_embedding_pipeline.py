from gunpowder import build, Coordinate, BatchRequest
import pytest

from neurolight.pipelines import DEFAULT_CONFIG, embedding_pipeline
from test_sources import get_test_data_sources


@pytest.mark.slow
@pytest.mark.parametrize("snapshot_every", [0, 1])
@pytest.mark.parametrize("train_embedding", [True, False])
@pytest.mark.parametrize("fusion_pipeline", [True, False])
@pytest.mark.parametrize("blend_mode", ["ADD"])
@pytest.mark.parametrize("aux_task", [True, False])
def test_embedding_pipeline(
    tmpdir, aux_task, blend_mode, fusion_pipeline, train_embedding, snapshot_every
):
    setup_config = DEFAULT_CONFIG
    setup_config.training.profile_every = 1
    setup_config.pipeline.fusion_pipeline = fusion_pipeline
    setup_config.pipeline.train_embedding = train_embedding
    setup_config.snapshot.every = snapshot_every
    setup_config.training.tensorboard_log_dir = str(tmpdir)
    setup_config.snapshot.directory = str(tmpdir)
    setup_config.snapshot.file_name = "test_snapshot"
    setup_config.fusion.blend_mode = blend_mode
    setup_config.emb_model.aux_task.enabled = aux_task
    pipeline, requests = embedding_pipeline(setup_config, get_test_data_sources)
    request = BatchRequest()
    for key, shape in requests:
        request.add(key, shape)
    with build(pipeline):
        pipeline.request_batch(request)
