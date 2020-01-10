from .pipelines import embedding_pipeline, foreground_pipeline, data_gen_pipeline

import json
from pathlib import Path

cwd = Path(__file__).parent
config_file = cwd / "default_config.json"

DEFAULT_CONFIG = json.load(config_file.open("r"))

__all__ = ["embedding_pipeline", "foreground_pipeline", "data_gen_pipeline", "DEFAULT_CONFIG"]
