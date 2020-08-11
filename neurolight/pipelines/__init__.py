from .pipelines import embedding_pipeline, foreground_pipeline

import json
from pathlib import Path

cwd = Path(__file__).parent
config_file = cwd / "default_config.json"
eval_config_file = cwd / "evaluation_config.json"

DEFAULT_CONFIG = json.load(config_file.open("r"))
DEFAULT_EVAL_CONFIG = json.load(eval_config_file.open("r"))

__all__ = ["embedding_pipeline", "foreground_pipeline", "DEFAULT_CONFIG", "DEFAULT_EVAL_CONFIG"]
