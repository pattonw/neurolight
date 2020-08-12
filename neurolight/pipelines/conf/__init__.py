from pathlib import Path

from omegaconf import OmegaConf

from .structs import Config


cwd = Path(__file__).parent
configs_directory = cwd / "yamls"

config_files = [
    "candidates",
    "clahe",
    "data",
    "data_gen",
    "data_processing",
    "eval",
    "fusion",
    "matchings",
    "model",
    "optimizer",
    "pipeline",
    "precache",
    "random_location",
    "snapshot",
    "training",
    "um_loss",
]

DEFAULT_CONFIG: Config = OmegaConf.structured(Config())
# DEFAULT_CONFIG = OmegaConf.merge(
#     *(
#         [structured_conf]
#         + [
#             OmegaConf.load((configs_directory / file_name / "default.yaml").open("r"))
#             for file_name in config_files
#         ]
#     )
# )

OmegaConf.set_struct(DEFAULT_CONFIG, True)
DEFAULT_EVAL_CONFIG = DEFAULT_CONFIG
