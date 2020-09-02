from dataclasses import dataclass, field
from enum import Enum

from typing import List, Optional


class Dataset(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


@dataclass
class Roi:
    offset: List[int] = "???"
    shape: List[int] = "???"


@dataclass
class Data:
    voxel_size: List[int] = field(default_factory=lambda: [1000, 300, 300])
    raw_n5: str = "fluorescence-near-consensus.n5"
    train_samples: List[str] = field(
        default_factory=lambda: ["2018-07-02", "2018-12-01"]
    )
    test_samples: List[str] = field(default_factory=lambda: ["2018-08-01"])
    validation_samples: List[str] = field(default_factory=lambda: ["2018-10-01"])
    data_set: Dataset = Dataset.TRAIN
    samples_path: str = "/nrs/funke/mouselight"
    benchmark_data_path: str = "/groups/mousebrainmicro/mousebrainmicro/benchmarking_datasets/Manual-GT"
    transform_template: str = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"
    mongo_db_template: str = "mouselight-{sample}-{source}"
    mongo_url: str = "???"
    matched_source: str = "matched"
    input_shape: List[int] = field(default_factory=lambda: [80, 256, 256])
    output_shape: List[int] = field(default_factory=lambda: [32, 120, 120])

    roi: Roi = Roi(offset=[None, None, None], shape=[None, None, None])
