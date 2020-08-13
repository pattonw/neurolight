from dataclasses import dataclass, field
from enum import Enum

from typing import List, Optional, Any


class Neighborhood(Enum):
    SMALL = 6
    LARGE = 26


@dataclass
class AuxTask:
    enabled: bool = False
    distance: float = 15000
    neighborhood: Neighborhood = 26


@dataclass
class UNet:
    net_name: str
    directory: str
    setup: str
    num_fmaps: int
    fmap_inc_factors: int
    checkpoint: Optional[int]
    # Must use Any here since type checking lists of lists is apparently not
    # supported in omegaconf version 2.0.0
    downsample_factors: Any = field(
        default_factory=lambda: [[1, 3, 3], [2, 2, 2], [2, 2, 2]]
    )
    kernel_size_up: Any = field(
        default_factory=lambda: [
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3]],
        ]
    )
    activation: str = "ReLU"


@dataclass
class Embedding(UNet):
    net_name: str = "emb_net"
    directory: str = (
        "/groups/mousebrainmicro/home/pattonw/Code/Scripts/"
        "neurolight_experiments/mouselight/06_torch_embedding"
    )
    setup: str = "???"
    checkpoint: Optional[int] = None
    num_fmaps: Optional[int] = 2
    fmap_inc_factors: int = 2
    embedding_dims: int = 3
    normalize_embeddings: bool = True
    aux_task: AuxTask = AuxTask()


@dataclass
class Foreground(UNet):
    net_name: str = "fg_net"
    directory: str = (
        "/groups/mousebrainmicro/home/pattonw/Code/Scripts/"
        "neurolight_experiments/mouselight/05_torch_foreground"
    )
    setup: str = "???"
    checkpoint: Optional[int] = None
    num_fmaps: int = 2
    fmap_inc_factors: int = 2

