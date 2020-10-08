from dataclasses import dataclass, field

from typing import List

from .snapshot import Snapshot
from .clahe import Clahe


@dataclass
class Eval:
    sample: str = "2018-10-01"
    distance_attribute: str = "distance"
    num_thresholds: int = 30
    threshold_range: List[float] = field(default_factory=lambda: [0.2, 1])
    edge_threshold_fg: float = 0.92
    component_threshold_fg: float = 10000
    component_threshold_emb: float = 10000
    device: str = "cuda"
    blocks: List[int] = field(default_factory=lambda: list(range(1, 26)))
    tile_scan: bool = True
    snapshot: Snapshot = Snapshot(
        every=1, directory="eval_results", file_name="{checkpoint}_{block}.zarr"
    )
    clahe: Clahe = Clahe(
        enabled=False,
        pre_fusion=False,
        post_fusion=False,
        clip_limit=0.01,
        normalize=True,
    )
