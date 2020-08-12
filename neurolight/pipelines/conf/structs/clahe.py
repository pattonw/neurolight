from dataclasses import dataclass, field

from typing import List


@dataclass
class Clahe:
    enabled: bool = True
    pre_fusion: bool = True
    post_fusion: bool = True
    clip_limit: float = 0.01
    normalize: bool = False
    kernel_size: List[int] = field(default_factory=lambda: [20, 64, 64])
