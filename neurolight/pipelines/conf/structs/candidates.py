from dataclasses import dataclass, field
from enum import Enum

from typing import List


class CandidateMode(Enum):
    SKEL = 0
    NMS = 1


@dataclass
class Candidates:
    spacing: float = 5000
    threshold: float = 0.1
    nms_window_size: List[int] = field(default_factory=lambda: [3, 3, 3])
    nms_threshold: float = 0.1
    mode: CandidateMode = CandidateMode.SKEL
