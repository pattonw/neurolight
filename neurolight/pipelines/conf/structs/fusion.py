from dataclasses import dataclass
from enum import Enum


class BlendMode(Enum):
    LABEL_MASK = 0
    ADD = 1


@dataclass
class Fusion:
    blend_mode: BlendMode = BlendMode.ADD
    blend_smoothness: float = 3.0
