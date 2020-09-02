from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


@dataclass
class UmLoss:
    alpha: float = 1.4142
    coordinate_scale: float = 0.01
    add_coords: bool = True
    balance_um_loss: float = 0.5
    quadruple_loss: bool = False
    constrained: bool = True
    loss_mode: Mode = Mode.ADDITIVE
