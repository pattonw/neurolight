from dataclasses import dataclass


@dataclass
class UmLoss:
    alpha: float = 1.4142
    coordinate_scale: float = 0.01
    add_coords: bool = True
    balance_um_loss: bool = True
    quadruple_loss: bool = False
    constrained: bool = True
