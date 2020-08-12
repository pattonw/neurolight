from dataclasses import dataclass


@dataclass
class Optimizer:
    radam: bool = False
