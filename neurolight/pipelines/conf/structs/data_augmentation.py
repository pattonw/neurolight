from dataclasses import dataclass

from typing import Union, Tuple

@dataclass
class DataAugmentation:
    noise_mean_mean: float = 0.0
    noise_mean_var: float = 0
    noise_var_mean: float = 0.0
    noise_var_var: float = 0

