from dataclasses import dataclass


@dataclass
class DataProcessing:
    neuron_radius: float = 1.5
    mask_radius: float = 1
    blend_smoothness: float = 3
    clahe: bool = False
    distance_threshold: float = 1e-4
    distance_scale: float = 1
