from dataclasses import dataclass

from typing import Optional, List


@dataclass
class RandomLocation:
    point_balance_radius: float = 70
    min_dist_to_fallback: float = 30000
    max_random_location_points: float = 1000000
    distance_attribute: str = "distance_to_fallback"
    balance_degree_keys: Optional[List[int]] = None
    balance_degree_values: Optional[List[float]] = None
