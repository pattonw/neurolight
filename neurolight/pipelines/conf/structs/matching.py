from dataclasses import dataclass


@dataclass
class Matching:
    gap_crossing_dist: float = 4.8
    match_distance_threshold: float = 7.6
    distance_threshold: float = 1e-4
    use_gurobi: bool = False
    matching_failures_dir: str = "???"
