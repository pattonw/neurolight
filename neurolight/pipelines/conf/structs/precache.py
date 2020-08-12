from dataclasses import dataclass


@dataclass
class PreCache:
    cache_size: int = 1
    num_workers: int = 1
