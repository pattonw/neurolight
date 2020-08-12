from dataclasses import dataclass


@dataclass
class Training:
    num_iterations: int = 1
    checkpoint_every: int = 10000
    profile_every: int = 1
    tensorboard_log_dir: str = "tensorboard_logs"
