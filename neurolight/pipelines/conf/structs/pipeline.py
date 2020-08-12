from dataclasses import dataclass


@dataclass
class Pipeline:
    train_embedding: bool = False
    train_foreground: bool = False
    fusion_pipeline: bool = True
    distances: bool = True
