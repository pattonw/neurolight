from dataclasses import dataclass

from .candidates import Candidates
from .clahe import Clahe
from .data import Data
from .data_gen import DataGen
from .data_processing import DataProcessing
from .eval import Eval
from .fusion import Fusion
from .matching import Matching
from .model import Embedding, Foreground
from .optimizer import Optimizer
from .pipeline import Pipeline
from .precache import PreCache
from .random_location import RandomLocation
from .snapshot import Snapshot
from .training import Training
from .um_loss import UmLoss


@dataclass
class Config:
    candidates: Candidates = Candidates()
    clahe: Clahe = Clahe()
    data: Data = Data()
    data_gen: DataGen = DataGen()
    data_processing: DataProcessing = DataProcessing()
    eval: Eval = Eval()
    fusion: Fusion = Fusion()
    matching: Matching = Matching()
    fg_model: Foreground = Foreground()
    emb_model: Embedding = Embedding()
    optimizer: Optimizer = Optimizer()
    pipeline: Pipeline = Pipeline()
    precache: PreCache = PreCache()
    random_location: RandomLocation = RandomLocation()
    snapshot: Snapshot = Snapshot()
    training: Training = Training()
    um_loss: UmLoss = UmLoss()
