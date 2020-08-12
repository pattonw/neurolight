from dataclasses import dataclass


@dataclass
class Snapshot:
    every: int = 0
    file_name: str = "snapshot_{iteration}.hdf"
    directory: str = "snapshots"
