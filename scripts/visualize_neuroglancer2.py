import daisy
import neuroglancer
import numpy as np
import sys
import itertools
import h5py
import networkx as nx
import json
from pathlib import Path
from scipy.ndimage.filters import maximum_filter

from neurolight.visualizations.neuroglancer2 import add_snapshot

from neurolight.pipelines import DEFAULT_CONFIG

# from funlib.show.neuroglancer import add_layer
import mlpack as mlp

if __name__ == "__main__":

    neuroglancer.set_server_bind_address("0.0.0.0")

    args = sys.argv[1:]
    snapshot_file = args[0]

    voxel_size = [1000, 300, 300]

    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units="nm", scales=voxel_size
    )


    viewer = neuroglancer.Viewer()
    viewer.dimensions = dimensions


    with viewer.txn() as s:
        add_snapshot(
            s,
            Path(snapshot_file),
            volume_paths=[
                "predicted_volumes/fg/unet/setup_00/99500/9",
                "predicted_volumes/fg/unet/setup_02/150000/9",
            ],
            graph_paths=[],
        )

    print(viewer)
    input("Hit ENTER to quit!")
