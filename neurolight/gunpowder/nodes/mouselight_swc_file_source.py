from gunpowder.graph_points import GraphPoint

from neurolight.transforms.swc_to_graph import parse_swc
from .swc_file_source import SwcFileSource

import numpy as np
import networkx as nx

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_transform(transform_path: Path):
    with transform_path.open("r") as tf_file:
        text = tf_file.read()
    lines = text.split("\n")
    constants = {}
    for line in lines:
        if len(line) > 0:
            variable, value = line.split(":")
            constants[variable] = float(value)
    spacing = (
        np.array([constants["sx"], constants["sy"], constants["sz"]])
        / 2 ** (constants["nl"] - 1)
        / 1000
    )
    origin = spacing * (
        (np.array([constants["ox"], constants["oy"], constants["oz"]]) // spacing)
        / 1000
    )
    return origin, spacing


class MouselightSwcFileSource(SwcFileSource):
    """An extension of SwcFileSource that uses a transform.txt file to transform coordinates.

    reads points into pixel space based on the transform.txt file
    """

    def __init__(self, *args, **kwargs):
        transform_file = kwargs.pop("transform_file")
        ignore_human_nodes = kwargs.pop("ignore_human_nodes", True)

        super().__init__(*args, **kwargs)
        self.ignore_human_nodes = ignore_human_nodes
        self.transform_file = Path(transform_file)

    def _parse_swc(self, filename: Path):
        """Read one point per line. If ``ndims`` is 0, all values in one line
        are considered as the location of the point. If positive, only the
        first ``ndims`` are used. If negative, all but the last ``-ndims`` are
        used.
        """
        if "cube" in filename.name:
            return 0

        tree = parse_swc(
            filename,
            self.transform_file,
            resolution=[self.scale[i] for i in self.transpose],
            transpose=self.transpose,
        )

        assert len(list(nx.weakly_connected_components(tree))) == 1

        points = {}

        for node, attrs in tree.nodes.items():
            if not self.ignore_human_nodes or attrs["human_placed"]:
                points[node] = GraphPoint(
                    point_type=attrs["point_type"],
                    location=attrs["location"],
                    radius=attrs["radius"],
                )

        human_edges = set()
        if self.ignore_human_nodes:
            for u, v in tree.edges:
                if u not in points or v not in points:
                    human_edges.add((u, v))
        edges = set((u, v) for (u, v) in tree.edges)
        if not self.directed:
            edges = edges | set((v, u) for u, v in tree.edges())
        self._add_points_to_source(points, set(tree.edges) - human_edges)

        return len(list(nx.weakly_connected_components(tree)))
