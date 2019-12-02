import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import itertools
import unittest
import logging
from pathlib import Path
from typing import List

from funlib.match.graph_to_tree_matcher import match_graph_to_tree, get_matched
from funlib.match.preprocess import mouselight_preprocessing
import os
import sys
import threading
import time


class SuppressStream(object):
    def __init__(self, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, "w")
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()


logger = logging.getLogger(__name__)


def parse_npy_graph(filename):
    a = np.load(filename)
    edge_list = a["edge_list"]
    node_ids = a["node_ids"]
    locations = a["locations"]
    graph = nx.DiGraph()
    graph.add_edges_from(edge_list)
    graph.add_nodes_from(
        [(nid, {"location": l}) for nid, l in zip(node_ids, locations)]
    )
    return graph


def test_examples(edge_len: float, edge_dist: float, skip: List[str]):
    valid_examples = Path("valid")
    invalid_examples = Path("invalid")

    edge_length_sum = 0
    edge_count = 0

    for valid in valid_examples.iterdir():
        if valid in skip:
            continue
        graph = parse_npy_graph(valid / "graph.npz")
        mouselight_preprocessing(graph, min_dist=edge_len)
        tree = parse_npy_graph(valid / "tree.npz")
        failed = False
        with SuppressStream(sys.stdout) as guard:
            try:
                matched = get_matched(graph, tree, "matched", edge_dist)
                for u, v in matched.edges():
                    edge_length_sum += np.linalg.norm(
                        matched.nodes[u]["location"] - matched.nodes[v]["location"]
                    )
                    edge_count += 1

            except Exception:
                failed = True
        if failed:
            return (False, np.nan)

    for invalid in invalid_examples.iterdir():
        graph = parse_npy_graph(invalid / "graph.npz")
        mouselight_preprocessing(graph, min_dist=edge_len)
        tree = parse_npy_graph(invalid / "tree.npz")
        passed = False
        with SuppressStream(sys.stdout) as guard:
            try:
                match_graph_to_tree(graph, tree, "matched", edge_dist)
                passed = True
            except Exception:
                pass
        if passed:
            return (False, np.nan)

    return (True, edge_length_sum / edge_count)


skip = [Path("valid/015"), Path("valid/022"), Path("valid/000"), Path("valid/007")]
preprocessed_edge_len_range = np.arange(40, 62, 2)
max_edge_distance = np.arange(0, 104, 4)

smallest = (4, 19)

avg_edge_lengths = np.zeros([len(preprocessed_edge_len_range), len(max_edge_distance)])
for (i, el), (j, ed) in tqdm(
    list(
        itertools.product(
            enumerate(preprocessed_edge_len_range), enumerate(max_edge_distance)
        )
    )
):
    if i == 4 and j == 19:
        print(el, ed)
        valid, avg_edge_len = test_examples(el, ed, skip)
        avg_edge_lengths[i, j] = avg_edge_len

with sns.axes_style("white"):
    ax = sns.heatmap(avg_edge_lengths)
    plt.show()
