from gunpowder.points import PointsKey, Point
from gunpowder.graph_points import GraphPoints
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.batch_request import BatchRequest
from gunpowder.roi import Roi
from gunpowder.points_spec import PointsSpec
from gunpowder.batch import Batch
from gunpowder.profiling import Timing

import networkx as nx
import numpy as np
import random
from scipy.spatial.ckdtree import cKDTree

import logging
from collections import deque
from typing import List, Tuple
import math

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SyntheticLightLike(BatchProvider):
    def __init__(
        self,
        points: PointsKey,
        thetas: List[float] = np.array([0.2, 0.3, 0.35]) * math.pi,
        num_nodes: Tuple[int, int] = [100, 500],
        p_die: float = 0.1,
        split_ps: List[float] = [0.95, 0.04, 0.01],
        r: float = 1,
        ma: int = 2,
        dims: int = 3,
        n_obj: int = 1,
    ):
        self.points = points
        self.thetas = thetas
        self.max_split = len(thetas)
        self.num_nodes = num_nodes
        self.p_die = p_die
        self.split_ps = split_ps
        self.r = r
        self.ma = ma
        self.dims = dims
        self.n_obj = n_obj

    def setup(self):
        self.provides(
            self.points, PointsSpec(roi=Roi((None,) * self.dims, (None,) * self.dims))
        )

    def provide(self, request: BatchRequest) -> Batch:
        random.seed(request.random_seed)
        np.random.seed(request.random_seed)

        timing = Timing(self, "provide")
        timing.start()

        batch = Batch()

        roi = request[self.points].roi

        region_shape = roi.get_shape()

        trees = []
        for _ in range(self.n_obj):
            for _ in range(100):
                root = np.random.random(len(region_shape)) * region_shape
                tree = self._grow_tree(
                    root, Roi((0,) * len(region_shape), region_shape)
                )
                if self.num_nodes[0] <= len(tree.nodes) <= self.num_nodes[1]:
                    break
            trees.append(tree)

        trees_graph = nx.disjoint_union_all(trees)

        points = {
            node_id: Point(np.floor(node_attrs["pos"]) + roi.get_begin())
            for node_id, node_attrs in trees_graph.nodes.items()
        }

        batch[self.points] = GraphPoints(points, request[self.points], list(trees_graph.edges))

        timing.stop()
        batch.profiling_stats.add(timing)

        # self._plot_tree(tree)

        return batch

    def _grow_tree(self, root: np.ndarray, roi: Roi) -> nx.DiGraph:
        tree = nx.DiGraph()
        tree.add_node(0, pos=root)
        leaves = deque([0])
        next_id = 1

        while len(leaves) > 0 and len(tree.nodes) < self.num_nodes[1]:
            current_id = leaves.pop()
            current_loc = tree.nodes[current_id]["pos"]

            tail = [current_loc]
            while len(tail) < self.ma and len(list(tree.predecessors(current_id))) == 1:
                preds = list(tree.predecessors(current_id))
                tail.append(tree.nodes[preds[0]]["pos"])

            if len(tail) == 1:
                tail.append(np.random.random(tail[0].shape)-0.5 + tail[0])

            moving_direction = np.array(tail[0] - tail[-1])

            if len(tree.nodes) >= self.num_nodes[0] and random.random() < self.p_die:
                continue
            else:
                num_branches = np.random.choice(
                    range(1, self.max_split + 1), p=self.split_ps
                )
                next_points = self._gen_n(num_branches, current_loc, moving_direction)
                for point in next_points:
                    if not roi.contains(point):
                        continue
                    tree.add_node(next_id, pos=point)
                    tree.add_edge(current_id, next_id)
                    leaves.appendleft(next_id)
                    next_id += 1

        return tree

    def _gen_n(self, n, pos, direction):
        theta = self.thetas[n - 1]
        taken = []
        for i in range(n):
            yield pos + self.r * self._new_dir(direction, theta, taken)

    def _new_dir(self, direction, theta, taken):
        k = 0
        while k < 10000:
            new_direction = np.random.rand(len(direction)) - 0.5
            angle = self._angle_between(direction, new_direction)
            if angle < theta and all(
                [
                    self._angle_between(old, new_direction) > theta / (2 * len(taken))
                    for old in taken
                ]
            ):
                taken.append(new_direction / np.linalg.norm(new_direction))
                return new_direction / np.linalg.norm(taken[-1])
            k += 1
        raise ValueError("no valid direction found")

    def _is_valid_vec(self, v, ref_v, past_vs):
        angle = self._angle_between(v, ref_v)
        radius_fraction = np.linalg.norm(v) / self.r
        return angle < self.theta * radius_fraction and all(
            [self._angle_between(v, p_v) > self.phi for p_v in past_vs]
        )

    def _angle_between(self, a, b):
        return abs(np.math.atan2(np.linalg.det([a, b]), np.dot(a, b)))

    def _plot_tree_matplotlib(self, tree):
        tree = nx.convert_node_labels_to_integers(tree)
        points = np.array([node_attrs["pos"] for node_attrs in tree.nodes.values()])
        edges = np.array(tree.edges)
        plt.plot(
            points[:, 0].flatten()[tuple(edges.T)],
            points[:, 1].flatten()[tuple(edges.T)],
            "y-",
        )
        plt.plot(points[:, 0].flatten(), points[:, 1].flatten(0))
        plt.show()

    def _plot_tree(self, tree):
        import plotly.graph_objects as go

        edge_x = []
        edge_y = []
        for edge in tree.edges():
            x0, y0 = tree.node[edge[0]]["pos"]
            x1, y1 = tree.node[edge[1]]["pos"]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        for node in tree.nodes():
            x, y = tree.node[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(tree.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append("# of connections: " + str(len(adjacencies[1])))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<br>Network graph made with Python",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        fig.show()
