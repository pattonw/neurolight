import gunpowder as gp
import numpy as np
import networkx as nx

from neurolight_evaluation.graph_matching.comatch.edges_xy import get_edges_xy
from neurolight_evaluation.graph_metrics import psudo_graph_edit_distance
from comatch import match_components

from collections import deque
import copy


class MergeGraphs(gp.BatchFilter):
    def __init__(self, specs, gt, mst):
        self.specs = specs
        self.gt = gt
        self.mst = mst

    def setup(self):
        self.enable_autoskip()
        all_rois = []
        for block, block_specs in self.specs.items():
            ground_truth = block_specs["ground_truth"]
            mst_pred = block_specs["mst_pred"]

            for key, spec in [ground_truth, mst_pred]:
                current_spec = self.spec[key].copy()
                current_spec.roi = spec.roi
                self.updates(key, current_spec)
                all_rois.append(current_spec.roi)

        self.total_roi = all_rois[0]
        for roi in all_rois[1:]:
            self.total_roi = self.total_roi.union(roi)
        self.provides(
            self.mst, gp.GraphSpec(roi=gp.Roi((None,) * 3, (None,) * 3), directed=False)
        )
        self.provides(
            self.gt, gp.GraphSpec(roi=gp.Roi((None,) * 3, (None,) * 3), directed=False)
        )

    def prepare(self, request):
        deps = gp.BatchRequest()
        for block, block_specs in self.specs.items():
            ground_truth = block_specs["ground_truth"]
            mst_pred = block_specs["mst_pred"]

            for key, spec in [ground_truth, mst_pred]:
                deps[key] = spec

        return deps

    def process(self, batch, request):

        outputs = gp.Batch()

        gt_graph = nx.Graph()
        mst_graph = nx.Graph()

        for block, block_specs in self.specs.items():
            ground_truth_key = block_specs["ground_truth"][0]
            mst_key = block_specs["mst_pred"][0]
            block_gt_graph = batch[ground_truth_key].to_nx_graph().to_undirected()
            block_mst_graph = batch[mst_key].to_nx_graph().to_undirected()
            gt_graph = nx.disjoint_union(gt_graph, block_gt_graph)
            mst_graph = nx.disjoint_union(mst_graph, block_mst_graph)

        for node, attrs in gt_graph.nodes.items():
            attrs["id"] = node
        for node, attrs in mst_graph.nodes.items():
            attrs["id"] = node


        outputs[self.gt] = gp.Graph.from_nx_graph(
            gt_graph, gp.GraphSpec(roi=gp.Roi((None,) * 3, (None,) * 3), directed=False)
        )
        outputs[self.mst] = gp.Graph.from_nx_graph(
            mst_graph,
            gp.GraphSpec(roi=gp.Roi((None,) * 3, (None,) * 3), directed=False),
        )
        return outputs


class Evaluate(gp.BatchFilter):
    def __init__(
        self,
        gt,
        mst,
        output,
        location_attr="location",
        small_component_threshold=5000,
        comatch_threshold=4000,
        edit_distance_node_spacing=5000,
        edge_threshold_attr="distance",
    ):
        self.mst = mst
        self.gt = gt
        self.output = output
        self.location_attr = location_attr
        self.small_component_threshold = small_component_threshold
        self.comatch_threshold = comatch_threshold
        self.edit_distance_node_spacing = edit_distance_node_spacing
        self.edge_threshold_attr = edge_threshold_attr

    def setup(self):
        self.enable_autoskip()
        self.provides(self.output, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.mst] = gp.GraphSpec(roi=gp.Roi((None,) * 3, (None,) * 3))
        deps[self.gt] = gp.GraphSpec(roi=gp.Roi((None,) * 3, (None,) * 3))
        return deps

    def process(self, batch, request):
        num_thresholds = 30

        outputs = gp.Batch()

        gt_graph = batch[self.gt].to_nx_graph().to_undirected()
        mst_graph = batch[self.mst].to_nx_graph().to_undirected()

        edges = [
            (edge, attrs[self.edge_threshold_attr])
            for edge, attrs in mst_graph.edges.items()
        ]
        edges = list(sorted(edges, key=lambda x: x[1]))
        min_threshold = edges[0][1]
        max_threshold = edges[-1][1]
        threshold_range = max_threshold - min_threshold
        min_threshold -= threshold_range / num_thresholds
        max_threshold += threshold_range / num_thresholds
        thresholds = np.linspace(min_threshold, max_threshold, num=num_thresholds)

        current_threshold_mst = nx.Graph()

        edge_deque = deque(edges)
        optimal_edit_distance = float("inf")

        for threshold in thresholds:
            while len(edge_deque) > 0 and edge_deque[0][1] < threshold:
                (u, v), _ = edge_deque.popleft()
                attrs = mst_graph.edges[(u, v)]
                current_threshold_mst.add_edge(u, v)
                current_threshold_mst.add_node(u, **mst_graph.nodes[u])
                current_threshold_mst.add_node(v, **mst_graph.nodes[v])

            temp = copy.deepcopy(current_threshold_mst)

            # remove small connected_components
            false_pos_nodes = []
            for cc in nx.connected_components(temp):
                cc_graph = temp.subgraph(cc)
                min_loc = None
                max_loc = None
                for node, attrs in cc_graph.nodes.items():
                    node_loc = attrs[self.location_attr]
                    if min_loc is None:
                        min_loc = node_loc
                    else:
                        min_loc = np.min(np.array([node_loc, min_loc]), axis=0)
                    if max_loc is None:
                        max_loc = node_loc
                    else:
                        max_loc = np.max(np.array([node_loc, max_loc]), axis=0)
                if np.linalg.norm(min_loc - max_loc) < self.small_component_threshold:
                    false_pos_nodes += list(cc)
            for node in false_pos_nodes:
                temp.remove_node(node)

            nodes_x = list(temp.nodes)
            nodes_y = list(gt_graph.nodes)

            node_labels_x = {
                node: component
                for component, nodes in enumerate(nx.connected_components(temp))
                for node in nodes
            }

            node_labels_y = {
                node: component
                for component, nodes in enumerate(
                    nx.connected_components(gt_graph)
                )
                for node in nodes
            }

            edges_yx = get_edges_xy(
                gt_graph,
                temp,
                location_attr=self.location_attr,
                node_match_threshold=self.comatch_threshold,
            )
            edges_xy = [(v, u) for u, v in edges_yx]

            result = match_components(
                nodes_x, nodes_y, edges_xy, node_labels_x, node_labels_y
            )

            edit_distance = psudo_graph_edit_distance(
                result[1],
                node_labels_x,
                node_labels_y,
                mst_graph,
                gt_graph,
                self.location_attr,
                node_spacing=self.edit_distance_node_spacing,
            )

            optimal_edit_distance = min(edit_distance, optimal_edit_distance)

        outputs[self.output] = gp.Array(
            np.array([optimal_edit_distance]), gp.ArraySpec(nonspatial=True)
        )
        return outputs
