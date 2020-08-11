import gunpowder as gp
import numpy as np
import networkx as nx

from neurolight_evaluation.graph_matching.comatch.edges_xy import get_edges_xy
from neurolight_evaluation.graph_metrics import psudo_graph_edit_distance
from comatch import match_components

from collections import deque
import copy
import logging

logger = logging.getLogger(__name__)


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
        roi=None,
        details=None,
        location_attr="location",
        small_component_threshold=10000,
        comatch_threshold=4000,
        edit_distance_node_spacing=5000,
        edge_threshold_attr="distance",
        num_thresholds=5,
        threshold_range=(0, 1),
        connectivity=None,
        output_graph=None,
    ):
        self.mst = mst
        self.gt = gt
        self.output = output
        self.roi = roi
        if self.roi is None:
            self.roi = gp.Roi((None,) * 3, (None,) * 3)
        self.location_attr = location_attr
        self.small_component_threshold = small_component_threshold
        self.comatch_threshold = comatch_threshold
        self.edit_distance_node_spacing = edit_distance_node_spacing
        self.edge_threshold_attr = edge_threshold_attr
        self.num_thresholds = num_thresholds
        self.threshold_range = threshold_range
        self.details = details
        self.connectivity = connectivity
        self.output_graph = output_graph

    def setup(self):
        self.enable_autoskip()
        self.provides(self.output, gp.ArraySpec(nonspatial=True))
        if self.details is not None:
            self.provides(self.details, self.spec[self.mst].copy())
        if self.output_graph is not None:
            self.provides(self.output_graph, self.spec[self.mst].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.mst] = gp.GraphSpec(roi=self.roi)
        deps[self.gt] = gp.GraphSpec(roi=self.roi)
        if self.connectivity is not None:
            deps[self.connectivity] = gp.GraphSpec(roi=self.roi)
        return deps

    def process(self, batch, request):
        num_thresholds = self.num_thresholds
        threshold_range = self.threshold_range

        outputs = gp.Batch()

        gt_graph = batch[self.gt].to_nx_graph().to_undirected()
        mst_graph = batch[self.mst].to_nx_graph().to_undirected()
        if self.connectivity is not None:
            connectivity_graph = batch[self.connectivity].to_nx_graph().to_undirected()

        # assert mst_graph.number_of_nodes() > 0, f"mst_graph is empty!"

        if self.details is not None:
            matching_details_graph = nx.Graph()
            if mst_graph.number_of_nodes() == 0:
                node_offset = max([node for node in mst_graph.nodes]+[-1]) + 1
                label_offset = len(list(nx.connected_components(mst_graph))) + 1

                for node, attrs in mst_graph.nodes.items():
                    matching_details_graph.add_node(node, **copy.deepcopy(attrs))
                for edge, attrs in mst_graph.edges.items():
                    matching_details_graph.add_edge(
                        edge[0], edge[1], **copy.deepcopy(attrs)
                    )
                for node, attrs in gt_graph.nodes.items():
                    matching_details_graph.add_node(
                        node + node_offset, **copy.deepcopy(attrs)
                    )
                    matching_details_graph.nodes[node + node_offset]["id"] = (
                        node + node_offset
                    )
                for edge, attrs in gt_graph.edges.items():
                    matching_details_graph.add_edge(
                    edge[0] + node_offset, edge[1] + node_offset, **copy.deepcopy(attrs)
                    )

        edges = [
            (edge, attrs[self.edge_threshold_attr])
            for edge, attrs in mst_graph.edges.items()
        ]
        edges = list(sorted(edges, key=lambda x: x[1]))
        edge_lens = [e[1] for e in edges]
        # min_threshold = edges[0][1]
        if len(edge_lens) > 0:
            min_threshold = edge_lens[int(len(edge_lens) * threshold_range[0])]
            max_threshold = edge_lens[int(len(edge_lens) * threshold_range[1]) - 1]
        else:
            min_threshold = 0
            max_threshold = 1
        thresholds = np.linspace(min_threshold, max_threshold, num=num_thresholds)

        current_threshold_mst = nx.Graph()

        edge_deque = deque(edges)
        edit_distances = []
        split_costs = []
        merge_costs = []
        false_pos_costs = []
        false_neg_costs = []
        num_nodes = []
        num_edges = []

        best_score = None
        best_graph = None

        for threshold_index, threshold in enumerate(thresholds):
            logger.warning(f"Using threshold: {threshold}")
            while len(edge_deque) > 0 and edge_deque[0][1] <= threshold:
                (u, v), _ = edge_deque.popleft()
                attrs = mst_graph.edges[(u, v)]
                current_threshold_mst.add_edge(u, v)
                current_threshold_mst.add_node(u, **mst_graph.nodes[u])
                current_threshold_mst.add_node(v, **mst_graph.nodes[v])

            if self.connectivity is not None:
                temp = nx.Graph()
                next_node = max([node for node in connectivity_graph.nodes]) + 1
                for i, cc in enumerate(nx.connected_components(current_threshold_mst)):
                    component_subgraph = current_threshold_mst.subgraph(cc)
                    for node in component_subgraph.nodes:
                        temp.add_node(node, **dict(connectivity_graph.nodes[node]))
                        temp.nodes[node]["component"] = i
                    for edge in connectivity_graph.edges:
                        if (
                            edge[0] in temp.nodes
                            and edge[1] in temp.nodes
                            and temp.nodes[edge[0]]["component"]
                            == temp.nodes[edge[1]]["component"]
                        ):
                            temp.add_edge(
                                edge[0], edge[1], **dict(connectivity_graph.edges[edge])
                            )
                        elif False:
                            path = nx.shortest_path(
                                connectivity_graph, edge[0], edge[1]
                            )
                            cloned_path = []
                            for node in path:
                                if node in temp.nodes:
                                    cloned_path.append(node)
                                else:
                                    cloned_path.append(next_node)
                                    next_node += 1
                            path_len = len(cloned_path) - 1
                            for i, j in zip(range(path_len), range(1, path_len + 1)):
                                u = cloned_path[i]
                                if u not in temp.nodes:
                                    temp.add_node(
                                        u, **dict(connectivity_graph.nodes[path[i]])
                                    )
                                v = cloned_path[j]
                                if v not in temp.nodes:
                                    temp.add_node(
                                        v, **dict(connectivity_graph.nodes[path[j]])
                                    )

                                temp.add_edge(
                                    u,
                                    v,
                                    **dict(connectivity_graph.edges[path[i], path[j]]),
                                )

            else:
                temp = copy.deepcopy(current_threshold_mst)
                for i, cc in enumerate(nx.connected_components(temp)):
                    for node in cc:
                        attrs = temp.nodes[node]
                        attrs["component"] = i

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
                node: attrs["component"] for node, attrs in temp.nodes.items()
            }

            node_labels_y = {
                node: component
                for component, nodes in enumerate(nx.connected_components(gt_graph))
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
                nodes_x,
                nodes_y,
                edges_xy,
                copy.deepcopy(node_labels_x),
                copy.deepcopy(node_labels_y),
            )

            if self.details is not None:
                # add a match details graph to the batch
                # details is a graph containing nodes from both mst and gt
                # to access details of a node, use `details.nodes[node]["details"]`
                # where the details returned are a numpy array of shape (num_thresholds, _).
                # the _ values stored per threshold are success, fp, fn, merge, split, selected, mst, gt

                # NODES
                # success, mst: n matches to only nodes of 1 label, which matches its own label
                # success, gt: n matches to only nodes of 1 label, which matches its own label
                # fp: n in mst matches to nothing
                # fn: n in gt matches to nothing
                # merge: n in mst matches to a node with label not matching its own
                # split: n in gt matches to a node with label not matching its own
                # selected: n in mst in thresholded graph

                # EDGES
                # success, mst: both endpoints successful
                # success, gt: both endpoints successful
                # fp: both endpoints fp
                # fn: both endpoints fn
                # merge: e in mst: only one endpoint successful
                # split: e in gt: only one endpoint successful
                # selected: e in mst in thresholded graph
                (label_matches, node_matches, splits, merges, fps, fns) = result
                # create lookup tables:
                x_label_match_lut = {}
                y_label_match_lut = {}
                for a, b in label_matches:
                    x_matches = x_label_match_lut.setdefault(a, set())
                    x_matches.add(b)
                    y_matches = y_label_match_lut.setdefault(b, set())
                    y_matches.add(a)
                x_node_match_lut = {}
                y_node_match_lut = {}
                for a, b in node_matches:
                    x_matches = x_node_match_lut.setdefault(a, set())
                    x_matches.add(b)
                    y_matches = y_node_match_lut.setdefault(b, set())
                    y_matches.add(a)

                for node, attrs in matching_details_graph.nodes.items():
                    gt = int(node >= node_offset)
                    mst = 1 - gt

                    if gt == 1:
                        node = node - node_offset

                    selected = gt or (node in temp.nodes())

                    if selected:
                        success, fp, fn, merge, split, label_pair = self.node_matching_result(
                            node,
                            gt,
                            x_label_match_lut,
                            y_label_match_lut,
                            x_node_match_lut,
                            y_node_match_lut,
                            node_labels_x,
                            node_labels_y,
                        )
                    else:
                        success, fp, fn, merge, split, label_pair = (
                            0,
                            0,
                            0,
                            0,
                            0,
                            (-1, -1),
                        )

                    data = attrs.setdefault(
                        "details", np.zeros((len(thresholds), 7), dtype=bool)
                    )
                    data[threshold_index] = [
                        selected,
                        success,
                        fp,
                        fn,
                        merge,
                        split,
                        gt,
                    ]
                    label_pairs = attrs.setdefault("label_pair", [])
                    label_pairs.append(label_pair)
                    assert len(label_pairs) == threshold_index + 1
                for (u, v), attrs in matching_details_graph.edges.items():
                    (
                        u_selected,
                        u_success,
                        u_fp,
                        u_fn,
                        u_merge,
                        u_split,
                        u_gt,
                    ) = matching_details_graph.nodes[u]["details"][threshold_index]

                    (
                        v_selected,
                        v_success,
                        v_fp,
                        v_fn,
                        v_merge,
                        v_split,
                        v_gt,
                    ) = matching_details_graph.nodes[v]["details"][threshold_index]

                    assert u_gt == v_gt
                    e_gt = u_gt

                    u_label_pair = matching_details_graph.nodes[u]["label_pair"][
                        threshold_index
                    ]
                    v_label_pair = matching_details_graph.nodes[v]["label_pair"][
                        threshold_index
                    ]

                    e_selected = u_selected and v_selected
                    e_success = (
                        e_selected
                        and u_success
                        and v_success
                        and (u_label_pair == v_label_pair)
                    )
                    e_fp = u_fp and v_fp
                    e_fn = u_fn and v_fn
                    e_merge = e_selected and (not e_success) and (not e_fp) and not e_gt
                    e_split = e_selected and (not e_success) and (not e_fn) and e_gt
                    assert not (e_success and e_merge)
                    assert not (e_success and e_split)

                    data = attrs.setdefault(
                        "details", np.zeros((len(thresholds), 7), dtype=bool)
                    )
                    if e_success:
                        label_pairs = attrs.setdefault("label_pair", [])
                        label_pairs.append(u_label_pair)
                        assert len(label_pairs) == threshold_index + 1
                    else:
                        label_pairs = attrs.setdefault("label_pair", [])
                        label_pairs.append((-1, -1))
                        assert len(label_pairs) == threshold_index + 1
                    data[threshold_index] = [
                        e_selected,
                        e_success,
                        e_fp,
                        e_fn,
                        e_merge,
                        e_split,
                        e_gt,
                    ]

            edit_distance, (
                split_cost,
                merge_cost,
                false_pos_cost,
                false_neg_cost,
            ) = psudo_graph_edit_distance(
                result[1],
                node_labels_x,
                node_labels_y,
                temp,
                gt_graph,
                self.location_attr,
                node_spacing=self.edit_distance_node_spacing,
                details=True,
            )

            edit_distances.append(edit_distance)
            split_costs.append(split_cost)
            merge_costs.append(merge_cost)
            false_pos_costs.append(false_pos_cost)
            false_neg_costs.append(false_neg_cost)
            num_nodes.append(len(temp.nodes))
            num_edges.append(len(temp.edges))

            # save the best version:
            if best_score is None:
                best_score = edit_distance
                best_graph = copy.deepcopy(temp)
            elif edit_distance < best_score:
                best_score = edit_distance
                best_graph = copy.deepcopy(temp)

        outputs[self.output] = gp.Array(
            np.array(
                [
                    edit_distances,
                    thresholds,
                    num_nodes,
                    num_edges,
                    split_costs,
                    merge_costs,
                    false_pos_costs,
                    false_neg_costs,
                ]
            ),
            gp.ArraySpec(nonspatial=True),
        )
        if self.output_graph is not None:
            outputs[self.output_graph] = gp.Graph.from_nx_graph(
                best_graph, gp.GraphSpec(roi=batch[self.gt].spec.roi, directed=False)
            )
        if self.details is not None:
            outputs[self.details] = gp.Graph.from_nx_graph(
                matching_details_graph,
                gp.GraphSpec(roi=batch[self.gt].spec.roi, directed=False),
            )
        return outputs

    def node_matching_result(
        self,
        node,
        gt,
        x_label_match_lut,
        y_label_match_lut,
        x_node_match_lut,
        y_node_match_lut,
        node_labels_x,
        node_labels_y,
    ):
        if gt == 0:
            # label of this node
            node_label = node_labels_x[node]
            # all nodes in gt that node matches to
            targets = x_node_match_lut.get(node, set())
            # targets labels
            target_labels = set([node_labels_y[n] for n in targets])

            # all labels to which this nodes label matches
            expected_target_labels = x_label_match_lut[node_label]

            # This node has label "A", all of its matching targets have label "B", and "B"
            # is in the set of labels that match to "A"
            success = int(
                len(target_labels) == 1
                and ((expected_target_labels | target_labels) == expected_target_labels)
            )

            fn = 0
            fp = len(targets) == 0
            merge = len(target_labels) > 1
            split = 0

            assert (
                success or fp or merge
            ), f"targets: {targets}, target_labels: {target_labels}, expected_target_labels: {expected_target_labels}"

            if success:
                label_pair = (node_label, list(target_labels)[0])
            else:
                label_pair = (-1, -1)

        else:
            node_label = node_labels_y[node]
            targets = y_node_match_lut.get(node, set())
            target_labels = set().union(set([node_labels_x[n] for n in targets]))

            expected_target_labels = y_label_match_lut[node_label]

            # This node has label "A", all of its matching targets have label "B", and "B"
            # is in the set of labels that match to "A"
            success = int(
                len(target_labels) == 1
                and ((expected_target_labels | target_labels) == expected_target_labels)
            )

            fn = len(targets) == 0
            fp = 0
            merge = 0
            split = len(target_labels) > 1

            assert (
                success or fn or split
            ), f"targets: {targets}, target_labels: {target_labels}, expected_target_labels: {expected_target_labels}"

            if success:
                label_pair = (list(target_labels)[0], node_label)
            else:
                label_pair = (-1, -1)

        return success, fp, fn, merge, split, label_pair
