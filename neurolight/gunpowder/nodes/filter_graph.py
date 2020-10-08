from gunpowder import BatchFilter, BatchRequest, Batch, GraphKey, Graph

import networkx as nx
import numpy as np

import logging

logger = logging.getLogger(__name__)


class FilterSmallComponents(BatchFilter):
    def __init__(
        self,
        graph: GraphKey,
        size_threshold: float,
        component_attr: str = "component",
        size_attr: str = "component_size",
    ):
        self.graph = graph
        self.size_threshold = size_threshold
        self.component_attr = component_attr
        self.size_attr = size_attr

    def setup(self):
        self.updates(self.graph, self.spec[self.graph].copy())

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.graph] = request[self.graph].copy()
        return deps

    def process(self, batch, request):
        outputs = Batch()

        g = batch[self.graph].to_nx_graph()

        for node, attrs in list(g.nodes.items()):
            if attrs[self.size_attr] < self.size_threshold:
                g.remove_node(node)

        outputs[self.graph] = Graph.from_nx_graph(g, batch[self.graph].spec.copy())
        return outputs


class FilterSmallBranches(BatchFilter):
    def __init__(self, graph: GraphKey, node_threshold: float):
        self.graph = graph
        self.node_threshold = node_threshold

    def setup(self):
        self.updates(self.graph, self.spec[self.graph].copy())

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.graph] = request[self.graph].copy()
        return deps

    def process(self, batch, request):
        outputs = Batch()

        g = batch[self.graph].to_nx_graph()

        cc_func = (
            nx.weakly_connected_components
            if g.is_directed()
            else nx.connected_components
        )

        ccs = cc_func(g)
        for cc in list(ccs):
            self.remove_spines_from_leaves(cc, g, cc_func)
        logger.warning(f"g has {len(g.nodes())} nodes post filtering")

        outputs[self.graph] = Graph.from_nx_graph(g, batch[self.graph].spec.copy())
        return outputs

    def remove_spines(self, cc, g, cc_func):
        finished = False
        while not finished:
            finished = True
            g_component = g.subgraph(cc)

            branch_points = [n for n in g_component.nodes if g_component.degree(n) > 2]
            removed = 0
            for i, branch_point in enumerate(branch_points):
                remaining = [n for n in cc if n != branch_point]
                remaining_g = g_component.subgraph(remaining)

                remaining_ccs = list(cc_func(remaining_g))
                for remaining_cc in list(remaining_ccs):
                    if (
                        self.cable_len(
                            g,
                            list(remaining_cc) + [branch_point],
                            limit=self.node_threshold,
                        )
                        <= self.node_threshold
                    ):
                        for n in remaining_cc:
                            g.remove_node(n)
                            finished = False
                            removed += 1

    def remove_spines_from_leaves(self, cc, g, cc_func):
        finished = False
        g_component = g.subgraph(cc)
        leaves_to_keep = set()

        while not finished:
            finished = True
            leaves = set([n for n in g_component.nodes if g_component.degree(n) == 1])
            leaves = leaves.difference(leaves_to_keep)

            removed = 0
            for i, leaf in enumerate(leaves):
                dist_to_branch = 0
                to_remove = []

                for u, v in nx.dfs_edges(g_component, leaf):
                    dist_to_branch += np.linalg.norm(
                        g_component.nodes[u]["location"]
                        - g_component.nodes[v]["location"]
                    )
                    to_remove.append(u)
                    # stop or append to to_remove
                    if (
                        dist_to_branch > self.node_threshold
                        or g_component.degree(v) > 2
                    ):
                        break
                if dist_to_branch < self.node_threshold:
                    for node in to_remove:
                        g.remove_node(node)
                        removed += 1
                else:
                    leaves_to_keep.add(leaf)
            finished = removed == 0

    def cable_len(self, g, nodes, limit=float("inf")):
        sub_g = g.subgraph(nodes)
        cable_len = 0
        for u, v in sub_g.edges():
            u_loc = sub_g.nodes[u]["location"]
            v_loc = sub_g.nodes[v]["location"]
            edge_len = np.linalg.norm(u_loc - v_loc)
            cable_len += edge_len
            if cable_len > limit:
                return limit + 1
        return cable_len


class FragmentIntersections(BatchFilter):
    def __init__(self, graph: GraphKey):
        self.graph = graph

    def setup(self):
        self.updates(self.graph, self.spec[self.graph].copy())

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.graph] = request[self.graph].copy()
        return deps

    def process(self, batch, request):
        outputs = Batch()
        g = batch[self.graph].to_nx_graph()

        branch_points = [n for n in g.nodes if g.degree(n) > 2]

        for branch_point in branch_points:
            if g.is_directed():
                successors = list(g.successors(branch_point))
                predecessors = list(g.predecessors(branch_point))
                lowest = min(successors + predecessors)
                for successor in successors:
                    if successor != lowest:
                        g.remove_edge(branch_point, successor)
                for predecessor in predecessors:
                    if predecessor != lowest:
                        g.remove_edge(predecessor, branch_point)
            else:
                neighbors = sorted(list(g.neighbors(branch_point)))
                for neighbor in neighbors[1:]:
                    g.remove_edge(branch_point, neighbor)

        outputs[self.graph] = Graph.from_nx_graph(g, batch[self.graph].spec.copy())
        return outputs
