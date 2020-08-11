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
        logger.debug(f"g has {len(g.nodes())} nodes pre filtering")

        cc_func = (
            nx.weakly_connected_components
            if g.is_directed()
            else nx.connected_components
        )

        ccs = cc_func(g)
        for cc in list(ccs):
            finished = False
            while not finished:
                finished = True
                g_component = g.subgraph(cc)

                branch_points = [
                    n for n in g_component.nodes if g_component.degree(n) > 2
                ]
                logger.debug(
                    f"Connected component has {len(g_component.nodes)} nodes and {len(branch_points)} branch points"
                )
                removed = 0
                for i, branch_point in enumerate(branch_points):
                    remaining = [n for n in cc if n != branch_point]
                    remaining_g = g_component.subgraph(remaining)

                    remaining_ccs = list(cc_func(remaining_g))
                    logger.debug(
                        f"After removing branch point {i}, cc is broken into pieces sized: {[len(x) for x in remaining_ccs]}"
                    )
                    for remaining_cc in list(remaining_ccs):
                        if (
                            self.cable_len(g, list(remaining_cc) + [branch_point])
                            <= self.node_threshold
                        ):
                            for n in remaining_cc:
                                g.remove_node(n)
                                finished = False
                                removed += 1
                logger.debug(f"Removed {removed} nodes from this cc")
        logger.debug(f"g has {len(g.nodes())} nodes post filtering")

        outputs[self.graph] = Graph.from_nx_graph(g, batch[self.graph].spec.copy())
        return outputs

    def cable_len(self, g, nodes):
        sub_g = g.subgraph(nodes)
        cable_len = 0
        for u, v in sub_g.edges():
            u_loc = sub_g.nodes[u]["location"]
            v_loc = sub_g.nodes[v]["location"]
            edge_len = np.linalg.norm(u_loc - v_loc)
            cable_len += edge_len
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
