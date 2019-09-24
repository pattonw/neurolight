import pylp
import numpy as np

import itertools


class GraphToTreeMatcher:

    NO_MATCH_COST = 10e5

    def __init__(self, graph, tree, match_distance_threshold):

        self.graph = graph
        self.tree = tree
        self.match_distance_threshold = match_distance_threshold

        self.__preprocess_tree()
        self.__find_possible_matches()
        self.__create_inidicators()
        self.__create_objective()

    def match(self):
        """Return a list of tuples from ``graph`` edges to ``tree`` edges (or
        ``None``, if no match was found).
        """

        self.__solve()

        # TODO: return matches as lists

    def __preprocess_tree(self):
        for node, degree in list(self.tree.degree()):
            if degree > 3:
                neighbors = list(self.tree.neighbors(node))
                for i, neighbor in enumerate(neighbors):
                    self.tree.add_node(
                        (node, neighbor), location=self.tree.nodes[node]["location"]
                    )
                    self.tree.add_edge(neighbor, (node, neighbor))
                    self.tree.remove_edge(node, neighbor)
                    for j in range(i):
                        self.tree.add_edge((node, neighbor), (node, neighbors[j]))

    def __find_possible_matches(self):

        # TODO: replace with proper distance between edges

        self.possible_matches = {}

        for g_u, g_v, graph_e_data in self.graph.edges(data=True):

            graph_e_data["__possible_matches"] = [(None, self.NO_MATCH_COST)]

            for t_u, t_v, tree_e_data in self.tree.edges(data=True):

                distance = self.__edge_distance((g_u, g_v), (t_u, t_v))

                if distance <= self.match_distance_threshold:
                    graph_e_data["__possible_matches"].append(((t_u, t_v), distance))

    def __edge_distance(self, graph_edge, tree_edge):
        # average the distance of the endpoints of the graph edge to the tree edge
        g_u, g_v = graph_edge
        dist = (
            self.__point_to_edge_dist(g_u, tree_edge)
            + self.__point_to_edge_dist(g_v, tree_edge)
        ) / 2
        return dist

    def __point_to_edge_dist(self, point, edge):
        point_loc = self.graph.nodes[point]["location"]
        edge_0_loc = self.tree.nodes[edge[0]]["location"]
        edge_1_loc = self.tree.nodes[edge[1]]["location"]
        slope = edge_1_loc - edge_0_loc
        edge_mag = np.linalg.norm(slope)
        if np.isclose(edge_mag, 0):
            return np.linalg.norm(edge_0_loc - point_loc)
        frac = np.clip(
            np.dot(point_loc - edge_0_loc, slope) / np.dot(slope, slope), 0, 1
        )
        min_dist = np.linalg.norm(frac * slope + edge_0_loc - point_loc)
        return min_dist

    def __create_inidicators(self):

        # one binary indicator per edge in graph and possible match edge in
        # tree

        num_variables = 0

        self.match_indicators = {}
        self.match_indicator_costs = []

        for graph_e, graph_e_data in self.graph.edges.items():

            self.match_indicators[graph_e] = {}

            for tree_e, distance in graph_e_data["__possible_matches"]:

                self.match_indicators[graph_e][tree_e] = num_variables
                self.match_indicator_costs.append(distance)

                num_variables += 1

        # one binary indicator for each possible pair of edges adjacent
        # to every node

        self.node_indicators = {}
        self.node_indicators_costs = []

        for g_n in self.graph.nodes():

            match_combinations = [
                self.graph.edges[(n, g_n)]["__possible_matches"]
                for n in self.graph.neighbors(g_n)
            ]

            for configuration in itertools.product(*match_combinations):
                self.node_indicators[(g_n, *configuration)] = num_variables
                num_variables += 1


def match_graph_to_tree(graph, tree, match_attribute):
    MATCH_DISTANCE_THRESHOLD = 100
    matcher = GraphToTreeMatcher(graph, tree, MATCH_DISTANCE_THRESHOLD)
    matches = matcher.match()

    for e1, e2 in matches:
        graph.edges[e1][match_attribute] = e2
