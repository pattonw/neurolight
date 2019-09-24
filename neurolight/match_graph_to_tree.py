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
        # TODO: replace more than ternary nodes with fully connected graphs in
        # self.tree
        # TODO: remember backwards mapping of edges to original tree
        pass

    def __find_possible_matches(self):

        # TODO: replace with proper distance between edges

        self.possible_matches = {}

        for g_u, g_v, graph_e_data in self.graph.edges(data=True):

            graph_e_data['__possible_matches'] = []

            for t_u, t_v, tree_e_data in self.tree.edges(data=True):

                distance = np.linalg.norm(
                    self.graph.nodes[g_u]['location'] -
                    self.tree.nodes[t_u]['location'])

                if distance <= self.match_distance_threshold:
                    graph_e_data['__possible_matches'].append(
                        (t_u, t_v, distance))

    def __create_inidicators(self):

        # one binary indicator per edge in graph and possible match edge in
        # tree

        num_variables = 0

        self.match_indicators = {}
        self.match_indicator_costs = []

        for g_u, g_v, graph_e_data in self.graph.edges(data=True):

            self.match_indicators[(g_u, g_v)] = {}

            for t_u, t_v, distance in graph_e_data['__possible_matches']:

                self.match_indicators[(g_u, g_v)][(t_u, t_v)] = num_variables
                self.match_indicator_costs.append(distance)

                num_variables += 1

            # no-match
            self.match_indicators[(g_u, g_v)][None] = num_variables
            self.match_indicator_costs.append(NO_MATCH_COST)
            num_variables += 1

        # one binary indicator for each possible match configuration around a
        # node in graph

        for g_n in self.graph.nodes():

            match_combinations = [
                self.graph.edges[(n, g_n)]['__possible_matches']
                for n in self.graph.neighbors(g_n)
            ]

            for configuration in itertools.product(match_combinations):
                # TODO: continue


def match_graph_to_tree(
        graph,
        tree,
        match_attribute):

def match_graph_to_tree(graph, tree, match_attribute):
    MATCH_DISTANCE_THRESHOLD = 100
    matcher = GraphToTreeMatcher(graph, tree, MATCH_DISTANCE_THRESHOLD)
    matches = matcher.match()

    for e1, e2 in matches:
        graph.edges[e1][match_attribute] = e2
