import pylp
import numpy as np
import networkx as nx

import itertools


class GraphToTreeMatcher:

    NO_MATCH_COST = 10e5

    def __init__(
        self,
        graph: nx.Graph,
        tree: nx.DiGraph,
        match_distance_threshold: float,
        epsilon: float = 0.1,
    ):

        self.graph = graph.to_directed()
        self.tree = tree
        assert nx.is_arborescence(self.tree), (
            "cannot match an arbitrary source to an arbitrary target. "
            + "target graph should be a tree. i.e. DAG with max in_degree = 1"
        )

        self.match_distance_threshold = match_distance_threshold
        self.epsilon = epsilon

        self.objective = None
        self.constraints = None

        self.__preprocess_tree()
        self.__find_possible_matches()
        self.__create_inidicators()
        self.__create_constraints()
        self.__create_objective()

    def match(self):
        """Return a list of tuples from ``graph`` edges to ``tree`` edges (or
        ``None``, if no match was found).
        """

        solution = self.__solve()

        matches = []
        for graph_e in self.graph.edges():
            for l, i in self.match_indicators[graph_e].items():
                if solution[i] > 0.5:
                    matches.append((graph_e, l))

        return matches

    def __solve(self):

        solver = pylp.create_linear_solver(pylp.Preference.Gurobi)
        solver.initialize(self.num_variables, pylp.VariableType.Binary)
        solver.set_num_threads(1)
        solver.set_timeout(120)

        solver.set_objective(self.objective)

        consistent = False
        while not consistent:

            solver.set_constraints(self.constraints)
            solution, message = solver.solve()
            consistent = self.__check_consistency(solution)

        return solution

    def __check_consistency(self, solution):

        return True

    def __assign_edges_to_graph(self, solution):
        for graph_e, graph_e_attrs in self.graph.edges().items():
            for possible_match, cost in graph_e_attrs["__possible_matches"]:
                coefficient_ind = self.match_indicators[graph_e][possible_match]
                coefficient = solution[coefficient_ind]
                if coefficient == 1:
                    graph_e_attrs["__assigned_edge"] = possible_match

    def __preprocess_tree(self):
        """
        No special handling of high order branching. consider:
        G:          o               | T:        o
                    |               |           |
                    o               |           |
                    |               |           |
              o-o-o-o-o-o-o-o-o     |   o-------o----------o
                        |           |            \
                        o           |             \
                        |           |              \
                        o           |               o
        Our constraints would be unable to handle this case which is very
        likely if G is generated through some skeletonization algorithm.
        If this is likely to occur in you G, you may need to do some pre
        processing to add potential edges such that it is possible to extract
        a subset of edges of G that are topologically identical to T.
        """
        pass

    def __find_possible_matches(self):

        self.possible_matches = {}

        # iterate over out edges to avoid repeated computations.
        for graph_n in self.graph.nodes():
            for graph_e_out in self.graph.out_edges(graph_n):
                e_out = graph_e_out
                e_in = tuple([graph_e_out[1], graph_e_out[0]])

                for tree_e, tree_e_data in self.tree.edges.items():
                    distance = self.__edge_distance(graph_e_out, tree_e)
                    cost = self.__cost(distance)

                if distance <= self.match_distance_threshold:
                        out_matches = self.graph.edges[e_out].setdefault(
                            "__possible_matches", []
                        )
                        in_matches = self.graph.edges[e_in].setdefault(
                            "__possible_matches", []
                        )

                        out_matches.append((tree_e, cost))
                        in_matches.append((tree_e, cost))

    def __cost(self, distance: float) -> float:
        """
        The cost formula given some distance between two edges.
        Negative costs indicate rewards
        """
        return -self.match_distance_threshold / max(self.epsilon, distance)

    def __edge_distance(self, graph_edge, tree_edge):
        # average the distance of the endpoints of the graph edge to the tree edge
        g_u, g_v = graph_edge[0], graph_edge[1]
        dist = (
            self.__point_to_edge_dist(g_u, tree_edge)
            + self.__point_to_edge_dist(g_v, tree_edge)
        ) / 2
        return dist

    def __point_to_edge_dist(self, point, edge):
        point_loc = self.graph.nodes[point]["location"]
        u_loc = self.tree.nodes[edge[0]]["location"]
        v_loc = self.tree.nodes[edge[1]]["location"]
        slope = v_loc - u_loc
        edge_mag = np.linalg.norm(slope)
        if np.isclose(edge_mag, 0):
            return np.linalg.norm(u_loc - point_loc)
        frac = np.clip(np.dot(point_loc - u_loc, slope) / np.dot(slope, slope), 0, 1)
        min_dist = np.linalg.norm(frac * slope + u_loc - point_loc)
        return min_dist

    def __create_inidicators(self):

        # one binary indicator per edge in graph and possible match edge in
        # tree

        self.num_variables = 0

        self.match_indicators = {}
        self.match_indicator_costs = []

        for graph_e, graph_e_attrs in self.graph.edges.items():

            graph_e_indicators = self.match_indicators.setdefault(graph_e, {})

            for tree_e, distance in graph_e_attrs["__possible_matches"]:
                graph_e_indicators[tree_e] = self.num_variables
                self.match_indicator_costs.append(distance)

                self.num_variables += 1

        self.transition_indicators = {}

        for graph_n, graph_n_attrs in self.graph.nodes.items():
            """
            Enumerating allowable transitions:
            two cases:
            1) graph_n is part way through an edge
                - graph_n inputs == graph_n outputs
                - there is only 1 input and 1 output, the rest are None
            2) graph_n is essentially assigned to some tree_n.
                - graph_n inputs must match to tree_n inputs (1)
                - graph_n outputs must match to tree_n outputs (k)
                - a subset of graph_n inputs and outputs must make
                a 1-1 and onto mapping with tree_n inputs and outputs
                - remaining graph_n inputs and outputs must be none
            """
            node_indicators = self.transition_indicators.setdefault(graph_n, {})

            graph_in_edges = self.graph.in_edges(graph_n)
            graph_out_edges = self.graph.out_edges(graph_n)

            possible_tree_in_edges = set(
                [
                    tree_edge
                    for graph_e in graph_in_edges
                    for tree_edge, cost in self.graph.edges[graph_e][
                        "__possible_matches"
                    ]
                ]
            )

            possible_tree_out_edges = set(
                [
                    tree_edge
                    for graph_e in graph_out_edges
                    for tree_edge, cost in self.graph.edges[graph_e][
                        "__possible_matches"
                    ]
                ]
            )

            possible_chains = possible_tree_in_edges & possible_tree_out_edges

            possible_edge_matches = {
                graph_e: set(
                    e for e, c in self.graph.edges()[graph_e]["__possible_matches"]
                )
                for graph_e in set(graph_in_edges) | set(graph_out_edges)
            }

            # case 1:
            for possible_chain in possible_chains:
                for g_in, g_out in itertools.product(graph_in_edges, graph_out_edges):
                    if (
                        g_in != g_out
                        and g_in != tuple(g_out[::-1])
                        and possible_chain in possible_edge_matches[g_in]
                        and possible_chain in possible_edge_matches[g_out]
                    ):
                        node_indicators[self.num_variables] = {
                            g_in: possible_chain,
                            g_out: possible_chain,
                        }

                        # set the rest to None:
                        for graph_e in itertools.chain(graph_in_edges, graph_out_edges):
                            node_indicators[self.num_variables].setdefault(
                                graph_e, None
                            )

                        self.num_variables += 1

            # case 2:
            possible_nodes = set(
                n
                for e in itertools.chain(
                    possible_tree_in_edges, possible_tree_out_edges
                )
                for n in e
            )
            for possible_node in possible_nodes:
                ness_in_edges = list(self.tree.in_edges(possible_node))
                ness_out_edges = list(self.tree.out_edges(possible_node))
                g_ins = itertools.combinations(graph_in_edges, len(ness_in_edges))
                g_outs = itertools.combinations(graph_out_edges, len(ness_out_edges))
                for ins, outs in itertools.product(g_ins, g_outs):
                    if (
                        all(
                            [
                                ness_out in possible_edge_matches[g_out]
                                for g_out, ness_out in zip(outs, ness_out_edges)
                            ]
                        )
                        and all(
                            [
                                ness_in in possible_edge_matches[g_in]
                                for g_in, ness_in in zip(ins, ness_in_edges)
                            ]
                        )
                        and all(g_in not in outs for g_in in ins)
                        and all(tuple(g_in[::-1]) not in outs for g_in in ins)
                    ):
                        config = node_indicators.setdefault(self.num_variables, {})
                        config.update(
                            {
                                g_out: ness_out
                                for g_out, ness_out in zip(outs, ness_out_edges)
                            }
                        )
                        config.update(
                            {g_in: ness_in for g_in, ness_in in zip(ins, ness_in_edges)}
                        )

                        # set the rest to None:
                        for graph_e in itertools.chain(graph_in_edges, graph_out_edges):
                            config.setdefault(graph_e, None)

                        self.num_variables += 1

    def __create_constraints(self):

        self.constraints = pylp.LinearConstraints()

        # pick exactly one of the match_indicators per edge:

        for graph_e in self.graph.edges():

            constraint = pylp.LinearConstraint()
            for match_indicator in self.match_indicators[graph_e].values():
                constraint.set_coefficient(match_indicator, 1)
                constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(1)

                self.constraints.add(constraint)

        for graph_n in self.graph.nodes():
            # each node has a set of configurations, of which only one can be active
            unique_config_constraint = pylp.LinearConstraint()
            for node_indicator, configuration in self.transition_indicators[
                graph_n
            ].items():
                unique_config_constraint.set_coefficient(node_indicator, 1)

                config_constraint = pylp.LinearConstraint()
                num_assignments = 0
                num_nones = 0
                for graph_e, tree_e in configuration.items():
                    if tree_e is not None:
                        match_indicator = self.match_indicators[graph_e][tree_e]
                        config_constraint.set_coefficient(match_indicator, -1)
                        num_assignments += 1
                for graph_e, tree_e in configuration.items():
                    if tree_e is None:
                        for match_indicator in self.match_indicators[graph_e].values():
                            config_constraint.set_coefficient(
                                match_indicator, num_assignments
                            )
                            num_nones += 1

                # if x_node_indicator: on the edge, only satisfied if all assignments = True
                config_constraint.set_coefficient(
                    node_indicator, num_assignments * (num_nones + 1)
                )
                config_constraint.set_relation(pylp.Relation.LessEqual)
                config_constraint.set_value(num_assignments * num_nones)
                self.constraints.add(config_constraint)

            unique_config_constraint.set_relation(pylp.Relation.Equal)
            unique_config_constraint.set_value(1)
            self.constraints.add(unique_config_constraint)

    def __create_objective(self):

        self.objective = pylp.LinearObjective(self.num_variables)

        for i, c in enumerate(self.match_indicator_costs):
            self.objective.set_coefficient(i, c)


def match_graph_to_tree(graph, tree, match_attribute, match_distance_threshold):

    matcher = GraphToTreeMatcher(graph, tree, match_distance_threshold)
    matches = matcher.match()

    for e1, e2 in matches:
        graph.edges[e1][match_attribute] = e2
