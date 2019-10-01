import pylp
import numpy as np
import networkx as nx

import itertools
import logging
from typing import Hashable, Tuple, List, Iterable, Set, Dict

logger = logging.getLogger(__name__)

Edge = Tuple[Hashable, Hashable]


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

    def match(self) -> List[Tuple[Edge, Edge]]:
        """Return a list of tuples from ``graph`` edges to ``tree`` edges (or
        ``None``, if no match was found).
        """

        solution = self.__solve()

        matches = []
        for graph_e in self.graph.edges():
            for l, i in self.match_indicators[graph_e].items():
                if solution[i] > 0.5:
                    matches.append((graph_e, l))

        logger.debug(
            f"Found optimal solution with score: {self._score_solution(solution)}"
        )
        for graph_e, tree_e in matches:
            logger.debug(
                f"edge {graph_e} assigned to {tree_e} with score: "
                + f"{self.match_indicator_costs[self.match_indicators[graph_e][tree_e]]}"
            )

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

    def _score_solution(self, solution):
        total = 0
        for i, cost in enumerate(self.match_indicator_costs):
            total += cost * solution[i]
        return total

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

                pms_in = self.possible_matches.setdefault(e_in, set())
                pms_out = self.possible_matches.setdefault(e_out, set())

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

                        pms_in.add(tree_e)
                        pms_out.add(tree_e)

    def __cost(self, distance: float) -> float:
        """
        The cost formula given some distance between two edges.
        Negative costs indicate rewards
        """
        return -self.match_distance_threshold / max(self.epsilon, distance)

    def __edge_distance(self, graph_e: Edge, tree_e: Edge) -> float:
        # average the distance of the endpoints of the graph edge to the tree edge
        g_u, g_v = graph_e[0], graph_e[1]
        dist = (
            self.__point_to_edge_dist(g_u, tree_e)
            + self.__point_to_edge_dist(g_v, tree_e)
        ) / 2
        return dist

    def __point_to_edge_dist(self, point: Hashable, edge: Edge):
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

    def __tree_candidates(self, graph_edges: Iterable[Edge]) -> Set[Edge]:
        """
        Returns all tree edges that can be assigned to at least one of the graph edges.
        """
        edge_view = self.graph.edges()
        return set(
            [
                tree_edge
                for graph_e in graph_edges
                for tree_edge, _ in edge_view[graph_e]["__possible_matches"]
            ]
        )

    def __can_match(self, graph_e: Edge, tree_e: Edge) -> bool:
        return tree_e in self.possible_matches[graph_e]

    def __all_match(self, graph_es: Iterable[Edge], tree_es: Iterable[Edge]):
        return all([self.__can_match(g_e, t_e) for g_e, t_e in zip(graph_es, tree_es)])

    def __valid_chain(self, u: Edge, v: Edge, match: Edge) -> bool:
        return (
            u != v
            and u != tuple(v[::-1])
            and self.__can_match(u, match)
            and self.__can_match(v, match)
        )

    def __valid_transition(
        self,
        g_ins: Iterable[Edge],
        g_outs: Iterable[Edge],
        t_ins: Iterable[Edge],
        t_outs: Iterable[Edge],
    ) -> bool:
        g_ins, g_outs, t_ins, t_outs = [list(x) for x in (g_ins, g_outs, t_ins, t_outs)]
        return (
            self.__all_match(g_outs, t_outs)  # all out assignments possible
            and self.__all_match(g_ins, t_ins)  # all in assignments possible
            and all(g_in not in g_outs for g_in in g_ins)  # no repeated edges
            and all(tuple(g_in[::-1]) not in g_outs for g_in in g_ins)
        )

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
        self.root_transitions = []

        for graph_n, graph_n_attrs in self.graph.nodes.items():
            """
            Enumerating allowable transitions:
            two cases:
            1) graph_n is part way through an edge
                - there is only 1 valid in_edge and 1 valid out_edge,
                  the remaining edges adjacent to graph_n are assigned to None
                - the assignment for graph_n's in_edge and out_edge are the same
            2) graph_n is at a branch point in tree_n.
                - graph_n in_edges must match to tree_n in_edges.
                  There should only be 1 in_edge.
                - graph_n out_edges must match to tree_n out_edges.
                  There could be any number of out_edges
                - a subset of graph_n in_edges and out_edges must make
                  a 1-1 and onto mapping with tree_n in_edges and out_edges
                - remaining graph_n in_edges and out_edges must be assigned to None
            """
            node_indicators = self.transition_indicators.setdefault(graph_n, {})

            graph_in_edges = self.graph.in_edges(graph_n)
            graph_out_edges = self.graph.out_edges(graph_n)

            tree_in_cands = self.__tree_candidates(graph_in_edges)
            tree_out_cands = self.__tree_candidates(graph_out_edges)

            possible_chains = tree_in_cands & tree_out_cands

            # case 1:
            for possible_chain in possible_chains:
                for g_in, g_out in itertools.product(graph_in_edges, graph_out_edges):
                    if self.__valid_chain(g_in, g_out, possible_chain):
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
                n for e in itertools.chain(tree_in_cands, tree_out_cands) for n in e
            )
            for possible_node in possible_nodes:
                # t_ins and t_outs are fixed for a possible assignment
                t_ins = self.tree.in_edges(possible_node)
                t_outs = self.tree.out_edges(possible_node)
                # get all possible g_ins and g_outs that might satisfy t_ins and t_outs
                g_ins = itertools.permutations(graph_in_edges, len(t_ins))
                g_outs = itertools.permutations(graph_out_edges, len(t_outs))
                for ins, outs in itertools.product(g_ins, g_outs):
                    if self.__valid_transition(g_ins, g_outs, t_ins, t_outs):
                        if self.tree.in_degree(possible_node) == 0:
                            self.root_transitions.append(self.num_variables)
                        config = node_indicators.setdefault(self.num_variables, {})
                        config.update(
                            {g_out: ness_out for g_out, ness_out in zip(outs, t_outs)}
                        )
                        config.update(
                            {g_in: ness_in for g_in, ness_in in zip(ins, t_ins)}
                        )

                        # set the rest to None:
                        for graph_e in itertools.chain(graph_in_edges, graph_out_edges):
                            config.setdefault(graph_e, None)

                        self.num_variables += 1

            # allow all None transitions
            config = node_indicators.setdefault(self.num_variables, {})

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
                            config_constraint.set_coefficient(match_indicator, 1)
                            num_nones += 1

                # if x_node_indicator: on the edge, only satisfied if all assignments = True
                config_constraint.set_coefficient(
                    node_indicator, num_assignments + num_nones
                )
                config_constraint.set_relation(pylp.Relation.LessEqual)
                config_constraint.set_value(num_nones)
                self.constraints.add(config_constraint)

            unique_config_constraint.set_relation(pylp.Relation.Equal)
            unique_config_constraint.set_value(1)
            self.constraints.add(unique_config_constraint)

        # only 1 transition from none to root allowed:
        unique_root_constraint = pylp.LinearConstraint()
        for transition_indicator in self.root_transitions:
            unique_root_constraint.set_coefficient(transition_indicator, 1)
        unique_root_constraint.set_relation(pylp.Relation.LessEqual)
        unique_root_constraint.set_value(1)
        self.constraints.add(unique_root_constraint)

    def __create_objective(self):

        self.objective = pylp.LinearObjective(self.num_variables)

        for i, c in enumerate(self.match_indicator_costs):
            self.objective.set_coefficient(i, c)

    def enforce_expected_assignments(self, expected_assignments: Dict[Edge, Edge]):
        expected_assignment_constraint = pylp.LinearConstraint()
        for c, s in expected_assignments.items():
            if s is None:
                for ns, cost in self.graph.edges[c]["__possible_matches"]:
                    expected_assignment_constraint.set_coefficient(
                        self.match_indicators[c][ns], -1
                    )
            else:
                expected_assignment_constraint.set_coefficient(
                    self.match_indicators[c][s], 1
                )
        expected_assignment_constraint.set_relation(pylp.Relation.Equal)
        expected_assignment_constraint.set_value(8)
        self.constraints.add(expected_assignment_constraint)


def match_graph_to_tree(
    graph: nx.Graph,
    tree: nx.DiGraph,
    match_attribute: str,
    match_distance_threshold: float,
):

    matcher = GraphToTreeMatcher(graph, tree, match_distance_threshold)
    matches = matcher.match()

    for e1, e2 in matches:
        graph.edges[e1][match_attribute] = e2
