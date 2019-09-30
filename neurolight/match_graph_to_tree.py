import pylp
import numpy as np

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

        for graph_e, graph_e_data in self.graph.edges.items():
            graph_e = Edge(*graph_e)

            self.match_indicators[graph_e] = {}

            for tree_e, distance in graph_e_data["__possible_matches"]:
                self.match_indicators[graph_e][tree_e] = self.num_variables
                self.match_indicator_costs.append(distance)

                self.num_variables += 1

    def __create_constraints(self):

        self.constraints = pylp.LinearConstraints()

        # pick exactly one of the match_indicators per edge

        for graph_e in self.graph.edges():
            graph_e = Edge(*graph_e)

            constraint = pylp.LinearConstraint()
            for match_indicator in self.match_indicators[graph_e].values():
                constraint.set_coefficient(match_indicator, 1)
            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(1)

            self.constraints.add(constraint)

        # branch nodes in graph have to correspond to branch nodes in tree
        #
        # We model this constraint by requiring that as soon as a graph node
        # has degree > 2, the number of times a label is used on incident edges
        # is at most 1.

        for n in self.graph.nodes():

            k = self.graph.degree(n)

            # can't be a branch node
            if k <= 2:
                continue

            edges = [Edge(*e) for e in self.graph.edges(n)]
            possible_matches = []
            for graph_e in edges:
                for tree_e, distance in self.graph.edges[(graph_e.u, graph_e.v)][
                    "__possible_matches"
                ]:
                    possible_matches += [tree_e]
            possible_matches = set(possible_matches)

            for tree_e in possible_matches:
                # k = degree of node
                # n = number of None edges
                # x = effective degree of node (k - n)
                # y = # of edges labelled tree_e
                #
                # (k - 2)*y + x <= 2k - 2
                # (k - 2)*y + k - n <= 2k -2
                # (k - 2)*y - n <= k - 2

                constraint = pylp.LinearConstraint()

                # (k - 2)*y
                if not tree_e.empty:
                    for graph_e in edges:
                        i_el = self.match_indicators[graph_e].get(tree_e, None)
                        if i_el is None:
                            continue
                        constraint.set_coefficient(i_el, k - 2)

                # + n
                for e in edges:
                    i_e0 = self.match_indicators[e][Edge()]
                    constraint.set_coefficient(i_e0, -1)

                # <= k - 2
                constraint.set_relation(pylp.Relation.LessEqual)
                constraint.set_value(k - 2)

                self.constraints.add(constraint)

    def __create_objective(self):

        self.objective = pylp.LinearObjective(self.num_variables)

        for i, c in enumerate(self.match_indicator_costs):
            self.objective.set_coefficient(i, c)


def match_graph_to_tree(graph, tree, match_attribute, match_distance_threshold):

    matcher = GraphToTreeMatcher(graph, tree, match_distance_threshold)
    matches = matcher.match()

    for e1, e2 in matches:
        graph.edges[(e1.u, e1.v)][match_attribute] = e2
