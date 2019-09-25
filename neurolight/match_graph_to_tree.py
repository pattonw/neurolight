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
        self.__create_constraints()
        self.__create_objective()

        self.objective = None
        self.constraints = None

    def match(self):
        """Return a list of tuples from ``graph`` edges to ``tree`` edges (or
        ``None``, if no match was found).
        """

        solution = self.__solve()

        matches = []
        for e in self.graph.edges():
            for l, i in self.match_indicators[e].items():
                if solution[i] > 0.5:
                    matches.append(e, l)

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

        # TODO: check if solution is consistent
        # if not, add additional constraints and return False

        output_graph = self.__solution_to_tree(solution)

        consistent = True

        for edge, edge_attrs in output_graph.edges().items():
            for neighbor_edge, neighbor_edge_attrs in itertools.chain(
                output_graph.edges(edge[0]), output_graph.edges(edge[1])
            ):
                if (
                    neighbor_edge_attrs["assigned_edge"]
                    not in self.adjacent_edges[edge_attrs["assigned_edge"]]
                ):
                    consistent = False

                    constraint = pylp.LinearConstraint()
                    coefficient_a = self.match_indicators[edge][
                        edge_attrs["assigned_edge"]
                    ]
                    coefficient_b = self.match_indicators[neighbor_edge][
                        neighbor_edge_attrs["assigned_edge"]
                    ]

                    constraint.set_coefficient(coefficient_a, 1)
                    constraint.set_coefficient(coefficient_b, 1)

                    constraint.set_relation(pylp.Relation.LessEqual)
                    constraint.set_value(1)

                    self.constraints.add(constraint)

        return consistent

    def __solution_to_tree(self, solution):
        for edge, edge_attrs in self.graph.edges().items():
            seen = 0
            for possible_assignment in edge_attrs["possible_matches"]:
                coefficient_ind = self.match_indicators[edge][possible_assignment]
                coefficient = solution[coefficient_ind]
                if coefficient == 1:
                    edge_attrs["assigned_edge"] = possible_assignment
                    seen += 1
            assert seen == 1, f"edge {edge} did not get an assignment!"

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

        # For each edge, keep track of all edges that share a node with it
        # including itself and the "no match" edge None.
        # The graph cannot contain adjacent edge assignments that are not adjacent
        # in the tree.
        self.adjacent_edges = {}
        for edge in self.tree.edges():
            self.adjacent_edges[edge] = set([None])
            u, v = edge
            for neighboring_edge in itertools.chain(
                self.tree.edges(u), self.tree.edges(v)
            ):
                self.adjacent_edges[edge].add(neighboring_edge)

    def __find_possible_matches(self):

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

        self.num_variables = 0

        self.match_indicators = {}
        self.match_indicator_costs = []

        for graph_e, graph_e_data in self.graph.edges.items():

            self.match_indicators[graph_e] = {}

            for tree_e, distance in graph_e_data["__possible_matches"]:

                self.match_indicators[graph_e][tree_e] = self.num_variables
                self.match_indicator_costs.append(distance)

                self.num_variables += 1

    def __create_constraints(self):

        self.constraints = pylp.LinearConstraints()

        # pick exactly one of the match_indicators per edge

        for graph_e in self.graph.edges():

            constraint = pylp.LinearConstraint()
            for match_indicator in self.match_indicators[graph_e].values():
                constraint.set_coefficient(match_indicators, 1)
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

            edges = self.graph.adj(n)
            possible_matches = []
            for e in edges:
                possible_matches += self.graph.edges[e]['__possible_matches']
            possible_matches = set(possible_matches)

            for l in possible_matches:
                # k = degree of node
                # x = effective degree of node (k - # of None edges)
                # y = # of edges labelled l
                #
                # (k - 2)*y + x <= 2k - 2

                constraint = pylp.LinearConstraint()

                # (k - 2)*y
                if l is not None:
                    for e in edges:
                        i_el = self.match_indicators[e].get(l, None)
                        if i_el is None:
                            continue
                        constraint.set_coefficient(i_el, k - 2)

                # + x
                for e in edges:
                    i_e0 = self.match_indicators[e][None]
                    constraint.set_coefficient(i_e0, 1)

                # <= 2k - 2
                constraint.set_relation(pylp.Relation.LessEqual)
                constraint.set_value(2*k - 2)

                self.constraints.add(constraint)

    def __create_objective(self):

        self.objective = pylp.LinearObjective(self.num_variables)

        for i, c in enumerate(self.match_indicator_costs):
            self.objective.set_coefficient(i, c)

def match_graph_to_tree(graph, tree, match_attribute, match_distance_treshold):

    matcher = GraphToTreeMatcher(graph, tree, match_distance_treshold)
    matches = matcher.match()

    for e1, e2 in matches:
        graph.edges[e1][match_attribute] = e2
