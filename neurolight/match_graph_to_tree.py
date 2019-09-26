import pylp
import numpy as np

import itertools


class Edge:
    def __init__(self, u=None, v=None):
        self.u = u
        self.v = v

    @property
    def empty(self):
        return self.u is None or self.v is None

    def __contains__(self, item):
        return item == self.u or item == self.v

    def __eq__(self, other):
        if not isinstance(other, Edge):
            raise ValueError("cannot compare to classes other than Edge")
        return (self.u == other.u or self.u == other.v) and (
            self.v == other.u or self.v == other.v
        )

    def __str__(self):
        if hash(self.u) <= hash(self.v):
            return f"({self.u}, {self.v})"
        else:
            return f"({self.v}, {self.u})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(hash(self.u) + hash(self.v))


class GraphToTreeMatcher:

    NO_MATCH_COST = 10e5

    def __init__(self, graph, tree, match_distance_threshold):

        self.graph = graph
        self.tree = tree
        self.match_distance_threshold = match_distance_threshold

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
            graph_e = Edge(*graph_e)
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

        # TODO: check if solution is consistent
        # if not, add additional constraints and return False

        self.__assign_edges_to_graph(solution)

        consistent = True

        for current_e, current_e_attrs in self.graph.edges().items():
            current_e = Edge(*current_e)
            current_assigned = current_e_attrs["__assigned_edge"]
            # since the edges are unordered, the tree edge ordering may not match the
            # graph edge ordering. Thus since we don't know whether to map u-u and v-v or
            # u-v and v-u, we will simply check that the assignments are consistent on both
            # sides.
            current_x = current_assigned.u
            current_y = current_assigned.v

            # make sure all edges on the x side are adjacent to u or v and
            # all edges on the v side are adjacent to the other.
            u_x_edges = []  # edges on the u side of current_e adjacent to x not y
            u_y_edges = []  # edges on the u side of current_e adjacent to y not x
            v_x_edges = []  # edges on the v side of current_e adjacent to x not y
            v_y_edges = []  # edges on the v side of current_e adjacent to y not x

            non_adjacents = [] # edges that are not adjacent to this edge

            # check all edges that are adjacent to the current edge u
            for u, v, u_neighbor_attrs in self.graph.edges(current_e.u, data=True):
                u_neighbor_e = Edge(u, v)
                u_neighbor_assigned_e = u_neighbor_attrs["__assigned_edge"]
                if (
                    current_x in u_neighbor_assigned_e
                    and current_y not in u_neighbor_assigned_e
            ):
                    u_x_edges.append((u_neighbor_e, u_neighbor_assigned_e))
                elif (
                    current_x not in u_neighbor_assigned_e
                    and current_y in u_neighbor_assigned_e
                ):
                    u_y_edges.append((u_neighbor_e, u_neighbor_assigned_e))
                elif (
                    current_x not in u_neighbor_assigned_e
                    and current_y not in u_neighbor_assigned_e
                ):
                    non_adjacents.append((u_neighbor_e, u_neighbor_assigned_e))
                else:
                    # repeated edges are fine as long as they are on chains.
                    # branch cases should be handled by constraints
                    pass

            # check all edges that are adjacent to the current edge v
            for u, v, v_neighbor_attrs in self.graph.edges(current_e.v, data=True):
                v_neighbor_e = Edge(u, v)
                v_neighbor_assigned_e = v_neighbor_attrs["__assigned_edge"]
                if (
                    current_x in v_neighbor_assigned_e
                    and current_y not in v_neighbor_assigned_e
                ):
                    v_x_edges.append((v_neighbor_e, v_neighbor_assigned_e))
                elif (
                    current_x not in v_neighbor_assigned_e
                    and current_y in v_neighbor_assigned_e
                ):
                    v_y_edges.append((v_neighbor_e, v_neighbor_assigned_e))
                elif (
                    current_x not in v_neighbor_assigned_e
                    and current_y not in v_neighbor_assigned_e
                ):
                    non_adjacents.append((v_neighbor_e, v_neighbor_assigned_e))
                else:
                    # repeated edges are fine as long as they are on chains.
                    # branch cases should be handled by constraints
                    pass

            for (x_edge, x_assigned), (y_edge, y_assigned) in itertools.chain(
                itertools.product(u_x_edges, u_y_edges),
                itertools.product(v_x_edges, v_y_edges),
            ):
                consistent = False

                constraint = pylp.LinearConstraint()
                coefficient_a = self.match_indicators[x_edge][x_assigned]
                coefficient_b = self.match_indicators[y_edge][y_assigned]
                coefficient_c = self.match_indicators[current_e][current_assigned]

                constraint.set_coefficient(coefficient_a, 1)
                constraint.set_coefficient(coefficient_b, 1)
                constraint.set_coefficient(coefficient_c, 1)

                constraint.set_relation(pylp.Relation.LessEqual)
                constraint.set_value(2)

                self.constraints.add(constraint)

            for (u_edge, u_assigned), (v_edge, v_assigned) in itertools.chain(
                itertools.product(u_x_edges, v_x_edges),
                itertools.product(u_y_edges, v_y_edges),
                ):
                    consistent = False

                    constraint = pylp.LinearConstraint()
                coefficient_a = self.match_indicators[u_edge][u_assigned]
                coefficient_b = self.match_indicators[v_edge][v_assigned]
                coefficient_c = self.match_indicators[current_e][current_assigned]

                constraint.set_coefficient(coefficient_a, 1)
                constraint.set_coefficient(coefficient_b, 1)
                constraint.set_coefficient(coefficient_c, 1)

                constraint.set_relation(pylp.Relation.LessEqual)
                constraint.set_value(2)

                self.constraints.add(constraint)

            for (adjacent_e, non_adjacent_assigned) in non_adjacents:
                # they are not adjacent to the current edge and thus should not be allowed
                
                consistent = False

                constraint = pylp.LinearConstraint()
                coefficient_a = self.match_indicators[current_e][current_assigned]
                coefficient_b = self.match_indicators[adjacent_e][non_adjacent_assigned]

                    constraint.set_coefficient(coefficient_a, 1)
                    constraint.set_coefficient(coefficient_b, 1)

                    constraint.set_relation(pylp.Relation.LessEqual)
                    constraint.set_value(1)

                    self.constraints.add(constraint)

        return consistent

    def __assign_edges_to_graph(self, solution):
        for graph_e, graph_e_attrs in self.graph.edges().items():
            graph_e = Edge(*graph_e)
            seen = 0
            for possible_match, cost in graph_e_attrs["__possible_matches"]:
                coefficient_ind = self.match_indicators[graph_e][possible_match]
                coefficient = solution[coefficient_ind]
                if coefficient == 1:
                    graph_e_attrs["__assigned_edge"] = possible_match
                    seen += 1

    def __preprocess_tree(self):
        for node, degree in list(self.tree.degree()):
            if degree > 3:
                neighbors = list(self.tree.neighbors(node))
                for i, neighbor in enumerate(neighbors):
                    self.tree.add_node(
                        (node, neighbor), location=self.tree.nodes[node]["location"]
                    )
                    self.tree.add_edge(neighbor, (node, neighbor))
                    for j in range(i):
                        self.tree.add_edge((node, neighbor), (node, neighbors[j]))
                self.tree.remove_node(node)

        # For each edge, keep track of all edges that share a node with it
        # including itself and the "no match" edge None.
        # The graph cannot contain adjacent edge assignments that are not adjacent
        # in the tree.
        self.adjacent_edges = {}
        for current_e in self.tree.edges():
            current_e = Edge(*current_e)
            self.adjacent_edges[current_e] = set([Edge()])
            for neighbor_e in itertools.chain(
                self.tree.edges(current_e.u), self.tree.edges(current_e.v)
            ):
                neighbor_e = Edge(*neighbor_e)
                self.adjacent_edges[current_e].add(neighbor_e)

    def __find_possible_matches(self):

        self.possible_matches = {}

        for g_u, g_v, graph_e_data in self.graph.edges(data=True):
            graph_e = Edge(g_u, g_v)
            graph_e_data["__possible_matches"] = [(Edge(), self.NO_MATCH_COST)]

            for t_u, t_v, tree_e_data in self.tree.edges(data=True):
                tree_e = Edge(t_u, t_v)
                distance = self.__edge_distance(graph_e, tree_e)

                if distance <= self.match_distance_threshold:
                    graph_e_data["__possible_matches"].append((tree_e, distance))

    def __edge_distance(self, graph_edge, tree_edge):
        # average the distance of the endpoints of the graph edge to the tree edge
        g_u, g_v = graph_edge.u, graph_edge.v
        dist = (
            self.__point_to_edge_dist(g_u, tree_edge)
            + self.__point_to_edge_dist(g_v, tree_edge)
        ) / 2
        return dist

    def __point_to_edge_dist(self, point, edge):
        point_loc = self.graph.nodes[point]["location"]
        u_loc = self.tree.nodes[edge.u]["location"]
        v_loc = self.tree.nodes[edge.v]["location"]
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
