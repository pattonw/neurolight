import pylp
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import rtree

import logging
import itertools
from copy import deepcopy
from typing import Hashable, Tuple, List, Dict

logger = logging.getLogger(__name__)

Edge = Tuple[Hashable, Hashable]


class GraphToTreeMatcher:

    NO_MATCH_COST = 10e5

    def __init__(
        self,
        graph: nx.Graph,
        tree: nx.DiGraph,
        match_distance_threshold: float,
        node_balance: float = 10,
    ):

        if isinstance(graph, nx.Graph):
        self.undirected_graph = graph
        self.graph = graph.to_directed()
        elif isinstance(graph, nx.DiGraph):
            self.undirected_graph = graph.to_undirected()
            # TODO: support graph as nx.DiGraph
            self.graph = self.undirected_graph.to_directed(graph)
        self.tree = tree
        assert nx.is_arborescence(self.tree), (
            "cannot match an arbitrary source to an arbitrary target. "
            + "target graph should be a tree. i.e. DAG with max in_degree = 1"
        )

        self.match_distance_threshold = match_distance_threshold
        self.node_balance = node_balance

        self.objective = None
        self.constraints = None

        self.__preprocess_graph()
        self.__initialize_spatial_indicies()
        self.__find_possible_matches()
        self.__create_inidicators()
        self.__create_constraints()
        self.__create_objective()

    @property
    def node_match_threshold(self):
        return self.match_distance_threshold

    @property
    def edge_match_threshold(self):
        return self.match_distance_threshold

    def match(self) -> List[Tuple[Edge, Edge]]:
        """Return a list of tuples from ``graph`` edges to ``tree`` edges (or
        ``None``, if no match was found).
        """

        solution = self.solve()

        matches = []
        for graph_e in self.graph.edges():
            for l, i in self.g2t_match_indicators[graph_e].items():
                if solution[i] > 0.5:
                    matches.append((graph_e, l))

        logger.debug(
            f"Found optimal solution with score: {self._score_solution(solution)}"
        )
        for graph_e, tree_e in matches:
            logger.debug(
                f"edge {graph_e} assigned to {tree_e} with score: "
                + f"{self.match_indicator_costs[self.g2t_match_indicators[graph_e][tree_e]]}"
            )

        return matches, self._score_solution(solution)

    def solve(self):

        solver = pylp.create_linear_solver(pylp.Preference.Gurobi)
        solver.initialize(self.num_variables, pylp.VariableType.Binary)
        solver.set_num_threads(1)
        solver.set_timeout(120)

        solver.set_objective(self.objective)

        consistent = False
        while not consistent:

            solver.set_constraints(self.constraints)
            solution, message = solver.solve()
            if "NOT" in message:
                raise ValueError("No solution could be found for this problem!")
            consistent = self.__check_consistency(solution)

        return solution

    def create_tree(self, solution):
        matched_tree = nx.DiGraph()
        for graph_e in self.graph.edges():
            for l, i in self.g2t_match_indicators[graph_e].items():
                if solution[i] > 0.5:
                    matched_tree.add_node(
                        graph_e[0], **deepcopy(self.graph.nodes[graph_e[0]])
                    )
                    matched_tree.add_node(
                        graph_e[1], **deepcopy(self.graph.nodes[graph_e[1]])
                    )
                    matched_tree.add_edge(*graph_e)
                    matched_tree.edges[graph_e]["matched_label"] = l
                    matched_tree.edges[graph_e][
                        "match_cost"
                    ] = self.match_indicator_costs[i]
        for graph_n in self.graph.nodes():
            for l, i in self.g2t_match_indicators[graph_n].items():
                if solution[i] > 0.5:
                    if graph_n not in matched_tree.nodes and l is not None:
                        logger.warning(
                            f"Node {graph_n} in G matched to {l} in T with no adjacent edges "
                            + f"matched!\nNode {l} in T has {len(list(self.tree.edges(l)))} "
                            + f"neighbors!"
                        )
                        matched_tree.add_node(
                            graph_n, **deepcopy(self.graph.nodes[graph_n])
                        )
                    elif graph_n not in matched_tree.nodes and l is None:
                        # Assigning None to nodes in G has no cost, thus there are many optimal
                        # solutions containing excess assignments. These can be ignored
                        pass
                    else:
                        matched_tree.nodes[graph_n]["match_label"] = l
                        matched_tree.nodes[graph_n][
                            "match_cost"
                        ] = self.match_indicator_costs[i]

        return matched_tree

    def _score_solution(self, solution):
        total = 0
        for graph_e in self.graph.edges():
            for l, i in self.g2t_match_indicators[graph_e].items():
                if solution[i] > 0.5:
                    total += self.match_indicator_costs[i]
        return total

    def __initialize_spatial_indicies(self):
        self.tree_kd_ids, tree_node_attrs = [
            list(x) for x in zip(*self.tree.nodes.items())
        ]
        self.tree_kd = cKDTree([attrs["location"] for attrs in tree_node_attrs])

        p = rtree.index.Property()
        p.dimension = 3
        self.tree_rtree = rtree.index.Index(properties=p)
        for i, (u, v) in enumerate(self.tree.edges()):
            u_loc = self.tree.nodes[u]["location"]
            v_loc = self.tree.nodes[v]["location"]
            mins = np.min(np.array([u_loc, v_loc]), axis=0)
            maxs = np.max(np.array([u_loc, v_loc]), axis=0)
            box = tuple(x for x in itertools.chain(mins.tolist(), maxs.tolist()))
            self.tree_rtree.insert(i, box, obj=(u, v))

    def __check_consistency(self, solution):
        return True

    def __preprocess_graph(self):
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
        # TODO: put graph nodes in a tree and use cKDTree.query_ball_tree
        for graph_n in self.graph.nodes():
            possible_tree_nodes = self.__tree_nodes_query(
                self.graph.nodes[graph_n]["location"], self.node_match_threshold
            )
            pm_node = self.possible_matches.setdefault(graph_n, {})
            for tree_n in possible_tree_nodes:
                node_cost = self.__node_cost(graph_n, tree_n)
                if node_cost is not None:
                    pm_node[tree_n] = node_cost

        # TODO: is there an rtree equivalent of cKDTree.query_ball_tree?
        for graph_e in self.undirected_graph.edges():

            possible_tree_edges = self.__tree_edge_query(
                self.graph.nodes[graph_e[0]]["location"],
                self.graph.nodes[graph_e[1]]["location"],
                self.edge_match_threshold,
            )

            pm_edge = self.possible_matches.setdefault(graph_e, {})

            for tree_e in possible_tree_edges:
                edge_cost = self.__edge_cost(graph_e, tree_e)
                if edge_cost is not None:
                    pm_edge[tree_e] = edge_cost

    def __tree_nodes_query(self, center: np.ndarray, radius: float) -> List[Hashable]:
        # simply query the tree kdtree for all nodes within the radius
        queried_ids = self.tree_kd.query_ball_point(center, radius)
        return [self.tree_kd_ids[i] for i in queried_ids]

    def __tree_edge_query(self, u_loc: np.ndarray, v_loc: np.ndarray, radius: float):
        # r tree only stores bounding boxes of lines, so we have to retrieve all potential
        # edges, and then filter them based on a line specific distance calculation
        rect = tuple(
            x
            for x in itertools.chain(
                (np.min(np.array([u_loc, v_loc]), axis=0) - radius).tolist(),
                (np.min(np.array([u_loc, v_loc]), axis=0) + radius).tolist(),
            )
        )
        possible_tree_edges = [
            x.object for x in self.tree_rtree.intersection(rect, objects=True)
        ]
        # line distances
        for x, y in possible_tree_edges:
            dist = self.__edge_dist(
                u_loc,
                v_loc,
                self.tree.nodes[x]["location"],
                self.tree.nodes[y]["location"],
            )
            if dist < radius:
                yield (x, y)

    def __edge_dist(
        self, u_loc: np.ndarray, v_loc: np.ndarray, x_loc: np.ndarray, y_loc: np.ndarray
    ) -> float:
        distance = (
            self.__point_to_edge_dist(u_loc, x_loc, y_loc)
            + self.__point_to_edge_dist(v_loc, x_loc, y_loc)
        ) / 2
        return distance

    def __node_cost(self, graph_n: Hashable, tree_n: Hashable) -> float:
        distance = np.linalg.norm(
            self.graph.nodes[graph_n]["location"] - self.tree.nodes[tree_n]["location"]
        )
        return self.node_balance * distance

    def __edge_cost(self, graph_e: Edge, tree_e: Edge) -> float:
        dist = self.__edge_dist(
            self.graph.nodes[graph_e[0]]["location"],
            self.graph.nodes[graph_e[1]]["location"],
            self.tree.nodes[tree_e[0]]["location"],
            self.tree.nodes[tree_e[1]]["location"],
        )
        return dist

    def __point_to_edge_dist(
        self, center: np.ndarray, u_loc: np.ndarray, v_loc: np.ndarray
    ) -> float:
        slope = v_loc - u_loc
        edge_mag = np.linalg.norm(slope)
        if np.isclose(edge_mag, 0):
            return np.linalg.norm(u_loc - center)
        frac = np.clip(np.dot(center - u_loc, slope) / np.dot(slope, slope), 0, 1)
        min_dist = np.linalg.norm(frac * slope + u_loc - center)
        return min_dist

    def __create_inidicators(self):

        # one binary indicator per edge in graph and possible match edge in
        # tree

        self.num_variables = 0

        self.g2t_match_indicators = {}
        self.t2g_match_indicators = {}
        self.match_indicator_costs = []

        for graph_e in self.graph.edges:

            g2t_e_indicators = self.g2t_match_indicators.setdefault(graph_e, {})

            pm = self.possible_matches.get(
                graph_e, self.possible_matches.get(tuple(graph_e[::-1]), {})
            )

            for tree_e, cost in pm.items():
                t2g_e_indicators = self.t2g_match_indicators.setdefault(tree_e, {})
                t2g_e_indicators[graph_e] = self.num_variables
                g2t_e_indicators[tree_e] = self.num_variables
                self.match_indicator_costs.append(cost)

                self.num_variables += 1

        for graph_n in self.graph.nodes:

            g2t_n_indicators = self.g2t_match_indicators.setdefault(graph_n, {})

            pm = self.possible_matches[graph_n]

            for tree_n, cost in pm.items():
                t2g_n_indicators = self.t2g_match_indicators.setdefault(tree_n, {})
                t2g_n_indicators[graph_n] = self.num_variables
                g2t_n_indicators[tree_n] = self.num_variables
                self.match_indicator_costs.append(cost)

                self.num_variables += 1

            # nodes can match to nothing
            g2t_n_indicators[None] = self.num_variables
            self.match_indicator_costs.append(0)

            self.num_variables += 1

    def __create_constraints(self):
        """
        constraints based on the implementation described here:
        https://hal.archives-ouvertes.fr/hal-00726076/document
        pg 11
        Pierre Le Bodic, Pierre Héroux, Sébastien Adam, Yves Lecourtier. An integer linear program for
        substitution-tolerant subgraph isomorphism and its use for symbol spotting in technical drawings.
        Pattern Recognition, Elsevier, 2012, 45 (12), pp.4214-4224. ffhal-00726076
        """

        self.constraints = pylp.LinearConstraints()

        # (2b) 1-1 Tree node to Graph node mapping
        for tree_n in self.tree.nodes():

            constraint = pylp.LinearConstraint()
            for match_indicator in self.t2g_match_indicators.get(tree_n, {}).values():
                constraint.set_coefficient(match_indicator, 1)
            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(1)
            self.constraints.add(constraint)

        # (2c): 1-n Tree edge to Graph edge mapping (n=1 if multi_edge_match == False)
        for tree_e in self.tree.edges():

            constraint = pylp.LinearConstraint()
            for match_indicator in self.t2g_match_indicators.get(tree_e, {}).values():
                constraint.set_coefficient(match_indicator, -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(-1)
            self.constraints.add(constraint)

        # (2d) 1-1 Graph node to Tree node mapping
        for graph_n in self.graph.nodes():

            constraint = pylp.LinearConstraint()
            for match_indicator in self.g2t_match_indicators[graph_n].values():
                constraint.set_coefficient(match_indicator, 1)
            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(1)
            self.constraints.add(constraint)

        # (2e, 2f)
        for graph_n in self.graph.nodes():
            for tree_n in self.possible_matches[graph_n]:
                g2t_n_indicator = self.g2t_match_indicators[graph_n][tree_n]

                # (2e) 1-1 Tree out edge to Graph out edge mapping for any pair of matched nodes
                for tree_out_e in self.tree.out_edges(tree_n):

                    constraint = pylp.LinearConstraint()
                    constraint.set_coefficient(g2t_n_indicator, 1)
                    for graph_out_e in self.graph.out_edges(graph_n):
                        g2t_e_indicator = self.g2t_match_indicators[graph_out_e].get(
                            tree_out_e, None
                        )
                        if g2t_e_indicator is not None:
                            constraint.set_coefficient(g2t_e_indicator, -1)
                    constraint.set_relation(pylp.Relation.LessEqual)
                    constraint.set_value(0)
                    self.constraints.add(constraint)

                # (2f) 1-1 Tree in edge to Graph in edge mapping for any pair of matched nodes
                for tree_in_e in self.tree.in_edges(tree_n):

                    constraint = pylp.LinearConstraint()
                    constraint.set_coefficient(g2t_n_indicator, 1)
                    for graph_in_e in self.graph.in_edges(graph_n):
                        g2t_e_indicator = self.g2t_match_indicators[graph_in_e].get(
                            tree_in_e, None
                        )
                        if g2t_e_indicator is not None:
                            constraint.set_coefficient(g2t_e_indicator, -11)
                    constraint.set_relation(pylp.Relation.LessEqual)
                    constraint.set_value(0)
                    self.constraints.add(constraint)

        # EXTRA CONSTRAINT: balanced in_edges/out_edges
        # The previous 5 are enough to cover isomorphisms, but we need to model chains
        # in G representing edges in S
        # 1) If two vertices are matched together, an edge originating at the vertex of S
        # must be mapped to n edges targeting the vertex of G, and n+1 edges originating
        # from the vertex in G.
        # 1) is Redundant since this is enforced by previous constraints with n=0
        # 2) Similarly, an edge targeting the vertex of S must be mapped to n+1 edges targeting
        # the vertex of G, and n edges originating from the vertex in G.
        # 2) is Redundant since this is enforced by previous constraints with n=0
        # 3) If a vertex in G is not mapped to a vertex in S, any edge in S
        # matched to n edges targeting the vertex in G, must also be mapped to n edges
        # originating from the vertex in G.

        # MATH
        # given x_ij for all i in V(G) and j in V(T)
        # given y_ab_cd for all (a,b) in E(G) and (c,d) in E(T)
        # For every node i in V(G) and edge (c, d) in E(T)
        #   let n_i_cd = SUM(y_ei_cd) over all edges (e, i) in E(G) (number if in edges)
        #   let m_i_cd = SUM(y_ie_cd) over all edges (i, e) in E(G) (number of out edges)
        #   let imbalance = SUM(x_ih * {1 if h==c, -1 if h==d}) over all nodes h in N(T)
        #   constrain n_i_cd - m_i_cd + imbalance = 0
        for graph_n in self.graph.nodes():

            possible_edges = set(
                tree_e
                for tree_n in self.g2t_match_indicators[graph_n]
                for tree_e in self.tree.edges(tree_n)
            )
            for tree_e in possible_edges:
                equality_constraint = pylp.LinearConstraint()
                for graph_in_e in self.graph.in_edges(graph_n):
                    indicator = self.g2t_match_indicators[graph_in_e].get(tree_e, None)
                    if indicator is not None:
                        # -1 if an in edge matches
                        equality_constraint.set_coefficient(indicator, -1)
                for graph_out_e in self.graph.out_edges(graph_n):
                    indicator = self.g2t_match_indicators[graph_out_e].get(tree_e, None)
                    if indicator is not None:
                        # +1 if an out edge matches
                        equality_constraint.set_coefficient(indicator, 1)

                for tree_n, tree_n_indicator in self.g2t_match_indicators[
                    graph_n
                ].items():
                    # tree_e must be an out edge
                    if tree_n == tree_e[0]:
                        equality_constraint.set_coefficient(tree_n_indicator, -1)
                    # tree_e must be an in edge
                    if tree_n == tree_e[1]:
                        equality_constraint.set_coefficient(tree_n_indicator, 1)

                equality_constraint.set_relation(pylp.Relation.Equal)
                equality_constraint.set_value(0)
                self.constraints.add(equality_constraint)

        # Avoid crossovers in chains and branch points
        # given x_ij for all i in V(G) and j in V(T)
        # given y_ab_cd for all (a, b) in E(G) and (c, d) in E(T)
        # let degree(None) = 2
        # For every node i in V(G)
        #   let N = SUM(degree(c)*x_ic) for all c in V(T) Union None
        #   let y = SUM(y_ai_cd) + SUM(y_ia_cd) for all a adjacent to i, and all (c,d) in E(T)
        #   y - N <= 0
        for graph_n in self.graph.nodes():
            degree_constraint = pylp.LinearConstraint()
            for tree_n, tree_n_indicator in self.g2t_match_indicators[graph_n].items():
                n = 2 if tree_n is None else self.tree.degree(tree_n)
                degree_constraint.set_coefficient(tree_n_indicator, -n)
            # TODO: support graph being a nx.DiGraph
            for neighbor in self.graph.neighbors(graph_n):
                for adj_edge_indicator in self.g2t_match_indicators.get(
                    (graph_n, neighbor), {}
                ).values():
                    degree_constraint.set_coefficient(adj_edge_indicator, 1)

            degree_constraint.set_relation(pylp.Relation.LessEqual)
            degree_constraint.set_value(0)
            self.constraints.add(degree_constraint)

    def __create_objective(self):

        self.objective = pylp.LinearObjective(self.num_variables)

        for i, c in enumerate(self.match_indicator_costs):
            self.objective.set_coefficient(i, c)

    def enforce_expected_assignments(self, expected_assignments: Dict[Edge, Edge]):
        expected_assignment_constraint = pylp.LinearConstraint()
        num_assignments = 0
        for graph_e, tree_e in expected_assignments.items():
            if tree_e is None:
                for ns, cost in self.possible_matches.get(graph_e, {}).items():
                    expected_assignment_constraint.set_coefficient(
                        self.g2t_match_indicators[graph_e][ns], -1
                    )
            else:
                expected_assignment_constraint.set_coefficient(
                    self.g2t_match_indicators[graph_e][tree_e], 1
                )
                num_assignments += 1
        expected_assignment_constraint.set_relation(pylp.Relation.Equal)
        expected_assignment_constraint.set_value(num_assignments)
        self.constraints.add(expected_assignment_constraint)


def match_graph_to_tree(
    graph: nx.Graph,
    tree: nx.DiGraph,
    match_attribute: str,
    match_distance_threshold: float,
):
    for graph_e, graph_e_attrs in graph.edges.items():
        if match_attribute in graph_e_attrs:
            del graph_e_attrs[match_attribute]

    matcher = GraphToTreeMatcher(graph, tree, match_distance_threshold)
    matches, score = matcher.match()

    for e1, e2 in matches:
        match_attr = graph.edges[e1].setdefault(match_attribute, e2)
        if isinstance(match_attribute, list):
            match_attr.append(e1)
        elif match_attr != e2:
            graph.edges[e1][match_attribute] = [match_attr, e2]

    return matcher, score
