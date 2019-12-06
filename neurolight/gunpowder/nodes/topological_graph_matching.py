from gunpowder import (
    PointsKey,
    BatchRequest,
    Batch,
    BatchFilter,
    SpatialGraph,
    GraphPoints,
)
from funlib.match.helper_functions import match
import networkx as nx

import copy
import logging
from typing import Optional, Set, Hashable
from pathlib import Path
import pickle

from neurolight.match.costs import get_costs
from neurolight.match.preprocess import mouselight_preprocessing

logger = logging.getLogger(__name__)


class TopologicalMatcher(BatchFilter):
    def __init__(
        self,
        G: PointsKey,
        T: PointsKey,
        matched: PointsKey,
        match_distance_threshold: float = 100,
        max_gap_crossing: float = 50,
        node_balance: float = 10,
        failures: Optional[Path] = None,
        try_complete: bool = True,
        use_gurobi: bool = True,
        location_attr: str = "location",
        penalty_attr: str = "penalty",
    ):

        self.G = G
        self.T = T
        self.matched = matched
        self.match_distance_threshdold = match_distance_threshold
        self.max_gap_crossing = max_gap_crossing
        self.node_balance = node_balance
        self.failures = failures
        self.try_complete = try_complete
        self.use_gurobi = use_gurobi

        self.location_attr = location_attr
        self.penalty_attr = penalty_attr

    def setup(self):
        self.enable_autoskip()
        self.provides(self.matched, copy.deepcopy(self.spec[self.T]))

    def prepare(self, request: BatchRequest) -> BatchRequest:
        deps = BatchRequest()
        deps[self.G] = copy.deepcopy(request[self.matched])
        deps[self.T] = copy.deepcopy(request[self.matched])
        return deps

    def process(self, batch: Batch, request: BatchRequest):

        graph = copy.deepcopy(batch[self.G].graph).to_undirected()
        mouselight_preprocessing(graph, self.max_gap_crossing)
        tree = copy.deepcopy(batch[self.T].graph)

        success = False
        if self.try_complete:
            matched, success = self.__solve_tree(graph, tree)
            if not success:
                logger.debug("failed to match a full tree")
        if not success:
            matched, success = self.__solve_piecewise(graph, tree)

        result = GraphPoints._from_graph(matched, copy.deepcopy(batch[self.T].spec))

        batch[self.matched] = result

    def __solve_tree(self, graph, tree):

        if len(graph.nodes) < 1 or len(tree.nodes) < 1:
            return SpatialGraph()
        logger.debug("initializing matcher")
        node_costs, edge_costs = get_costs(
            graph,
            tree,
            location_attr=self.location_attr,
            penalty_attr=self.penalty_attr,
            node_match_threshold=self.match_distance_threshdold,
            edge_match_threshold=self.match_distance_threshdold,
            node_balance=self.node_balance,
        )
        success = True
        try:
            matched = match(
                graph, tree, node_match_costs=node_costs, edge_match_costs=edge_costs
            )
        except ValueError as e:
            logger.debug(e)
            self.__save_failed_matching(graph, tree)
            matched = nx.DiGraph()
            success = False

        return matched, success

    def __solve_piecewise(self, graph, tree):
        """
        If the matching cannot be done as a whole, attempt to do it piecewise.
        This has several advantages and disadvantages.

        Advantages:
            If there are many components, will at least get a partial
                solution even if one component is un-matchable.
        Disadvantages:
            Slow, requires running the matcher many times, this involves a lot
                of repeated iterating over nodes and edges to recreate the
                constraints and indicators that we have already made before
            Crossovers will fail since it is likely that the optimal route
                for both halves will share an edge in the middle.
                Solving this requires more redundant running of the matching
                algorithm.
        """
        node_component_map = {}
        matched_components = []
        invalid_sets = set()
        wccs = list(nx.weakly_connected_components(tree))
        for i, wcc in enumerate(wccs):
            component = nx.DiGraph()
            component.add_nodes_from((n, tree.nodes[n]) for n in wcc)
            component.add_edges_from(
                (n, nbr, d)
                for n, nbrs in tree.adj.items()
                if n in wcc
                for nbr, d in nbrs.items()
                if nbr in wcc
            )
            component.graph.update(tree.graph)

            comp_matched, success = self.__solve_tree(graph, component)
            if success:
                matched_components.append(comp_matched)

                # keep track of which nodes were used
                for node in comp_matched.nodes:
                    mapped_components = node_component_map.setdefault(node, [])
                    mapped_components += [i]
                    if len(mapped_components) > 1:
                        # These are sets of components in T that reuse the same node
                        # in G. They can be matched, and probably would be successfully
                        # matched if they had knowledge of each other due to preprocessing.
                        invalid_sets.add(tuple(mapped_components))
            else:
                logger.debug("Failed to match a component!")

        valid_component_count = len(matched_components)
        logger.debug(
            f"Found a solution for {len(matched_components)} of {len(wccs)} components!"
        )
        for invalid_set in invalid_sets:
            component_group = set(x for ind in invalid_set for x in wccs[ind])
            group_subgraph = tree.subgraph(component_group)
            matched, success = self.__solve_tree(graph, group_subgraph)
            if success:
                matched_components.append(matched)
            else:
                logger.debug("Failed to match a component group")
        logger.debug(
            f"Found a solution for {len(matched_components) - valid_component_count} "
            f"of {len(invalid_sets)} component groups!"
        )
        if len(matched_components) == 0:
            return nx.DiGraph(), False
        try:
            complete = nx.disjoint_union_all(matched_components)
        except Exception:
            # This is technically possible I guess. If there was a crossover
            # that failed when components were run individually, and was very
            # close to a third component, s.t. when running the two components
            # of the crossover together, it forced one of them to use a node
            # that was already in use by the third. At this point I'm just going to throw
            # away that data rather than keep running more matchings.
            # If this becomes a problem, consider a recursive/iterative solution.
            complete = nx.disjoint_union_all(matched_components[: len(wccs)])
            self.__save_failed_matching(graph, tree, failure_type=2)
        return complete, True

    def __save_failed_matching(
        self,
        graph: SpatialGraph,
        tree: SpatialGraph,
        component: Optional[Set[Hashable]] = None,
        batch_id: Optional[int] = None,
        failure_type: int = 0,
    ):
        """
        On matching failure, save the graph, tree and component for which the matching failed.
        Only saves up to 100 failures. After that point it will simply stop saving them.
        """
        if self.failures is None:
            return
        if not self.failures.exists():
            self.failures.mkdir()
        if self.failures.is_dir():
            count = len(list(self.failures.iterdir()))
            if count >= 1000:
                return
            failure_dir = self.failures / f"{count:03}"
            failure_dir.mkdir(exist_ok=True)
            pickle.dump(graph, (failure_dir / "graph.obj").open("wb"))
            pickle.dump(tree, (failure_dir / "tree.obj").open("wb"))
