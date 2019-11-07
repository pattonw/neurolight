from gunpowder import (
    PointsKey,
    BatchRequest,
    Batch,
    BatchFilter,
    SpatialGraph,
    GraphPoints,
)
from funlib.match.graph_to_tree_matcher import GraphToTreeMatcher
from funlib.match.preprocess import mouselight_preprocessing
import networkx as nx

import copy
import logging
from typing import Optional, Set, Hashable
from pathlib import Path
import pickle

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

    def setup(self):
        self.enable_autoskip()
        self.provides(self.matched, self.spec[self.T])

    def prepare(self, request: BatchRequest) -> BatchRequest:
        deps = BatchRequest()
        deps[self.G] = copy.deepcopy(request[self.matched])
        deps[self.T] = copy.deepcopy(request[self.matched])
        return deps

    def process(self, batch: Batch, request: BatchRequest):

        graph = copy.deepcopy(batch[self.G].graph)
        mouselight_preprocessing(graph, self.max_gap_crossing)
        tree = copy.deepcopy(batch[self.T].graph)

        try:
            if self.try_complete:
                matched = self.__solve_tree(graph, tree)
            else:
                raise ValueError("Skip initial solve!")
        except ValueError:
            try:
                matched = self.__solve_piecewise(graph, tree)
            except Exception:
                logger.warning("Failed to find a solution for T!")
                self.__save_failed_matching(graph, tree, failure_type=3)
                matched = SpatialGraph()

        result = GraphPoints._from_graph(matched, copy.deepcopy(batch[self.T].spec))

        batch[self.matched] = result

    def __solve_tree(self, graph, tree):

        logger.debug("initializing matcher")
        matcher = GraphToTreeMatcher(
            graph,
            tree,
            match_distance_threshold=self.match_distance_threshdold,
            node_balance=self.node_balance,
            use_gurobi=self.use_gurobi,
        )
        logger.debug("optimizing!")
        solution = matcher.solve()
        matched = matcher.create_tree(solution)
        logger.info("Solution found!")

        return matched

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
            component = tree.subgraph(wcc)
            try:
                comp_matched = self.__solve_tree(graph, component)
                matched_components.append(comp_matched)

                # keep track of which nodes were used
                for node in comp_matched.nodes:
                    mapped_components = node_component_map.setdefault(node, [])
                    if tuple(mapped_components) in invalid_sets:
                        invalid_sets.remove()
                    mapped_components += [i]
                    if len(mapped_components) > 1:
                        # These are sets of components in T that reuse the same node
                        # in G. They can be matched, and probably would be successfully
                        # matched if they had knowledge of each other due to preprocessing.
                        invalid_sets.add(tuple(mapped_components))
            except ValueError:
                # These are components of T that cannot be matched in G
                self.__save_failed_matching(graph, tree, wcc, failure_type=0)

        valid_component_count = len(matched_components)
        logger.info(
            f"Found a solution for {len(matched_components)} of {len(wccs)} components!"
        )
        for invalid_set in invalid_sets:
            component_group = set(x for ind in invalid_set for x in wccs[ind])
            group_subgraph = tree.subgraph(component_group)
            try:
                matched = self.__solve_tree(graph, group_subgraph)
            except ValueError:
                # Not sure what went wrong here! Individually these components passed.
                # Graph preprocessing must not have been good enough to handle this
                # case.
                self.__save_failed_matching(
                    graph, tree, component_group, failure_type=1
                )
                matched = SpatialGraph()
            matched_components.append(matched)
        logger.info(
            f"Found a solution for {len(matched_components) - valid_component_count} "
            f"of {len(invalid_sets)} component groups!"
        )
        try:
            complete = nx.disjoint_union_all(matched_components)
        except Exception:
            # This is technically possible I guess. If there was a crossover
            # that failed when components were run individually, and was very
            # close to a third component, s.t. when running the two components
            # of the crossover together, it forced one of them to use a node
            # that was already in use. At this point I'm just going to throw
            # away that data rather than keep running more matchings.
            # If this becomes a problem, consider a recursive/iterative program.
            complete = nx.disjoint_union_all(matched_components[: len(wccs)])
            self.__save_failed_matching(graph, tree, failure_type=2)
        return complete

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
        if not self.failures.exists():
            self.failures.mkdir()
        if self.failures.is_dir():
            count = len(list(self.failures.iterdir()))
            if count >= 1000:
                return
            data = {"graph": graph, "tree": tree, "component": component}
            filename = (
                f"{count:03}_{failure_type}.obj"
                if batch_id is None
                else f"{count:03}_{batch_id}_{failure_type}.obj"
            )
            pickle.dump(data, (self.failures / filename).open("wb"))
