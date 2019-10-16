from gunpowder import (
    PointsKey,
    BatchRequest,
    Batch,
    BatchFilter,
    SpatialGraph,
    GraphPoints,
)
from neurolight.match_graph_to_tree import GraphToTreeMatcher
import networkx as nx

import copy
import logging
from typing import Optional, Set, Hashable
from pathlib import Path

logger = logging.getLogger(__name__)


class TopologicalMatcher(BatchFilter):
    def __init__(
        self,
        G: PointsKey,
        T: PointsKey,
        matched: PointsKey,
        match_distance_threshold: float = 100,
        node_balance: float = 10,
        failures: Optional[Path] = None,
    ):

        self.G = G
        self.T = T
        self.matched = matched
        self.match_distance_threshdold = match_distance_threshold
        self.node_balance = node_balance
        self.failures = failures

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
        tree = copy.deepcopy(batch[self.T].graph)

        profiling_stats = batch.profiling_stats

        logger.debug("initializing matcher")
        final_solution = SpatialGraph()
        for wcc in nx.weakly_connected_components(tree):
            subgraph = tree.subgraph(wcc)
            matcher = GraphToTreeMatcher(
                graph,
                subgraph,
                match_distance_threshold=self.match_distance_threshdold,
                node_balance=self.node_balance,
                profiling_stats=profiling_stats,
            )
            logger.debug("optimizing!")
            try:
                solution = matcher.solve()
                matched_component = matcher.create_tree(solution)
                logger.info("Solution found!")
            except ValueError:
                logger.warning(
                    "Failed to find a solution for a connected component of T!"
                )
                # Losing out on a connected component does not lead to any ambiguous data,
                # thus we can simply ignore this connected component
                matched_component = SpatialGraph()

                if self.failures is not None:
                    self.__save_failed_component(graph, tree, wcc)
            logger.debug("solution found!")

            try:
                final_solution = nx.union(final_solution, matched_component)
            except nx.exception.NetworkXError:
                # Not sure what is causing this. It would be nice to have a way to
                # write failing cases out to disk so that they can be tested
                # and solved later.
                logging.warning(
                    "matcher tried to use a skeleton node in the matching of two seperate connected component!"
                )
                # In this case, the matcher may have assigned ambiguous ground truth and so we
                # should not allow it to pass as if it were valid
                final_solution = SpatialGraph()

                if self.failures is not None:
                    self.__save_failed_graph(graph, tree)
                break

        result = GraphPoints._from_graph(
            final_solution, copy.deepcopy(batch[self.T].spec)
        )

        batch[self.matched] = result

    def __save_failed_component(
        self, graph: SpatialGraph, tree: SpatialGraph, component: Set[Hashable]
    ):
        pass

    def __save_failed_graph(self, graph: SpatialGraph, tree: SpatialGraph):
        pass
