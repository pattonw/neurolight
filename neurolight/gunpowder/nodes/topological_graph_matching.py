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
        graph = mouselight_preprocessing(graph, 50)
        tree = copy.deepcopy(batch[self.T].graph)

        logger.debug("initializing matcher")

        matcher = GraphToTreeMatcher(
            graph,
            tree,
            match_distance_threshold=self.match_distance_threshdold,
            node_balance=self.node_balance,
            use_gurobi=True,
        )
        logger.debug("optimizing!")
        try:
            solution = matcher.solve()
            matched = matcher.create_tree(solution)
            logger.info("Solution found!")
        except ValueError:
            logger.warning("Failed to find a solution for T!")
            matched = SpatialGraph()

            if self.failures is not None:
                self.__save_failed_matching(graph, tree, batch_id=batch.id)

        result = GraphPoints._from_graph(
            matched, copy.deepcopy(batch[self.T].spec)
        )

        batch[self.matched] = result

    def __save_failed_matching(
        self,
        graph: SpatialGraph,
        tree: SpatialGraph,
        component: Optional[Set[Hashable]] = None,
        batch_id: Optional[int] = None,
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
                f"{count:03}.obj" if batch_id is None else f"{count:03}_{batch_id}.obj"
            )
            pickle.dump(data, (self.failures / filename).open("wb"))
