import copy
from gunpowder import Batch, BatchFilter, ArrayKey, PointsKey
from gunpowder.profiling import Timing
from typing import Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)

Key = Union[ArrayKey, PointsKey]


class DuplicateRequest(BatchFilter):
    """Given an upstream pipeline, this filter makes multiple requests, and merges
    them into one.

        Args:

            mapping (:class:``Dict[Key, Tuple[Key]]``):
                
                A mapping where each key is an upstream "source" key, and each
                value is a tuple of downstream "target" keys which will hold copies of the
                "source".

        """

    def __init__(self, mapping: Dict[Key, Tuple[Key]]):
        self.mapping = mapping

    def setup(self):
        """
        Provide targets that are copies of sources
        """
        for source, targets in self.mapping.items():
            for target in targets:
                self.provides(target, self.spec[source])

    def prepare(self, request, i: int):
        assert len(self.mapping) > 0, "mapping cannot be empty"

        # Don't need to remove targets from request since they are provided
        # and thus will be automatically removed
        for source, targets in self.mapping.items():
            assert source not in request, "Cannot request both {} and any of {}".format(
                source, targets
            )
            if len(targets) > i:
                if targets[i] in request:
                    request[source] = copy.deepcopy(request[targets[i]])

    def provide(self, request):
        """
        Instead of making multiple requests in the process phase,
        this provide method should be overwritten to get two copies of
        the required points and arrays. 
        """

        upstream_request = copy.deepcopy(request)

        timing_prepare = Timing(self, "prepare")
        timing_prepare.start()

        batches = []
        for i in range(max(len(targets) for targets in self.mapping.values())):
            self.prepare(upstream_request, i)
            self.remove_provided(upstream_request)

            batches.append(self.get_upstream_provider().request_batch(upstream_request))

        timing_prepare.stop()

        timing_process = Timing(self, "process")
        timing_process.start()

        batch = self.process(batches, request)

        timing_process.stop()

        batch.profiling_stats.add(timing_prepare)
        batch.profiling_stats.add(timing_process)

        return batch

    def process(self, batches, request):
        # TODO: Handle timing / misc data from upstream batches
        batch_union = Batch()
        for i, batch in enumerate(batches):
            for source, targets in self.mapping.items():
                if source in batch:
                    try:
                        batch_union[targets[i]] = batch[source]
                    except KeyError:
                        raise ValueError(
                            (
                                "Source should only be in batch {} if "
                                + "targets({}) has lenght > {}!"
                            ).format(targets)
                        )
        return batch_union
