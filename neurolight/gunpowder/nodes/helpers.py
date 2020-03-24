import gunpowder as gp
import numpy as np
import networkx as nx
import torch

import copy
from typing import Optional
import logging
import time

logger = logging.getLogger(__file__)


class UnSqueeze(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        outputs[self.array] = copy.deepcopy(batch[self.array])
        logger.warning(f"unsqueeze input shape: {outputs[self.array].data.shape}")
        outputs[self.array].data = (
            torch.from_numpy(batch[self.array].data).unsqueeze(0).numpy()
        )
        logger.warning(f"unsqueeze output shape: {outputs[self.array].data.shape}")
        return outputs


class Squeeze(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        outputs[self.array] = copy.deepcopy(batch[self.array])
        logger.warning(f"unsqueeze input shape: {outputs[self.array].data.shape}")
        outputs[self.array].data = (
            torch.from_numpy(batch[self.array].data).squeeze(0).numpy()
        )
        logger.warning(f"unsqueeze output shape: {outputs[self.array].data.shape}")
        return outputs


class ToInt64(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def setup(self):
        self.enable_autoskip()
        provided_spec = self.spec[self.array].copy()
        provided_spec.dtype = np.int64
        self.updates(self.array, provided_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        outputs[self.array] = copy.deepcopy(batch[self.array])
        outputs[self.array].data = batch[self.array].data.astype(np.int64)
        outputs[self.array].spec.dtype = np.int64
        return outputs


class BinarizeGt(gp.BatchFilter):
    def __init__(self, gt, gt_binary):

        self.gt = gt
        self.gt_binary = gt_binary

    def setup(self):
        self.enable_autoskip()
        spec = self.spec[self.gt].copy()
        spec.dtype = np.uint8
        self.provides(self.gt_binary, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.gt] = request[self.gt_binary].copy()
        return deps

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        spec.dtype = np.uint8

        binarized = gp.Array(data=(batch[self.gt].data > 0).astype(np.uint8), spec=spec)

        batch[self.gt_binary] = binarized


class Crop(gp.BatchFilter):
    def __init__(self, input_array: gp.ArrayKey, output_array: gp.ArrayKey):

        self.input_array = input_array
        self.output_array = output_array

    def setup(self):

        spec = self.spec[self.input_array].copy()
        self.provides(self.output_array, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):
        input_data = batch[self.input_array].data
        input_spec = batch[self.input_array].spec
        input_roi = input_spec.roi
        output_roi_shape = request[self.output_array].roi.get_shape()
        shift = (input_roi.get_shape() - output_roi_shape) / 2
        output_roi = gp.Roi(shift, output_roi_shape)
        output_data = input_data[
            tuple(
                map(
                    slice,
                    output_roi.get_begin() / input_spec.voxel_size,
                    output_roi.get_end() / input_spec.voxel_size,
                )
            )
        ]
        output_spec = copy.deepcopy(input_spec)
        output_spec.roi = output_roi

        output_array = gp.Array(output_data, output_spec)

        batch[self.output_array] = output_array


class RejectIfEmpty(gp.BatchFilter):
    def __init__(
        self,
        ensure_nonempty,
        centroid_size: Optional[gp.Coordinate] = None,
        request_limit: int = 100,
        num_components: int = 2,
    ):
        self.ensure_nonempty = ensure_nonempty
        self.request_limit = request_limit
        self.centroid_size = centroid_size
        self.num_components = num_components

    def setup(self):
        self.enable_autoskip()
        self.updates(
            self.ensure_nonempty, copy.deepcopy(self.spec[self.ensure_nonempty])
        )

    def prepare(self, request):
        return copy.deepcopy(request)

    def process(self, batch, request):
        k = 0
        initial_seed = request._random_seed
        upstream_request = request.copy()
        while self._isempty(batch[self.ensure_nonempty]) and k < self.request_limit:
            upstream_request._random_seed = initial_seed + k
            k += 1
            if k == self.request_limit:
                raise RuntimeError(
                    f"Failed to obtain a batch with {self.ensure_nonempty} non empty"
                )
            temp_batch = self.get_upstream_provider().request_batch(upstream_request)
            self._replace_batch(batch, temp_batch)

        if isinstance(self.ensure_nonempty, gp.PointsKey):
            logger.debug(
                f"{self.ensure_nonempty} has {len(batch[self.ensure_nonempty].data)} nodes"
            )

    def _replace_batch(self, batch, replacement):
        for key, val in replacement.items():
            batch[key] = val

        batch.profiling_stats.merge_with(replacement.profiling_stats)
        if replacement.loss is not None:
            batch.loss = replacement.loss
        if replacement.iteration is not None:
            batch.iteration = replacement.iteration

    def _isempty(self, dataset):
        full_roi = dataset.spec.roi
        size = full_roi.get_shape()
        small_roi = full_roi.copy()
        if self.centroid_size is not None:
            diff = self.centroid_size - size
            diff = diff / gp.Coordinate([2] * len(diff))
            small_roi = small_roi.grow(diff, diff)

        dataset = dataset.crop(small_roi, copy=True)

        if isinstance(dataset, gp.Array):
            values = np.unique(dataset.data)
            return len(values) <= self.num_components
        if isinstance(dataset, gp.Points):
            return (
                len(list(nx.weakly_connected_components(dataset.graph)))
                < self.num_components
            )


class ThresholdMask(gp.BatchFilter):
    def __init__(self, array, mask, threshold):
        self.array = array
        self.mask = mask
        self.threshold = threshold

    def setup(self):
        mask_spec = copy.deepcopy(self.spec[self.array])
        mask_spec.dtype = np.uint32
        self.provides(self.mask, mask_spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = copy.deepcopy(request[self.mask])
        return deps

    def process(self, batch, request):
        mask = (batch[self.array].data > self.threshold).astype(np.uint32)
        mask_spec = copy.deepcopy(batch[self.array].spec)
        mask_spec.dtype = np.uint32
        batch[self.mask] = gp.Array(mask, mask_spec)

        return batch


class Mask(gp.BatchFilter):
    def __init__(self, array, mask):
        self.array = array
        self.mask = mask

    def setup(self):
        array_spec = self.spec[self.array].copy()
        self.updates(self.mask, array_spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        deps[self.mask] = request[self.mask].copy()
        return deps

    def process(self, batch, request):
        mask = batch[self.mask]
        array = batch[self.array]
        array.data[mask.data == 0] = 0

        return batch


class FilterComponents(gp.BatchFilter):
    def __init__(self, points, node_offset: int):
        self.points = points
        self.node_offset = node_offset

    def setup(self):
        points_spec = self.spec[self.points].copy()
        self.updates(self.points, points_spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.points] = request[self.points].copy()
        return deps

    def process(self, batch, request):
        points = batch[self.points]
        graph = points.graph
        wccs = list(nx.weakly_connected_components(graph))
        for wcc in wccs:
            if not all([x < self.node_offset for x in wcc]):
                for node in wcc:
                    graph.remove_node(node)
