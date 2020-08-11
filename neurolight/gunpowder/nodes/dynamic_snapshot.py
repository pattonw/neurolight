import logging
import numpy as np
import os
import copy
from pathlib import Path

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.ext import h5py
from gunpowder import Snapshot

logger = logging.getLogger(__name__)


def example_record_snapshot_func(batch, history):
    """
    Returns True if this batch has the highest or second
    highest loss out of the last 100 seen batches.
    """
    if history is None:
        return True, [batch.loss]
    else:
        if sum([x > batch.loss for x in history]) < 2:
            return True, history[-99:] + [batch.loss]
        else:
            return False, history[-99:] + [batch.loss]


class DynamicSnapshot(Snapshot):
    """
    subclass of Snapshot node that saves a snapshot based on some
    data dependent values rather than simply every `n` iterations.

    To determine when to store a snapshot, provide as input a key-word
    argument called record_snapshot_func.

    record_snapshot_func: (f(batch, history_{t})) -> (bool, history_{t})):

        record_snapshot_func is a function that takes in a batch and
        some history, returns a bool which decides whether a snapshot
        should be stored, along with some history.
        History can take the form of any object. The first call will
        get None history.
    """

    def __init__(self, *args, **kwargs):
        self.record_snapshot_func = kwargs.pop("record_snapshot_func")
        super(self).__init__(*args, **kwargs)

    def process(self, batch, request):

        self.record_snapshot, self.history = self.record_snapshot_func(
            batch, self.history
        )

        if self.record_snapshot:

            try:
                os.makedirs(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(
                self.output_dir,
                self.output_filename.format(
                    id=str(batch.id).zfill(8), iteration=int(batch.iteration or 0)
                ),
            )
            logger.info("saving to %s" % snapshot_name)
            mode = "w" if not Path(snapshot_name).exists() else "r+"
            with h5py.File(snapshot_name, mode) as f:

                for (array_key, array) in batch.arrays.items():

                    if array_key not in self.dataset_names:
                        continue

                    ds_name = self.dataset_names[array_key]

                    if ds_name not in f.keys():
                        if array_key in self.dataset_dtypes:
                            dtype = self.dataset_dtypes[array_key]
                            dataset = f.create_dataset(
                                name=ds_name,
                                data=array.data.astype(dtype),
                                compression=self.compression_type,
                            )

                        else:
                            dataset = f.create_dataset(
                                name=ds_name,
                                data=array.data,
                                compression=self.compression_type,
                            )
                    else:
                        dataset = f[ds_name]

                    if not array.spec.nonspatial:
                        if array.spec.roi is not None:
                            dataset.attrs["offset"] = array.spec.roi.get_offset()
                        dataset.attrs["resolution"] = self.spec[array_key].voxel_size

                    if self.store_value_range:
                        dataset.attrs["value_range"] = (
                            np.asscalar(array.data.min()),
                            np.asscalar(array.data.max()),
                        )

                    # if array has attributes, add them to the dataset
                    for attribute_name, attribute in array.attrs.items():
                        dataset.attrs[attribute_name] = attribute

                for (graph_key, graph) in batch.graphs.items():
                    if graph_key not in self.dataset_names:
                        continue

                    ds_name = self.dataset_names[graph_key]

                    node_ids = []
                    locations = []
                    node_attrs = {}
                    edges = []
                    edge_attrs = {}
                    for attr in self.node_attrs.get(graph_key, []):
                        node_attrs[attr] = []
                    for attr in self.edge_attrs.get(graph_key, []):
                        edge_attrs[attr] = []
                    for node in graph.nodes:
                        node_ids.append(node.id)
                        locations.append(node.location)
                        for key in node_attrs.keys():
                            node_attrs[key].append(node.all[key])
                    for edge in graph.edges:
                        edges.append((edge.u, edge.v))
                        for key in edge_attrs.keys():
                            edge_attrs[key].append(edge.all[key])

                    if f"{ds_name}-ids" not in f.keys():
                        f.create_dataset(
                            name=f"{ds_name}-ids",
                            data=np.array(node_ids, dtype=int),
                            compression=self.compression_type,
                        )
                    if f"{ds_name}-locations" not in f.keys():
                        f.create_dataset(
                            name=f"{ds_name}-locations",
                            data=np.array(locations),
                            compression=self.compression_type,
                        )
                    if f"{ds_name}-edges" not in f.keys():
                        f.create_dataset(
                            name=f"{ds_name}-edges",
                            data=np.array(edges),
                            compression=self.compression_type,
                        )
                    for key, values in node_attrs.items():
                        if f"{ds_name}/node_attrs/{key}" not in f.keys():
                            f.create_dataset(
                                name=f"{ds_name}/node_attrs/{key}",
                                data=np.array(values),
                                compression=self.compression_type,
                            )
                    for key, values in edge_attrs.items():
                        if f"{ds_name}/edge_attrs/{key}" not in f.keys():
                            f.create_dataset(
                                name=f"{ds_name}/edge_attrs/{key}",
                                data=np.array(values),
                                compression=self.compression_type,
                            )

                if batch.loss is not None:
                    f["/"].attrs["loss"] = batch.loss

        self.n += 1
