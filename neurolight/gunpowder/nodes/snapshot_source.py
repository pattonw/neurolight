import gunpowder as gp
import h5py
import numpy as np


class SnapshotSource(gp.BatchProvider):
    def __init__(
        self, snapshot_file, datasets, directed=None, node_attrs=None, edge_attrs=None
    ):
        if directed is None:
            self.directed = {}
        else:
            self.directed = directed

        if node_attrs is None:
            self.node_attrs = {}
        else:
            self.node_attrs = node_attrs

        if edge_attrs is None:
            self.edge_attrs = {}
        else:
            self.edge_attrs = edge_attrs

        self.snapshot_file = snapshot_file
        self.datasets = datasets

    def setup(self):
        data = h5py.File(self.snapshot_file, 'r')
        for key, path in self.datasets.items():
            if isinstance(key, gp.ArrayKey):
                try:
                    x = data[path]
                except KeyError:
                    raise KeyError(f"Could not find {path}")
                spec = self.spec_from_dataset(x)
                self.provides(key, spec)
            elif isinstance(key, gp.GraphKey):
                try:
                    locations = data[f"{path}-locations"]
                except KeyError:
                    raise KeyError(f"Could not find {path}-locations")
                spec = gp.GraphSpec(
                    gp.Roi((None,) * len(locations[0]), (None,) * len(locations[0])),
                    directed=self.directed.get(key),
                )
                self.provides(key, spec)

    def provide(self, request):
        outputs = gp.Batch()
        data = h5py.File(self.snapshot_file)
        for key, path in self.datasets.items():
            if isinstance(key, gp.ArrayKey):
                result = self.array_from_path(data, path)
                outputs[key] = result
            elif isinstance(key, gp.GraphKey):
                result = self.graph_from_path(key, data, path)
                result.relabel_connected_components()
                outputs[key] = result

        outputs = outputs.crop(request)
        return outputs

    def spec_from_dataset(self, x):
        offset = x.attrs["offset"]
        voxel_size = x.attrs["resolution"]
        spatial_dims = len(voxel_size)
        shape = x.shape[-spatial_dims:] * voxel_size
        return gp.ArraySpec(roi=gp.Roi(offset, shape), voxel_size=voxel_size)

    def array_from_path(self, data, path):
        x = data[path]
        spec = self.spec_from_dataset(x)
        return gp.Array(np.array(x), spec)

    def graph_from_path(self, graph_key, data, path):
        saved_ids = data[f"{path}-ids"]
        saved_edges = data[f"{path}-edges"]
        saved_locations = data[f"{path}-locations"]
        node_attrs = [
            (attr, data[f"{path}/node_attrs/{attr}"])
            for attr in self.node_attrs.get(graph_key, [])
        ]
        attrs = [attr for attr, _ in node_attrs]
        attr_values = zip(
            *[values for _, values in node_attrs], (None,) * len(saved_locations)
        )
        nodes = [
            gp.Node(
                node_id,
                location=location,
                attrs={attr: value for attr, value in zip(attrs, values)},
            )
            for node_id, location, values in zip(
                saved_ids, saved_locations, attr_values
            )
        ]

        edge_attrs = [
            (attr, data[f"{path}/edge_attrs/{attr}"])
            for attr in self.edge_attrs.get(graph_key, [])
        ]
        attrs = [attr for attr, _ in edge_attrs]
        attr_values = zip(
            *[values for _, values in edge_attrs], (None,) * len(saved_edges)
        )
        edges = [
            gp.Edge(u, v, attrs={attr: value for attr, value in zip(attrs, values)})
            for (u, v), values in zip(saved_edges, attr_values)
        ]
        return gp.Graph(
            nodes,
            edges,
            gp.GraphSpec(
                gp.Roi(
                    (None,) * len(saved_locations[0]), (None,) * len(saved_locations[0])
                ),
                directed=self.directed.get(graph_key),
            ),
        )
