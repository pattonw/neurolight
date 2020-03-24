import daisy
import neuroglancer
import numpy as np
import sys
import itertools
import h5py
import networkx as nx
import json
import random
import sys
from pathlib import Path
from scipy.ndimage.filters import maximum_filter

from neurolight.pipelines import DEFAULT_CONFIG

from funlib.show.neuroglancer import add_layer
import mlpack as mlp

neuroglancer.set_server_bind_address("0.0.0.0")


def build_trees(node_ids, locations, edges, voxel_size):
    trees = nx.DiGraph()
    pbs = {
        int(node_id): node_location
        for node_id, node_location in zip(
            node_ids, locations
        )
    }
    for row in edges:
        u = int(row[0])
        v = int(row[-1])

        if u == -1 or v == -1:
            continue

        pos_u = daisy.Coordinate(tuple(pbs[u])) / voxel_size
        pos_v = daisy.Coordinate(tuple(pbs[v])) / voxel_size

        if u not in trees.nodes:
            trees.add_node(u, pos=pos_u)
        else:
            assert trees.nodes[u]["pos"] == pos_u, "locations don't match"
        if v not in trees.nodes:
            trees.add_node(v, pos=pos_v)
        else:
            assert trees.nodes[v]["pos"] == pos_v, "locations don't match"

        trees.add_edge(u, v, d=np.linalg.norm(pos_u - pos_v))
    return trees


def get_embedding_mst(embedding, alpha, coordinate_scale, offset, candidates):

    _, depth, height, width = embedding.shape
    coordinates = np.meshgrid(
        np.arange(0, depth * coordinate_scale[0], coordinate_scale[0]),
        np.arange(0, height * coordinate_scale[1], coordinate_scale[1]),
        np.arange(0, width * coordinate_scale[2], coordinate_scale[2]),
        indexing="ij",
    )
    for i in range(len(coordinates)):
        coordinates[i] = coordinates[i].astype(np.float32)
    embedding = np.concatenate([embedding, coordinates], 0)
    embedding = np.transpose(embedding, axes=[1, 2, 3, 0])
    embedding = embedding.reshape(depth * width * height, -1)
    candidates = candidates.reshape(depth * width * height)
    embedding = embedding[candidates == 1, :]

    emst = mlp.emst(embedding)["output"]

    mst = nx.DiGraph()
    for u, v, distance in emst:
        u = int(u)
        pos_u = pos = embedding[u][-3:] / coordinate_scale
        v = int(v)
        pos_v = pos = embedding[v][-3:] / coordinate_scale
        mst.add_node(u, pos=pos_u + offset)
        mst.add_node(v, pos=pos_v + offset)
        if alpha > distance:
            mst.add_edge(u, v)
    return mst


def build_trees_from_mst(
    emst, edges_u, edges_v, alpha, coordinate_scale, offset, voxel_size
):
    trees = nx.DiGraph()
    ndims = len(voxel_size)
    for edge, u, v in zip(np.array(emst), np.array(edges_u), np.array(edges_v)):
        if edge[2] > alpha:
            continue
        pos_u = daisy.Coordinate(
            (0,) * (3 - ndims)
            + tuple((u[-ndims:] / coordinate_scale) + (offset / voxel_size))
        )
        pos_v = daisy.Coordinate(
            (0,) * (3 - ndims)
            + tuple((v[-ndims:] / coordinate_scale) + (offset / voxel_size))
        )
        if edge[0] not in trees.nodes:
            trees.add_node(edge[0], pos=pos_u)
        else:
            assert trees.nodes[edge[0]]["pos"] == pos_u, "locations don't match"
        if edge[1] not in trees.nodes:
            trees.add_node(edge[1], pos=pos_v)
        else:
            assert trees.nodes[edge[1]]["pos"] == pos_v, "locations don't match"
        trees.add_edge(edge[0], edge[1], d=edge[2])
    return trees


def add_trees(s, trees, node_id, name, visible=False):
    if trees is None:
        return None
    # print(f"Drawing {name} with {len(trees.nodes)} nodes")
    for i, cc_nodes in enumerate(nx.weakly_connected_components(trees)):
        # print(f"drawing cc {i} with {len(cc_nodes)} nodes")
        cc = trees.subgraph(cc_nodes)
        mst = []
        for u, v in cc.edges():
            pos_u = np.array(cc.nodes[u]["pos"]) + 0.5
            pos_v = np.array(cc.nodes[v]["pos"]) + 0.5
            mst.append(
                neuroglancer.LineAnnotation(
                    point_a=pos_u[::-1], point_b=pos_v[::-1], id=next(node_id)
                )
            )
        # print(f"adding {len(mst)} edge annotations")

        s.layers.append(
            name="{}_{}".format(name, i),
            layer=neuroglancer.AnnotationLayer(annotations=mst),
            annotationColor="#{:02X}{:02X}{:02X}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            ),
            visible=visible,
        )


def visualize_hdf5(hdf5_file: Path, voxel_size, mst=False, maxima_for=None, skip=None):
    path_list = str(hdf5_file.absolute()).split("/")
    setups_dir = Path("/", *path_list[:-3])
    setup_config = DEFAULT_CONFIG
    try:
        setup_config.update(
            json.load((setups_dir / path_list[-3] / "config.json").open())
        )
    except:
        pass
    voxel_size = daisy.Coordinate(setup_config["VOXEL_SIZE"])
    coordinate_scale = (
        setup_config["COORDINATE_SCALE"] * np.array(voxel_size) / max(voxel_size)
    )
    dataset = h5py.File(hdf5_file)
    volumes = list(dataset.get("volumes", {}).keys())
    points = list(dataset.get("points", {}).keys())

    points = set([p.split("-")[0] for p in points])

    node_id = itertools.count(start=1)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        for volume in volumes:
            if skip == volume:
                continue
            v = daisy.open_ds(str(hdf5_file.absolute()), f"volumes/{volume}")
            if v.dtype == np.int64:
                v.materialize()
                v.data = v.data.astype(np.uint64)
            if volume == maxima_for:
                v.materialize()
                max_filtered = maximum_filter(v.data, (3, 10, 10))
                maxima = np.logical_and(max_filtered == v.data, v.data > 0.01)
                m = daisy.Array(maxima, v.roi, v.voxel_size)
                add_layer(s, m, f"{volume}-maxima")
            if volume == "embedding":
                offset = v.roi.get_offset()
                mst = get_embedding_mst(
                    v.data,
                    1,
                    coordinate_scale,
                    offset / voxel_size,
                    daisy.open_ds(
                        str(hdf5_file.absolute()), f"volumes/fg_maxima"
                    ).to_ndarray(),
                )
                add_trees(s, mst, node_id, name="MST", visible=True)
                v.materialize()
                v.data = (v.data + 1) / 2
            add_layer(s, v, volume, visible=False)

        for point_set in points:
            node_ids = dataset["points"][f"{point_set}-ids"]
            locations = dataset["points"][f"{point_set}-locations"]
            edges = dataset["points"][f"{point_set}-edges"]
            components = build_trees(node_ids, locations, edges, voxel_size)
            add_trees(s, components, node_id, name=point_set, visible=False)
        if mst and False:
            emst = h5py.File(hdf5_file)["emst"]
            edges_u = h5py.File(hdf5_file)["edges_u"]
            edges_v = h5py.File(hdf5_file)["edges_v"]
            alpha = setup_config["ALPHA"]
            coordinate_scale = setup_config["COORDINATE_SCALE"]
            offset = daisy.open_ds(
                str(hdf5_file.absolute()), f"volumes/gt_fg"
            ).roi.get_offset()
            mst_trees = build_trees_from_mst(
                emst, edges_u, edges_v, alpha, coordinate_scale, offset, voxel_size
            )
            add_trees(s, mst_trees, node_id, name="MST", visible=True)
    print(viewer)
    input("Hit ENTER to quit!")


def visualize_npy(npy_file: Path, voxel_size):
    voxel_size = daisy.Coordinate(voxel_size)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        v = np.load(npy_file)
        m = daisy.Array(
            v,
            daisy.Roi(daisy.Coordinate([0, 0, 0]), daisy.Coordinate(v.shape)),
            daisy.Coordinate([1, 1, 1]),
        )
        add_layer(s, m, f"npy array")
    print(viewer)
    input("Hit ENTER to quit!")
