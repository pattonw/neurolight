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

from funlib.show.neuroglancer import add_layer

neuroglancer.set_server_bind_address("0.0.0.0")


def build_trees(edge_rows, voxel_size):
    if edge_rows is None or len(edge_rows) == 0:
        return None
    trees = nx.DiGraph()
    pbs = {
        int(node_id): node_location
        for node_id, node_location in zip(
            tuple(edge_rows[:, 0]), tuple(edge_rows[:, 1:-1])
        )
    }
    for row in edge_rows:
        u = int(row[0])
        v = int(row[-1])

        if u == -1 or v == -1:
            continue

        pos_u = daisy.Coordinate(tuple(pbs[u])) / voxel_size
        pos_v = daisy.Coordinate(tuple(pbs[v])) / voxel_size

        if u not in trees.nodes:
            trees.add_node(u, pos=pos_u)
        else:
            assert trees.nodes[u]["pos"] == pos_u
        if v not in trees.nodes:
            trees.add_node(v, pos=pos_v)
        else:
            assert trees.nodes[v]["pos"] == pos_v

        trees.add_edge(u, v, d=np.linalg.norm(pos_u - pos_v))
    return trees


def build_trees_from_mst(
    emst, edges_u, edges_v, alpha, coordinate_scale, offset, voxel_size
):
    trees = nx.DiGraph()
    for edge, u, v in zip(np.array(emst), np.array(edges_u), np.array(edges_v)):
        if edge[2] > alpha:
            continue
        pos_u = daisy.Coordinate((u[-3:] / coordinate_scale) + (offset / voxel_size))
        pos_v = daisy.Coordinate((v[-3:] / coordinate_scale) + (offset / voxel_size))
        if edge[0] not in trees.nodes:
            trees.add_node(edge[0], pos=pos_u)
        else:
            assert trees.nodes[edge[0]]["pos"] == pos_u
        if edge[1] not in trees.nodes:
            trees.add_node(edge[1], pos=pos_v)
        else:
            assert trees.nodes[edge[1]]["pos"] == pos_v
        trees.add_edge(edge[0], edge[1], d=edge[2])
    return trees


def add_trees(s, trees, node_id, name, visible=False):
    if trees is None:
        return None
    print(f"Drawing {name} with {len(trees.nodes)} nodes")
    for i, cc_nodes in enumerate(nx.weakly_connected_components(trees)):
        print(f"drawing cc {i} with {len(cc_nodes)} nodes")
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
        print(f"adding {len(mst)} edge annotations")

        s.layers.append(
            name="{}_{}".format(name, i),
            layer=neuroglancer.AnnotationLayer(annotations=mst),
            annotationColor="#{:02X}{:02X}{:02X}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            ),
            visible=visible,
        )


def visualize_hdf5(hdf5_file: Path, voxel_size, mst=False):
    voxel_size = daisy.Coordinate(voxel_size)
    dataset = h5py.File(hdf5_file)
    volumes = list(dataset.get("volumes", {}).keys())
    points = list(dataset.get("points", {}).keys())

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        for volume in volumes:
            v = daisy.open_ds(str(hdf5_file.absolute()), f"volumes/{volume}")
            add_layer(s, v, volume, visible=False)

        node_id = itertools.count(start=1)
        for point_set in points:
            components = build_trees(dataset["points"][point_set], voxel_size)
            add_trees(s, components, node_id, name=point_set, visible=True)
        if mst:
            path_list = str(hdf5_file.absolute()).split("/")
            setups_dir = Path("/", *path_list[:-3])
            setup_config = json.load((setups_dir / "default_config.json").open())
            setup_config.update(
                json.load((setups_dir / path_list[-3] / "config.json").open())
            )

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

