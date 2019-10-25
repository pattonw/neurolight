import daisy
import neuroglancer
import numpy as np
import itertools
import networkx as nx
import random
from pathlib import Path
import pickle

from neurolight.transforms.npz_to_graph import parse_npy_graph

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
            assert trees.nodes[u]["location"] == pos_u
        if v not in trees.nodes:
            trees.add_node(v, pos=pos_v)
        else:
            assert trees.nodes[v]["location"] == pos_v

        trees.add_edge(u, v, d=np.linalg.norm(pos_u - pos_v))
    return trees


def add_trees(s, trees, node_id, name, visible=False):
    if trees is None:
        return None
    for i, cc_nodes in enumerate(nx.weakly_connected_components(trees)):
        cc = trees.subgraph(cc_nodes)
        mst = []
        for u, v in cc.edges():
            pos_u = np.array(cc.nodes[u]["location"]) + 0.5
            pos_v = np.array(cc.nodes[v]["location"]) + 0.5
            mst.append(
                neuroglancer.LineAnnotation(
                    point_a=pos_u[::-1], point_b=pos_v[::-1], id=next(node_id)
                )
            )

        s.layers.append(
            name="{}_{}".format(name, i),
            layer=neuroglancer.AnnotationLayer(annotations=mst),
            annotationColor="#{:02X}{:02X}{:02X}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            ),
            visible=visible,
        )


def visualize_match(example: Path):
    graph = parse_npy_graph(example / "graph.npz")
    tree = parse_npy_graph(example / "tree.npz")
    component = obj.get("component")

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers["blegh"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=np.zeros([1, 1, 1]).transpose([2, 1, 0]), voxel_size=[1, 1, 1]
            )
        )
        node_id = itertools.count(start=1)
        if component is not None:
            tree = tree.subgraph(component)
            add_trees(s, tree, node_id, name="component", visible=True)
        else:
            add_trees(s, tree, node_id, name="tree", visible=True)
        add_trees(s, graph, node_id, name="graph", visible=True)
    print(viewer)
    input("Hit ENTER to quit!")

