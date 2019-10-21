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

from funlib.show.neuroglancer import add_layer

f = sys.argv[1]

neuroglancer.set_server_bind_address("0.0.0.0")


def o(f, *args):
    try:
        return f(*args)
    except Exception:
        return None


voxel_size = daisy.Coordinate([10, 3, 3])

raw = o(daisy.open_ds, f, "volumes/raw")

print(raw.shape)

consensus = h5py.File(f).get("consensus", None)
skeletonization = h5py.File(f).get("skeletonization", None)
print(f"Skeletonization has shape {skeletonization.shape}")
matched = h5py.File(f).get("matched", None)


def build_trees_from_swc(swc_rows):
    if swc_rows is None or len(swc_rows) == 0:
        return None
    trees = nx.DiGraph()
    pb = []
    pbs = {
        int(node_id): node_location
        for node_id, node_location in zip(
            tuple(swc_rows[:, 0]), tuple(swc_rows[:, 1:-1])
        )
    }
    for row in swc_rows:
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
    s.layers.append(name="trees", layer=neuroglancer.AnnotationLayer(annotations=pb))


def add_trees(trees, node_id, name, visible=False):
    if trees is None:
        return None
    for i, cc_nodes in enumerate(nx.weakly_connected_components(trees)):
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

        s.layers.append(
            name="{}_{}".format(name, i),
            layer=neuroglancer.AnnotationLayer(annotations=mst),
            annotationColor="#{:02X}{:02X}{:02X}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            ),
            visible=visible,
        )


viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    if raw is not None:
        add_layer(s, raw, "volume", shader="rgb", c=[0, 0, 0])

    node_id = itertools.count(start=1)

    consensus = build_trees_from_swc(consensus)
    skeletonization = build_trees_from_swc(skeletonization)
    matched = build_trees_from_swc(matched)

    add_trees(consensus, node_id, name="Consensus", visible=True)
    add_trees(skeletonization, node_id, name="Skeletonization", visible=True)
    add_trees(matched, node_id, name="Matched", visible=True)

print(viewer)
input("Hit ENTER to quit!")
