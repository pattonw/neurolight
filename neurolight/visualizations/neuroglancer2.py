import daisy
import neuroglancer
import numpy as np
import sys
import itertools
import h5py
import zarr
import networkx as nx
import json
from pathlib import Path
import copy
import random

from comatch import match_components
from neurolight_evaluation.graph_matching.comatch.edges_xy import get_edges_xy
from neurolight_evaluation.graph_metrics import psudo_graph_edit_distance

from scipy.ndimage.filters import maximum_filter
from sklearn.decomposition import PCA

from neurolight.pipelines import DEFAULT_CONFIG

# from funlib.show.neuroglancer import add_layer
import mlpack as mlp

import logging

logger = logging.getLogger(__name__)


class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, cc, dimensions, voxel_size, node_attrs=None, edge_attrs=None):
        super(SkeletonSource, self).__init__(dimensions)
        self.vertex_attributes["distance"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["node_edge"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["edge_len"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["component"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "component_size"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.cc = cc
        self.voxel_size = voxel_size
        for node, attrs in self.cc.nodes.items():
            assert "location" in attrs

        self.component_ids = {}

    def get_skeleton(self, i):

        edges = []
        distances = []
        vertex_positions = []
        node_edge = []
        edge_len = []
        component = []
        component_size = []

        print(
            f"rendering nodes and edges with {len(self.cc.nodes)} nodes and {len(self.cc.edges)} edges"
        )

        for i, n in enumerate(self.cc.nodes):
            vertex_positions.append(
                self.cc.nodes[n]["location"] / daisy.Coordinate(self.voxel_size)
            )
            vertex_positions.append(
                self.cc.nodes[n]["location"] / daisy.Coordinate(self.voxel_size)
            )
            distances.append(0.1)
            distances.append(0.1)
            edges.append(2 * i)
            edges.append(2 * i + 1)
            node_edge.append(1)
            node_edge.append(1)
            edge_len.append(0)
            edge_len.append(0)
            component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[n].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[n].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            component_size.append(self.cc.nodes[n].get("component_size", 0))
            component_size.append(self.cc.nodes[n].get("component_size", 0))
        i += 1

        for j, (u, v) in enumerate(self.cc.edges):
            vertex_positions.append(
                self.cc.nodes[u]["location"] / daisy.Coordinate(self.voxel_size)
            )
            vertex_positions.append(
                self.cc.nodes[v]["location"] / daisy.Coordinate(self.voxel_size)
            )
            distances.append(self.cc.edges[(u, v)].get("distance", 0.5))
            distances.append(self.cc.edges[(u, v)].get("distance", 0.5))
            edges.append((2 * i) + 2 * j)
            edges.append((2 * i) + 2 * j + 1)
            node_edge.append(0)
            node_edge.append(0)
            edge_len.append(np.linalg.norm(vertex_positions[-1] - vertex_positions[-2]))
            edge_len.append(np.linalg.norm(vertex_positions[-1] - vertex_positions[-2]))
            component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[u].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            component.append(
                self.component_ids.setdefault(
                    self.cc.nodes[v].get("component_id", 0),
                    (
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                        random.randint(0, 255) / 256,
                    ),
                )
            )
            component_size.append(self.cc.nodes[u].get("component_size", 0))
            component_size.append(self.cc.nodes[v].get("component_size", 0))

        return neuroglancer.skeleton.Skeleton(
            vertex_positions=vertex_positions,
            edges=edges,
            vertex_attributes=dict(
                distance=np.array(distances),
                node_edge=node_edge,
                edge_len=edge_len,
                component=component,
                component_size=component_size,
            ),
            # edge_attribues=dict(distances=distances),
        )


class MatchSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, graph, dimensions, voxel_size):
        super(MatchSource, self).__init__(dimensions)
        self.vertex_attributes["source"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes["gt_edge"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes[
            "success_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes[
            "false_match"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "false_match_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["merge"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "merge_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["split"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "split_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["node_edge"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes[
            "selected_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes[
            "component_matchings"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "all_colors"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.graph = graph
        self.voxel_size = voxel_size

        self.init_colors()

    def init_colors(self):
        red = np.array((255, 128, 128)) / 256
        green = np.array((128, 255, 128)) / 256
        blue = np.array((128, 128, 255)) / 256
        yellow = np.array((255, 255, 128)) / 256
        purple = np.array((255, 128, 255)) / 256
        grey = np.array((125, 125, 125)) / 256
        cyan = np.array((0, 255, 255)) / 256
        self.error_vis_colors = {
            "split": red,
            "merge": blue,
            "true_pos": green,
            "false_neg": red,
            "false_pos": blue,
            "filtered": grey,
            "other": cyan,
        }
        self.error_vis_type = {
            (1, 0, 0, 0, 0, 0): "filtered",
            (0, 1, 0, 0, 0, 0): "true_pos",
            (0, 0, 1, 0, 0, 0): "false_pos",
            (0, 0, 0, 1, 0, 0): "false_neg",
            (0, 0, 0, 0, 1, 0): "merge",
            (0, 0, 0, 0, 0, 1): "split",
        }
        self.source_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.distance_color_range = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.matching_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.false_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.merge_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.split_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256

    def random_color(self):
        return np.array([random.randrange(256) / 256 for _ in range(3)])

    def is_no_match_label(self, label):
        return (
            label == max(max(self.labels_y.values()), max(self.labels_x.values())) + 1
        )

    def is_no_match_node(self, node):
        return node == max(max(self.nodes_x), max(self.nodes_y)) + 1

    def get_skeleton(self, i):

        # edges only show up if both of their end points satisfy the condition
        # in shader. i.e. *_edge > 0.5

        # gt_edge, an edge from mst to mst or gt to gt
        #   simple 1 node per node
        # distance_edge, an edge in mst, labeled with color
        #   needs individual nodes per edge to annotate with distance
        # matching edge, an edge in mst or gt labeled with color
        #   edge in mst or gt where labels same on either endpoint
        # false_match_edge, self edge on mst or gt without label match
        # merge_edge, edge in mst connecting two nodes matched to different labels
        # split_edge, edge in gt connecting two nodes matched to different labels

        edges = []
        node_edge = []
        vertex_positions = []
        success_edge = []
        source = []
        gt_edge = []
        false_match = []
        false_match_edge = []
        merge = []
        merge_edge = []
        split = []
        split_edge = []
        selected_edge = []
        component_matchings = []

        all_colors = []

        component_colors = {}

        threshold_index = 25

        for i, ((u, v), attrs) in enumerate(self.graph.edges.items()):
            node_edge.append(0)
            node_edge.append(0)

            selected, success, e_fp, e_fn, e_merge, e_split, e_gt = [
                int(x) for x in attrs["details"][threshold_index]
            ]

            selected_edge.append(selected)
            selected_edge.append(selected)

            success_edge.append(success)
            success_edge.append(success)

            component_matchings.append(
                component_colors.setdefault(
                    tuple(attrs["label_pair"][threshold_index]), self.random_color()
                )
            )
            component_matchings.append(
                component_colors.setdefault(
                    tuple(attrs["label_pair"][threshold_index]), self.random_color()
                )
            )

            assert not (success and e_merge)
            assert not (success and e_split)

            # assert (
            #     sum([1 - selected, success, e_fp, e_fn, e_merge, e_split]) == 1
            # ), f"{[1 - selected, success, e_fp, e_fn, e_merge, e_split]}"
            error = (1 - selected, success, e_fp, e_fn, e_merge, e_split)
            try:
                error_vis_color = self.error_vis_colors[
                    self.error_vis_type.get(error, "other")
                ]
            except KeyError as e:

                selected, success, u_fp, u_fn, u_merge, u_split, u_gt = [
                    int(x) for x in attrs["details"][threshold_index]
                ]
                raise KeyError(
                    f"error (filtered, success, fp, fn, merge, split): {error}"
                )

            all_colors.append(error_vis_color)
            all_colors.append(error_vis_color)

            # from index
            u_edge = 2 * i
            v_edge = 2 * i + 1
            # from graph_attr
            u_vertex_position = self.graph.nodes[u]["location"] / tuple(self.voxel_size)
            v_vertex_position = self.graph.nodes[v]["location"] / tuple(self.voxel_size)
            # both from mst
            u_source = self.source_colors[e_gt]
            v_source = self.source_colors[e_gt]
            # yes
            u_gt_edge = e_gt
            v_gt_edge = e_gt
            # no, has to be done on a per node basis
            u_false = self.false_colors[e_fn]
            v_false = self.false_colors[e_fn]
            u_false_edge = e_fp or e_fn
            v_false_edge = e_fp or e_fn
            # yes if both match to different labels
            u_merge_edge = e_merge
            v_merge_edge = e_merge
            u_merge = self.merge_colors[e_merge]
            v_merge = self.merge_colors[e_merge]
            # no
            u_split = self.split_colors[e_split]
            v_split = self.split_colors[e_split]
            u_split_edge = e_split
            v_split_edge = e_split

            edges.append(u_edge)
            edges.append(v_edge)
            vertex_positions.append(u_vertex_position)
            vertex_positions.append(v_vertex_position)
            source.append(u_source)
            source.append(v_source)
            gt_edge.append(u_gt_edge)
            gt_edge.append(v_gt_edge)
            false_match.append(u_false)
            false_match.append(v_false)
            false_match_edge.append(u_false_edge)
            false_match_edge.append(v_false_edge)
            merge.append(u_merge)
            merge.append(v_merge)
            merge_edge.append(u_merge_edge)
            merge_edge.append(v_merge_edge)
            split.append(u_split)
            split.append(v_split)
            split_edge.append(u_split_edge)
            split_edge.append(v_split_edge)
        i += 1

        for j, (n, attrs) in enumerate(self.graph.nodes.items()):
            node_edge.append(1)
            node_edge.append(1)

            selected, success, n_fp, n_fn, n_merge, n_split, n_gt = [
                int(x) for x in attrs["details"][threshold_index]
            ]

            if selected and n_gt:
                assert success or n_fn or n_split
            elif selected:
                assert success or n_fp or n_merge

            selected_edge.append(selected)
            selected_edge.append(selected)

            success_edge.append(success)
            success_edge.append(success)

            component_matchings.append(
                component_colors.setdefault(
                    tuple(attrs["label_pair"][threshold_index]), self.random_color()
                )
            )
            component_matchings.append(
                component_colors.setdefault(
                    tuple(attrs["label_pair"][threshold_index]), self.random_color()
                )
            )

            assert not (success and n_merge)
            assert not (success and n_split)

            # assert (
            #     sum([1 - selected, success, n_fp, n_fn, n_merge, n_split]) == 1
            # ), f"{[1 - selected, success, n_fp, n_fn, n_merge, n_split]}"
            error_vis_color = self.error_vis_colors[
                self.error_vis_type[
                    (1 - selected, success, n_fp, n_fn, n_merge, n_split)
                ]
            ]
            all_colors.append(error_vis_color)
            all_colors.append(error_vis_color)

            # from index
            u_edge = 2 * (i + j)
            v_edge = 2 * (i + j) + 1
            # from graph_attr
            u_vertex_position = self.graph.nodes[n]["location"] / tuple(self.voxel_size)
            v_vertex_position = self.graph.nodes[n]["location"] / tuple(self.voxel_size)
            # both from mst
            u_source = self.source_colors[n_gt]
            v_source = self.source_colors[n_gt]
            # yes
            u_gt_edge = n_gt
            v_gt_edge = n_gt

            u_false = self.false_colors[n_fn]
            v_false = self.false_colors[n_fn]
            u_false_edge = n_fp or n_fn
            v_false_edge = n_fp or n_fn
            # yes if both match to different labels
            u_merge_edge = n_merge
            v_merge_edge = n_merge
            u_merge = self.merge_colors[n_merge]
            v_merge = self.merge_colors[n_merge]
            # no
            u_split = self.split_colors[n_split]
            v_split = self.split_colors[n_split]
            u_split_edge = n_split
            v_split_edge = n_split

            edges.append(u_edge)
            edges.append(v_edge)
            vertex_positions.append(u_vertex_position)
            vertex_positions.append(v_vertex_position)
            source.append(u_source)
            source.append(v_source)
            gt_edge.append(u_gt_edge)
            gt_edge.append(v_gt_edge)
            false_match.append(u_false)
            false_match.append(v_false)
            false_match_edge.append(u_false_edge)
            false_match_edge.append(v_false_edge)
            merge.append(u_merge)
            merge.append(v_merge)
            merge_edge.append(u_merge_edge)
            merge_edge.append(v_merge_edge)
            split.append(u_split)
            split.append(v_split)
            split_edge.append(u_split_edge)
            split_edge.append(v_split_edge)
        i += 1

        return neuroglancer.skeleton.Skeleton(
            vertex_positions=vertex_positions,
            edges=edges,
            vertex_attributes=dict(
                success_edge=success_edge,
                source=np.array(source),
                gt_edge=np.array(gt_edge),
                false_match=false_match,
                false_match_edge=false_match_edge,
                merge=merge,
                merge_edge=merge_edge,
                split=split,
                split_edge=split_edge,
                node_edge=node_edge,
                selected_edge=selected_edge,
                component_matchings=component_matchings,
                all_colors=all_colors,
            ),
            # edge_attribues=dict(distances=distances),
        )


def build_trees(node_ids, locations, edges, node_attrs=None, edge_attrs=None):
    if node_attrs is None:
        node_attrs = {}
    if edge_attrs is None:
        edge_attrs = {}

    node_to_index = {n: i for i, n in enumerate(node_ids)}
    trees = nx.Graph()
    pbs = {
        int(node_id): node_location
        for node_id, node_location in zip(node_ids, locations)
    }
    for i, row in enumerate(edges):
        u = node_to_index.get(int(row[0]), -1)
        v = node_to_index.get(int(row[-1]), -1)

        e_attrs = {attr: values[i] for attr, values in edge_attrs.items()}

        if u == -1 or v == -1:
            continue

        pos_u = daisy.Coordinate(tuple(pbs[node_ids[u]]))
        pos_v = daisy.Coordinate(tuple(pbs[node_ids[v]]))

        if node_ids[u] not in trees.nodes:
            u_attrs = {attr: values[u] for attr, values in node_attrs.items()}
            trees.add_node(node_ids[u], location=pos_u, **u_attrs)
        if node_ids[v] not in trees.nodes:
            v_attrs = {attr: values[v] for attr, values in node_attrs.items()}
            trees.add_node(node_ids[v], location=pos_v, **v_attrs)

        trees.add_edge(node_ids[u], node_ids[v], **e_attrs)
    return trees


def get_embedding_mst(embedding, alpha, coordinate_scale, offset, candidates):

    _, depth, height, width = embedding.shape
    coordinates = np.meshgrid(
        np.arange(0, (depth - 0.5) * coordinate_scale[0], coordinate_scale[0]),
        np.arange(0, (height - 0.5) * coordinate_scale[1], coordinate_scale[1]),
        np.arange(0, (width - 0.5) * coordinate_scale[2], coordinate_scale[2]),
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
        pos_u = embedding[u][-3:] / coordinate_scale
        v = int(v)
        pos_v = embedding[v][-3:] / coordinate_scale
        mst.add_node(u, location=(pos_u + offset) * np.array([1000, 300, 300]))
        mst.add_node(v, location=(pos_v + offset) * np.array([1000, 300, 300]))
        if alpha > distance:
            mst.add_edge(u, v, d=distance)
    for node, attrs in mst.nodes.items():
        assert "location" in attrs
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
            trees.add_node(edge[0], location=pos_u)
        else:
            assert trees.nodes[edge[0]]["location"] == pos_u, "locations don't match"
        if edge[1] not in trees.nodes:
            trees.add_node(edge[1], location=pos_v)
        else:
            assert trees.nodes[edge[1]]["location"] == pos_v, "locations don't match"
        trees.add_edge(edge[0], edge[1], d=edge[2])
    return trees


def add_trees_no_skeletonization(
    s, trees, node_id, name, dimensions, visible=False, color=None
):
    mst = []
    for u, v in trees.edges():
        pos_u = np.array(trees.nodes[u]["location"]) + 0.5
        pos_v = np.array(trees.nodes[v]["location"]) + 0.5
        mst.append(
            neuroglancer.LineAnnotation(
                point_a=pos_u[::-1], point_b=pos_v[::-1], id=next(node_id)
            )
        )

    s.layers.append(
        name="{}".format(name),
        layer=neuroglancer.AnnotationLayer(annotations=mst),
        annotationColor="#{:02X}{:02X}{:02X}".format(255, 125, 125),
        visible=visible,
    )


def add_layer(context, array, name, visible=True, **kwargs):
    array_dims = len(array.shape)
    voxel_size = array.voxel_size
    attrs = {
        2: {"names": ["y", "x"], "units": "nm", "scales": voxel_size},
        3: {"names": ["z", "y", "x"], "units": "nm", "scales": voxel_size},
        4: {
            "names": ["c^", "z", "y", "x"],
            "units": ["", "nm", "nm", "nm"],
            "scales": [1, *voxel_size],
        },
    }
    dimensions = neuroglancer.CoordinateSpace(**attrs[array_dims])
    offset = np.array((0,) * (array_dims - 3) + array.roi.get_offset())
    offset = offset // attrs[array_dims]["scales"]
    # if len(offset) == 2:
    #     offset = (0,) + tuple(offset)

    if len(array.shape) > 3 and array.shape[0] > 3:
        pca = PCA(n_components=3)
        flattened = array.to_ndarray().reshape(array.shape[0], -1).T
        fitted = pca.fit_transform(flattened).T
        array.data = fitted.reshape((3,) + array.shape[1:])

    d = np.asarray(array.data)
    if array.data.dtype == np.dtype(bool):
        array.data = np.array(d, dtype=np.float32)

    channels = ",".join(
        [
            f"toNormalized(getDataValue({i}))" if i < array.shape[0] else "0"
            for i in range(3)
        ]
    )
    shader_4d = (
        """
void main() {
  emitRGB(vec3(%s));
}
"""
        % channels
    )
    shader_3d = None

    layer = neuroglancer.LocalVolume(
        data=array.data, dimensions=dimensions, voxel_offset=tuple(offset)
    )

    if array.data.dtype == np.dtype(np.uint64):
        context.layers.append(name=name, layer=layer, visible=visible)
    else:
        context.layers.append(
            name=name,
            layer=layer,
            visible=visible,
            shader=shader_4d if array_dims == 4 else shader_3d,
            **kwargs,
        )


def add_trees(
    s,
    trees,
    node_id,
    name,
    dimensions,
    offset,
    shape,
    visible=False,
    voxel_size=(1000, 300, 300),
):
    if trees is None:
        return None

    details = False
    for node, attrs in trees.nodes.items():
        if "details" in attrs:
            details = True
        else:
            details = False
        break

    if details:
        s.layers.append(
            name="{}".format(name),
            layer=neuroglancer.SegmentationLayer(
                source=[
                    neuroglancer.LocalVolume(
                        data=np.ones(shape, dtype=np.uint32),
                        dimensions=dimensions,
                        voxel_offset=offset,
                    ),
                    MatchSource(trees, dimensions, voxel_size=voxel_size),
                ],
                skeleton_shader="""
// coloring options:
// source
// distance
// matching
// false_match
// merge
// split

void main() {
  if (error_vis_edge < 0.4) discard;
  emitRGB(vec3(error_vis));
  
  //if (source_edge < 0.5) discard;
  //emitRGB(vec3(matching));

  //if (abs(source_edge-1.0) > 0.5) discard;
  //emitRGB(colormapJet(distance));
}
""",
                selected_alpha=0,
                not_selected_alpha=0,
            ),
        )
    else:
        s.layers.append(
            name="{}".format(name),
            layer=neuroglancer.SegmentationLayer(
                source=[
                    neuroglancer.LocalVolume(
                        data=np.ones(shape, dtype=np.uint32),
                        dimensions=dimensions,
                        voxel_offset=offset,
                    ),
                    SkeletonSource(trees, dimensions, voxel_size=voxel_size),
                ],
                skeleton_shader="""
#uicontrol float showautapse slider(min=0, max=2)

void main() {
if (distance > showautapse) discard;
emitRGB(colormapJet(distance));
}
""",
                selected_alpha=0,
                not_selected_alpha=0,
            ),
        )


def dimensions_from_volume(array):
    voxel_size = array.voxel_size
    spatial_dims = 3
    attrs = {"names": ["z", "y", "x"], "units": "nm", "scales": voxel_size}
    dimensions = neuroglancer.CoordinateSpace(**attrs)
    offset = np.array(array.roi.get_offset())
    offset = offset // attrs["scales"]
    return dimensions, offset, array.shape[-spatial_dims:]


def dimensions_from_guess(dataset, graph):
    locations = dataset[f"{graph}-locations"][()]
    lower = locations.min(axis=0)
    upper = locations.max(axis=0)
    voxel_size = [1000, 300, 300]
    lower -= lower % voxel_size
    upper += voxel_size - upper % voxel_size
    shape = ((upper - lower) / voxel_size).astype(int)

    attrs = {"names": ["z", "y", "x"], "units": "nm", "scales": voxel_size}
    dimensions = neuroglancer.CoordinateSpace(**attrs)

    offset = np.array(lower)
    offset = offset // voxel_size
    return dimensions, offset, shape


def dims_from_guess(graph):
    locations = np.array([attrs["location"] for attrs in graph.nodes.values()])
    try:
        lower = locations.min(axis=0)
        upper = locations.max(axis=0)
    except:
        lower = np.array([0, 0, 0])
        upper = np.array([0, 0, 0])
    voxel_size = [1000, 300, 300]
    lower -= lower % voxel_size
    upper += voxel_size - upper % voxel_size
    shape = ((upper - lower) / voxel_size).astype(int)

    attrs = {"names": ["z", "y", "x"], "units": "nm", "scales": voxel_size}
    dimensions = neuroglancer.CoordinateSpace(**attrs)

    offset = np.array(lower)
    offset = offset // voxel_size
    return dimensions, offset, shape


def grid_from_graph_guess(graph):
    locations = np.array([attrs["location"] for attrs in graph.nodes.values()])
    lower = locations.min(axis=0)
    upper = locations.max(axis=0)
    voxel_size = [1000, 300, 300]
    lower -= lower % voxel_size
    upper += voxel_size - upper % voxel_size
    shape = ((upper - lower) / voxel_size).astype(int)

    offset = np.array(lower)
    offset = offset // voxel_size
    return offset, shape


def dimensions_from_graph_guess():
    attrs = {"names": ["z", "y", "x"], "units": "nm", "scales": (1,) * 3}
    dimensions = neuroglancer.CoordinateSpace(**attrs)
    return dimensions


def get_volumes(h5_file, path):
    datasets = []
    try:
        for key in h5_file.get(path, {}).keys():
            datasets += get_volumes(h5_file, f"{path}/{key}")
        return datasets
    except AttributeError:
        return [path]


def get_graphs(h5_file, path):
    datasets = []
    path_components = path.split("/")
    path = "/".join(path_components[:-1])
    name = path_components[-1]

    try:
        for key in h5_file.get(path, {}).keys():
            if key == "node_attrs" or key == "edge_attrs":
                datasets += [path]
                continue
            if name in key:
                for g in h5_file.get(f"{path}/{name}").keys():
                    datasets += get_graphs(h5_file, f"{path}/{name}/{g}")
        return datasets
    except AttributeError:
        return [f"{path}/{name}"]
    except ValueError:
        return []


def add_snapshot(
    context,
    snapshot_file,
    name_prefix="",
    volume_paths=["volumes"],
    graph_paths=["points"],
    graph_node_attrs=None,
    graph_edge_attrs=None,
    # mst=["embedding", "fg_maxima"],
    mst=None,
    roi=None,
):
    if snapshot_file.name.endswith(".hdf"):
        f = h5py.File(snapshot_file, "r")
    elif snapshot_file.name.endswith(".n5"):
        f = zarr.open(str(snapshot_file.absolute()), "r")
    elif snapshot_file.name.endswith(".zarr"):
        f = zarr.open(str(snapshot_file.absolute()), "r")
    with f as dataset:
        volumes = []
        for volume in volume_paths:
            volumes += get_volumes(dataset, volume)
        graphs = set()
        for graph in graph_paths:
            graph_datasets = get_graphs(dataset, graph)
            graphs |= set([p.split("-")[0] for p in graph_datasets])
            print(f"got graphs: {graphs}")
        if graph_node_attrs is None:
            graph_node_attrs = {}
        if graph_edge_attrs is None:
            graph_edge_attrs = {}

        node_id = itertools.count(start=1)

        v = None
        for volume in volumes:
            v = daisy.open_ds(str(snapshot_file.absolute()), f"{volume}")
            if roi is not None:
                v = v.intersect(roi)
            if v.dtype == np.int64:
                v.materialize()
                v.data = v.data.astype(np.uint64)
            add_layer(context, v, f"{name_prefix}_{volume}", visible=False)

        if v is not None:
            dimensions, voxel_offset, voxel_shape = dimensions_from_volume(v)
        elif len(graphs) > 0:
            dimensions, voxel_offset, voxel_shape = dimensions_from_guess(
                dataset, list(graphs)[0]
            )

        if mst is not None:
            embeddings = daisy.open_ds(
                str(snapshot_file.absolute()), f"volumes/{mst[0]}"
            )
            offset = embeddings.roi.get_offset() / (1000, 300, 300)
            mst = get_embedding_mst(
                embeddings.data,
                1,
                (0.05, 0.015, 0.015),
                offset,
                daisy.open_ds(
                    str(snapshot_file.absolute()), f"volumes/{mst[1]}"
                ).to_ndarray(),
            )
            # add_trees(
            #     context,
            #     mst,
            #     node_id,
            #     name="MST",
            #     visible=True,
            #     dimensions=dimensions,
            #     offset=offset,
            #     shape=v.data.shape[-len(offset) :],
            # )
            v.materialize()
            v.data = (v.data + 1) / 2

        # graphs should not be dependent on volumes
        for graph in graphs:
            node_ids = dataset[f"{graph}-ids"]
            locations = dataset[f"{graph}-locations"]
            edges = dataset[f"{graph}-edges"]
            node_attrs = {}
            logger.info(
                f"Adding graph {graph} with {node_ids.shape} nodes and {edges.shape} edges"
            )
            try:
                for attr in dataset[f"{graph}/node_attrs"].keys():
                    node_attrs[attr] = dataset[f"{graph}/node_attrs/{attr}"]
            except KeyError:
                pass
            edge_attrs = {}
            try:
                for attr in dataset[f"{graph}/edge_attrs"].keys():
                    edge_attrs[attr] = dataset[f"{graph}/edge_attrs/{attr}"]
            except KeyError:
                pass
            components = build_trees(node_ids, locations, edges, node_attrs, edge_attrs)
            logger.info(
                f"Adding graph {graph} with {len(list(nx.connected_components(components)))} connected components"
            )
            add_trees(
                context,
                components,
                node_id,
                dimensions=dimensions,
                offset=voxel_offset,
                shape=voxel_shape,
                name=f"{name_prefix}_{graph}",
                visible=True,
            )


def add_match_layers(
    context,
    mst,
    gt,
    label_matchings,
    node_matchings,
    node_labels_mst,
    node_labels_gt,
    name="matching",
):
    dimensions, voxel_offset, voxel_shape = dims_from_guess(mst)
    graph_dims = dimensions_from_graph_guess()
    context.layers.append(
        name="{}".format(name),
        layer=neuroglancer.SegmentationLayer(
            source=[
                neuroglancer.LocalVolume(
                    data=np.ones(voxel_shape, dtype=np.uint32),
                    dimensions=dimensions,
                    voxel_offset=voxel_offset,
                ),
                MatchSource(
                    mst,
                    gt,
                    graph_dims,
                    label_matchings,
                    node_matchings,
                    node_labels_mst,
                    node_labels_gt,
                ),
            ],
            skeleton_shader="""
// coloring options:
// source
// distance
// matching
// false_match
// merge
// split

void main() {
  if (error_vis_edge < 0.4) discard;
  emitRGB(vec3(error_vis));
  
  //if (source_edge < 0.5) discard;
  //emitRGB(vec3(matching));

  //if (abs(source_edge-1.0) > 0.5) discard;
  //emitRGB(colormapJet(distance));
}
""",
            selected_alpha=0,
            not_selected_alpha=0,
        ),
    )


def add_mst_snapshot_with_stats(
    context,
    matching_data,
    mst_snapshot,
    gt,
    threshold_inds=None,
    graph_node_attrs=None,
    graph_edge_attrs=None,
    false_pos_threshold=None,
):
    if false_pos_threshold is None:
        false_pos_threshold = 0
    mst = tree_from_snapshot(
        mst_snapshot,
        "mst",
        graph_node_attrs=graph_node_attrs,
        graph_edge_attrs=graph_edge_attrs,
    )

    label_matchings, node_matchings, node_labels_mst, node_labels_gt, thresholds = (
        matching_data
    )
    edges_to_add = list(mst.edges.items())

    thresholded_graph = nx.Graph()
    for i, threshold in enumerate(thresholds):
        for j, ((u, v), attrs) in reversed(list(enumerate(edges_to_add))):
            if attrs["distance"] < threshold:
                thresholded_graph.add_node(u, **mst.nodes[u])
                thresholded_graph.add_node(v, **mst.nodes[v])
                thresholded_graph.add_edge(u, v, **attrs)
                del edges_to_add[j]

        if i in threshold_inds:

            temp = copy.deepcopy(thresholded_graph)

            false_pos_nodes = []
            for cc in nx.connected_components(temp):
                cc_graph = temp.subgraph(cc)
                min_loc = None
                max_loc = None
                for node, attrs in cc_graph.nodes.items():
                    node_loc = attrs["location"]
                    if min_loc is None:
                        min_loc = node_loc
                    else:
                        min_loc = np.min(np.array([node_loc, min_loc]), axis=0)
                    if max_loc is None:
                        max_loc = node_loc
                    else:
                        max_loc = np.max(np.array([node_loc, max_loc]), axis=0)
                if np.linalg.norm(min_loc - max_loc) < false_pos_threshold:
                    false_pos_nodes += list(cc)
            for node in false_pos_nodes:
                temp.remove_node(node)

            nodes_x = list(temp.nodes)
            nodes_y = list(gt.nodes)

            node_labels_x = {
                node: component
                for component, nodes in enumerate(nx.connected_components(temp))
                for node in nodes
            }

            node_labels_y = {
                node: component
                for component, nodes in enumerate(nx.connected_components(gt))
                for node in nodes
            }

            edges_yx = get_edges_xy(
                gt, temp, location_attr="location", node_match_threshold=4000
            )
            edges_xy = [(v, u) for u, v in edges_yx]

            (label_matches, node_matches, splits, merges, fps, fns) = match_components(
                nodes_x, nodes_y, edges_xy, node_labels_x, node_labels_y
            )

            erl, details = psudo_graph_edit_distance(
                node_matches,
                node_labels_x,
                node_labels_y,
                temp,
                gt,
                "location",
                node_spacing=5000,
                details=True,
            )

            add_match_layers(
                context,
                temp,
                gt,
                label_matchings[i],
                node_matchings[i],
                node_labels_mst[i],
                node_labels_gt[i],
                name=f"Matched-{threshold:.3f}",
            )


def tree_from_snapshot(
    snapshot_file, graph, graph_node_attrs=None, graph_edge_attrs=None
):
    with h5py.File(snapshot_file, "r") as dataset:
        if graph_node_attrs is None:
            graph_node_attrs = []
        if graph_edge_attrs is None:
            graph_edge_attrs = []
        node_ids = dataset["points"][f"{graph}-ids"]
        locations = dataset["points"][f"{graph}-locations"]
        edges = dataset["points"][f"{graph}-edges"]
        node_attrs = {}
        for attr in graph_node_attrs:
            node_attrs[attr] = dataset["points"][f"{graph}/node_attrs/{attr}"]
        edge_attrs = {}
        for attr in graph_edge_attrs:
            edge_attrs[attr] = dataset["points"][f"{graph}/edge_attrs/{attr}"]
        components = build_trees(node_ids, locations, edges, node_attrs, edge_attrs)
    return components


def visualize_hdf5(hdf5_file: Path, dimensions, mst=None, maxima_for=None, skip=None):
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
    viewer.dimensions = dimensions
    with viewer.txn() as s:
        for volume in volumes:
            if skip == volume:
                continue
            v = daisy.open_ds(str(hdf5_file.absolute()), f"volumes/{volume}")
            if volume == "embedding":
                v.materialize()
                v.data = (v.data + 1) / 2
            if len(v.data.shape) == 5:
                v.materialize()
                v.data = v.data[0]
                v.n_channel_dims -= 1
            if v.dtype == np.int64:
                v.materialize()
                v.data = v.data.astype(np.uint64)
            if volume == maxima_for:
                v.materialize()
                max_filtered = maximum_filter(v.data, (3, 10, 10))
                maxima = np.logical_and(max_filtered == v.data, v.data > 0.01).astype(
                    np.float32
                )
                m = daisy.Array(maxima, v.roi, v.voxel_size)
                add_layer(s, m, f"{volume}-maxima")
            if mst is not None and volume == mst[0]:
                offset = v.roi.get_offset() / voxel_size
                mst = get_embedding_mst(
                    v.data,
                    1,
                    coordinate_scale,
                    offset,
                    daisy.open_ds(
                        str(hdf5_file.absolute()), f"volumes/{mst[1]}"
                    ).to_ndarray(),
                )
                add_trees(
                    s,
                    mst,
                    node_id,
                    name="MST",
                    visible=True,
                    dimensions=dimensions,
                    offset=offset,
                    shape=v.data.shape[-len(offset) :],
                )
                v.materialize()
                v.data = (v.data + 1) / 2
            add_layer(s, v, volume, visible=False)

        for point_set in points:
            node_ids = dataset["points"][f"{point_set}-ids"]
            locations = dataset["points"][f"{point_set}-locations"]
            edges = dataset["points"][f"{point_set}-edges"]
            components = build_trees(node_ids, locations, edges)
            add_trees(
                s,
                components,
                node_id,
                name=point_set,
                visible=False,
                dimensions=dimensions,
            )
    print(viewer)
    input("Hit ENTER to quit!")
