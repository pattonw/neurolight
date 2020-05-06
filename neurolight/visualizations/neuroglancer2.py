import daisy
import neuroglancer
import numpy as np
import sys
import itertools
import h5py
import networkx as nx
import json
from pathlib import Path
import copy
import random

from scipy.ndimage.filters import maximum_filter

from neurolight.pipelines import DEFAULT_CONFIG

# from funlib.show.neuroglancer import add_layer
import mlpack as mlp


class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, cc, dimensions, voxel_size, node_attrs=None, edge_attrs=None):
        super(SkeletonSource, self).__init__(dimensions)
        self.cc = cc
        self.voxel_size = voxel_size
        for node, attrs in self.cc.nodes.items():
            assert "location" in attrs

    def get_skeleton(self, i):

        edges = []
        vertex_positions = []
        for i, (u, v) in enumerate(self.cc.edges):
            vertex_positions.append(
                self.cc.nodes[u]["location"] / daisy.Coordinate(self.voxel_size)
            )
            vertex_positions.append(
                self.cc.nodes[v]["location"] / daisy.Coordinate(self.voxel_size)
            )
            edges.append(2 * i)
            edges.append(2 * i + 1)

        return neuroglancer.skeleton.Skeleton(
            vertex_positions=vertex_positions,
            edges=edges,
            vertex_attributes=dict(),
            # edge_attribues=dict(distances=distances),
        )


class MatchSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(
        self,
        mst,
        gt,
        dimensions,
        label_matchings,
        node_matchings,
        node_labels_mst,
        node_labels_gt,
        node_attrs=None,
        edge_attrs=None,
    ):
        super(MatchSource, self).__init__(dimensions)
        self.vertex_attributes["source"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "source_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["distance"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        # self.vertex_attributes[
        #     "distance_edge"
        # ] = neuroglancer.skeleton.VertexAttributeInfo(
        #     data_type=np.float32, num_components=1
        # )
        self.vertex_attributes["matching"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "matching_edge"
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
        self.vertex_attributes[
            "gt_and_pred_components"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "gt_components"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.vertex_attributes[
            "error_vis_edge"
        ] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=1
        )
        self.vertex_attributes["error_vis"] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32, num_components=3
        )
        self.mst = mst
        self.nodes_x = [node for node in self.mst]
        self.labels_x = node_labels_mst
        self.gt = gt
        self.nodes_y = [node for node in self.gt]
        self.labels_y = node_labels_gt
        self.label_matchings = label_matchings
        self.node_matchings = node_matchings
        self.x_node_to_y_label = {}
        self.y_node_to_x_label = {}
        for (u, v) in self.node_matchings:
            labels = self.x_node_to_y_label.setdefault(u, set())
            labels.add(self.labels_y[v])
            labels = self.y_node_to_x_label.setdefault(v, set())
            labels.add(self.labels_x[u])

        nomatch_node = max(self.nodes_x + self.nodes_y) + 1

        for (u, v) in self.node_matchings:
            if u != nomatch_node:
                if v != nomatch_node:
                    assert (self.labels_x[u], self.labels_y[v]) in self.label_matchings

        self.check_ids()

        self.init_colors()

    def check_ids(self):
        no_label = max(max(self.labels_y.values()), max(self.labels_x.values())) + 1
        no_node = max(max(self.nodes_x), max(self.nodes_y)) + 1

        for labels_u, labels_v in self.label_matchings:
            assert labels_u == no_label or labels_u in self.labels_x.values()
        for labels_u, labels_v in self.label_matchings:
            assert labels_v == no_label or labels_v in self.labels_y.values()
        for node_u, node_v in self.node_matchings:
            assert node_u in self.nodes_x
        for node_u, node_v in self.node_matchings:
            assert node_v in self.nodes_y

    def init_colors(self):
        red = np.array((255, 128, 128)) / 256
        green = np.array((128, 255, 128)) / 256
        blue = np.array((128, 128, 255)) / 256
        yellow = np.array((255, 255, 128)) / 256
        purple = np.array((255, 128, 255)) / 256
        self.error_vis_colors = {
            "split": red,
            "merge": blue,
            "true_pos": green,
            "false_neg": yellow,
            "false_pos": purple,
        }
        self.source_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.distance_color_range = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.matching_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.false_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.merge_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.split_colors = np.array([(255, 128, 128), (128, 128, 255)]) / 256
        self.matching_colors = {
            label_match: (
                random.randrange(256) / 256,
                random.randrange(256) / 256,
                random.randrange(256) / 256,
            )
            for label_match in self.label_matchings
        }
        self.gt_component_colors = {
            component: (
                random.randrange(256) / 256,
                random.randrange(256) / 256,
                random.randrange(256) / 256,
            )
            for component in set(self.labels_y.values())
        }

    def is_no_match_label(self, label):
        return (
            label == max(max(self.labels_y.values()), max(self.labels_x.values())) + 1
        )

    def is_no_match_node(self, node):
        return node == max(max(self.nodes_x), max(self.nodes_y)) + 1

    def get_skeleton(self, i):

        # edges only show up if both of their end points satisfy the condition
        # in shader. i.e. *_edge > 0.5

        # source_edge, an edge from mst to mst or gt to gt
        #   simple 1 node per node
        # distance_edge, an edge in mst, labeled with color
        #   needs individual nodes per edge to annotate with distance
        # matching edge, an edge in mst or gt labeled with color
        #   edge in mst or gt where labels same on either endpoint
        # false_edge, self edge on mst or gt without label match
        # merge_edge, edge in mst connecting two nodes matched to different labels
        # split_edge, edge in gt connecting two nodes matched to different labels

        edges = []
        vertex_positions = []
        distances = []
        source = []
        source_edge = []
        distance = []
        distance_edge = []
        matching = []
        matching_edge = []
        false = []
        false_edge = []
        merge = []
        merge_edge = []
        split = []
        split_edge = []
        gt_and_pred_components = []
        gt_components = []
        error_vis = []
        error_vis_edge = []

        # get distance edges
        for i, ((u, v), attrs) in enumerate(self.mst.edges.items()):

            gt_u_labels = self.x_node_to_y_label.get(u, [])
            gt_v_labels = self.x_node_to_y_label.get(v, [])
            mst_u_label = self.labels_x[u]
            mst_v_label = self.labels_x[v]

            if len(gt_u_labels) == 1 and len(gt_v_labels) == 1:
                if (mst_u_label, list(gt_u_labels)[0]) not in self.matching_colors:
                    raise Exception()

            gt_u_label = list(gt_u_labels)[0] if len(gt_u_labels) == 1 else None
            gt_v_label = list(gt_v_labels)[0] if len(gt_v_labels) == 1 else None

            # from index
            u_edge = 2 * i
            v_edge = 2 * i + 1
            # from graph_attr
            u_vertex_position = self.mst.nodes[u]["location"]
            v_vertex_position = self.mst.nodes[v]["location"]
            # from edge attr
            u_distance = attrs["distance"]
            v_distance = attrs["distance"]
            # both from mst
            u_source = self.source_colors[0]
            v_source = self.source_colors[0]
            # yes
            u_source_edge = 1
            v_source_edge = 1
            # distance edge attr so both same
            u_distance = attrs["distance"]
            v_distance = attrs["distance"]
            # yes, from mst
            u_distance_edge = 1
            v_distance_edge = 1
            # yes if both match to same label
            u_matching = self.matching_colors.get(
                (mst_u_label, gt_u_label), (0.6, 0, 0)
            )
            v_matching = self.matching_colors.get(
                (mst_v_label, gt_v_label), (0.6, 0, 0)
            )
            u_matching_edge = (
                1
                if gt_u_label is not None
                and gt_v_label is not None
                and gt_u_label == gt_v_label
                else 0
            )
            v_matching_edge = (
                1
                if gt_u_label is not None
                and gt_v_label is not None
                and gt_u_label == gt_v_label
                else 0
            )
            # no, has to be done on a per node basis
            u_false = self.false_colors[0]
            v_false = self.false_colors[0]
            u_false_edge = 0
            v_false_edge = 0
            # yes if both match to different labels
            u_merge_edge = (
                1
                if gt_u_label is not None
                and gt_v_label is not None
                and gt_u_label != gt_v_label
                else 0
            )
            v_merge_edge = (
                1
                if gt_u_label is not None
                and gt_v_label is not None
                and gt_u_label != gt_v_label
                else 0
            )
            u_merge = self.merge_colors[u_merge_edge]
            v_merge = self.merge_colors[v_merge_edge]
            # no
            u_split = self.split_colors[0]
            v_split = self.split_colors[0]
            u_split_edge = 0
            v_split_edge = 0

            # error vis:
            # an mst edge is part of the error vis if it merges two nodes
            # with seperate ids, is part of the true pos (sometimes) or both
            # nodes are unmatched.
            error_vis_merge_edge = ((gt_u_label is None) != (gt_v_label is None)) or (
                gt_u_label is not None and gt_u_label != gt_v_label
            )
            true_pos_edge = (
                gt_u_label is not None
                and gt_v_label is not None
                and gt_u_label == gt_v_label
            )
            false_pos_edge = gt_u_label is None and gt_v_label is None
            assert error_vis_merge_edge + true_pos_edge + false_pos_edge <= 1
            if true_pos_edge:
                color = self.error_vis_colors["true_pos"]
            elif error_vis_merge_edge:
                color = self.error_vis_colors["merge"]
            elif false_pos_edge:
                color = self.error_vis_colors["false_pos"]
            else:
                color = (0, 0, 0)
            u_error_vis_edge = (
                error_vis_merge_edge + false_pos_edge + 0.5 * true_pos_edge
            )
            v_error_vis_edge = u_error_vis_edge
            u_error_vis = color
            v_error_vis = color

            edges.append(u_edge)
            edges.append(v_edge)
            vertex_positions.append(u_vertex_position)
            vertex_positions.append(v_vertex_position)
            source.append(u_source)
            source.append(v_source)
            source_edge.append(u_source_edge)
            source_edge.append(v_source_edge)
            distance.append(u_distance)
            distance.append(v_distance)
            distance_edge.append(u_distance_edge)
            distance_edge.append(v_distance_edge)
            matching.append(u_matching)
            matching.append(v_matching)
            matching_edge.append(u_matching_edge)
            matching_edge.append(v_matching_edge)
            false.append(u_false)
            false.append(v_false)
            false_edge.append(u_false_edge)
            false_edge.append(v_false_edge)
            merge.append(u_merge)
            merge.append(v_merge)
            merge_edge.append(u_merge_edge)
            merge_edge.append(v_merge_edge)
            split.append(u_split)
            split.append(v_split)
            split_edge.append(u_split_edge)
            split_edge.append(v_split_edge)
            gt_and_pred_components.append(u_matching)
            gt_and_pred_components.append(v_matching)
            gt_components.append(u_source)
            gt_components.append(v_source)
            distances.append(u_distance)
            distances.append(v_distance)
            error_vis_edge.append(u_error_vis_edge)
            error_vis_edge.append(v_error_vis_edge)
            error_vis.append(u_error_vis)
            error_vis.append(v_error_vis)
        i += 1

        for j, ((u, v), attrs) in enumerate(self.gt.edges.items()):

            gt_u_label = self.labels_y[u]
            gt_v_label = self.labels_y[v]
            mst_u_labels = self.y_node_to_x_label.get(u, [])
            mst_v_labels = self.y_node_to_x_label.get(v, [])

            mst_u_label = list(mst_u_labels)[0] if len(mst_u_labels) == 1 else None
            mst_v_label = list(mst_v_labels)[0] if len(mst_v_labels) == 1 else None

            # from index
            u_edge = 2 * (i + j)
            v_edge = 2 * (i + j) + 1
            # from graph_attr
            u_vertex_position = self.gt.nodes[u]["location"]
            v_vertex_position = self.gt.nodes[v]["location"]
            # from edge attr
            u_distance = 0
            v_distance = 0
            # both from mst
            u_source = self.source_colors[1]
            v_source = self.source_colors[1]
            # yes
            u_source_edge = 2
            v_source_edge = 2
            # distance edge attr so both same
            u_distance = 0
            v_distance = 0
            # no, from gt
            u_distance_edge = 0
            v_distance_edge = 0
            # yes if both match to same label
            u_matching = self.matching_colors.get(
                (mst_u_label, gt_u_label), (0.6, 0, 0)
            )
            v_matching = self.matching_colors.get(
                (mst_v_label, gt_v_label), (0.6, 0, 0)
            )
            u_matching_edge = (
                1
                if mst_u_label is not None
                and mst_v_label is not None
                and mst_u_label == mst_v_label
                else 0
            )
            v_matching_edge = (
                1
                if mst_u_label is not None
                and mst_v_label is not None
                and mst_u_label == mst_v_label
                else 0
            )
            # no, has to be done on a per node basis
            u_false = self.false_colors[0]
            v_false = self.false_colors[0]
            u_false_edge = 0
            v_false_edge = 0
            # yes if both match to different labels
            u_merge_edge = (
                1
                if mst_u_label is not None
                and mst_v_label is not None
                and mst_u_label != mst_v_label
                else 0
            )
            v_merge_edge = (
                1
                if mst_u_label is not None
                and mst_v_label is not None
                and mst_u_label != mst_v_label
                else 0
            )
            u_merge = self.merge_colors[u_merge_edge]
            v_merge = self.merge_colors[v_merge_edge]
            # no
            u_split = self.split_colors[0]
            v_split = self.split_colors[0]
            u_split_edge = 0
            v_split_edge = 0
            # gt_component_color
            u_component_color = self.gt_component_colors[self.labels_y[u]]
            v_component_color = self.gt_component_colors[self.labels_y[v]]

            # error vis:
            # an gt edge is part of the error vis if its nodes get
            # seperate ids or both nodes are unmatched.
            err_vis_split_edge = (mst_u_label is not None and mst_v_label is not None) and (
                mst_u_label != mst_v_label
            ) or ((mst_u_label is None) != (mst_v_label is None))
            false_neg_edge = mst_u_label is None and mst_v_label is None
            assert err_vis_split_edge + false_neg_edge <= 1
            if false_neg_edge:
                color = self.error_vis_colors["false_neg"]
            elif err_vis_split_edge:
                color = self.error_vis_colors["split"]
            else:
                color = (0, 0, 0)
            u_error_vis_edge = err_vis_split_edge + false_neg_edge
            v_error_vis_edge = u_error_vis_edge
            u_error_vis = color
            v_error_vis = color

            edges.append(u_edge)
            edges.append(v_edge)
            vertex_positions.append(u_vertex_position)
            vertex_positions.append(v_vertex_position)
            source.append(u_source)
            source.append(v_source)
            source_edge.append(u_source_edge)
            source_edge.append(v_source_edge)
            distance.append(u_distance)
            distance.append(v_distance)
            distance_edge.append(u_distance_edge)
            distance_edge.append(v_distance_edge)
            matching.append(u_matching)
            matching.append(v_matching)
            matching_edge.append(u_matching_edge)
            matching_edge.append(v_matching_edge)
            false.append(u_false)
            false.append(v_false)
            false_edge.append(u_false_edge)
            false_edge.append(v_false_edge)
            merge.append(u_merge)
            merge.append(v_merge)
            merge_edge.append(u_merge_edge)
            merge_edge.append(v_merge_edge)
            split.append(u_split)
            split.append(v_split)
            split_edge.append(u_split_edge)
            split_edge.append(v_split_edge)
            gt_and_pred_components.append(u_source)
            gt_and_pred_components.append(v_source)
            gt_components.append(u_component_color)
            gt_components.append(v_component_color)
            distances.append(u_distance)
            distances.append(v_distance)
            error_vis_edge.append(u_error_vis_edge)
            error_vis_edge.append(v_error_vis_edge)
            error_vis.append(u_error_vis)
            error_vis.append(v_error_vis)
        j += 1

        # get distance edges
        for k, (node, attrs) in enumerate(self.mst.nodes.items()):

            gt_u_labels = self.x_node_to_y_label.get(node, [])
            gt_u_label = list(gt_u_labels)[0] if len(gt_u_labels) == 1 else None
            mst_u_label = self.labels_x[node]

            # from index
            u_edge = 2 * (i + j + k)
            v_edge = 2 * (i + j + k) + 1
            # from graph_attr
            u_vertex_position = self.mst.nodes[node]["location"]
            # both from mst
            u_source = self.source_colors[0]
            u_source_edge = 0
            # distance edge attr so both same
            u_distance = 0
            # no
            u_distance_edge = 0
            # yes if both match to same label
            u_matching = (0, 0, 0)
            u_matching_edge = 0
            # no, has to be done on a per node basis
            u_false_edge = 1 if gt_u_label is None else 0
            u_false = self.false_colors[0]
            # yes if both match to different labels
            u_merge_edge = 0
            u_merge = self.merge_colors[u_merge_edge]
            # no
            u_split = self.split_colors[0]
            u_split_edge = 0

            u_error_vis_edge = 0
            v_error_vis_edge = 0
            u_error_vis = (0, 0, 0)
            v_error_vis = (0, 0, 0)

            edges.append(u_edge)
            edges.append(v_edge)
            vertex_positions.append(u_vertex_position)
            vertex_positions.append(u_vertex_position)
            source.append(u_source)
            source.append(u_source)
            source_edge.append(u_source_edge)
            source_edge.append(u_source_edge)
            distance.append(u_distance)
            distance.append(u_distance)
            distance_edge.append(u_distance_edge)
            distance_edge.append(u_distance_edge)
            matching.append(u_matching)
            matching.append(u_matching)
            matching_edge.append(u_matching_edge)
            matching_edge.append(u_matching_edge)
            false.append(u_false)
            false.append(u_false)
            false_edge.append(u_false_edge)
            false_edge.append(u_false_edge)
            merge.append(u_merge)
            merge.append(u_merge)
            merge_edge.append(u_merge_edge)
            merge_edge.append(u_merge_edge)
            split.append(u_split)
            split.append(u_split)
            split_edge.append(u_split_edge)
            split_edge.append(u_split_edge)
            gt_and_pred_components.append(u_matching)
            gt_and_pred_components.append(v_matching)
            gt_components.append(u_source)
            gt_components.append(v_source)
            distances.append(0)
            distances.append(0)
            error_vis_edge.append(u_error_vis_edge)
            error_vis_edge.append(v_error_vis_edge)
            error_vis.append(u_error_vis)
            error_vis.append(v_error_vis)
        k += 1

        # get distance edges
        for k, (node, attrs) in enumerate(self.gt.nodes.items()):

            mst_u_labels = self.y_node_to_x_label.get(node, [])
            mst_u_label = list(mst_u_labels)[0] if len(mst_u_labels) == 1 else None
            gt_u_label = self.labels_y[node]

            # from index
            u_edge = 2 * (i + j + k)
            v_edge = 2 * (i + j + k) + 1
            # from graph_attr
            u_vertex_position = self.gt.nodes[node]["location"]
            # both from mst
            u_source = self.source_colors[1]
            u_source_edge = 0
            # distance edge attr so both same
            u_distance = 0
            # no
            u_distance_edge = 0
            # yes if both match to same label
            u_matching = (0, 0, 0)
            u_matching_edge = 0
            # no, has to be done on a per node basis
            u_false_edge = 1 if mst_u_label is None else 0
            u_false = self.false_colors[1]
            # yes if both match to different labels
            u_merge_edge = 0
            u_merge = self.merge_colors[u_merge_edge]
            # no
            u_split = self.split_colors[0]
            u_split_edge = 0

            u_error_vis_edge = 0
            v_error_vis_edge = 0
            u_error_vis = (0, 0, 0)
            v_error_vis = (0, 0, 0)

            edges.append(u_edge)
            edges.append(v_edge)
            vertex_positions.append(u_vertex_position)
            vertex_positions.append(u_vertex_position)
            source.append(u_source)
            source.append(u_source)
            source_edge.append(u_source_edge)
            source_edge.append(u_source_edge)
            distance.append(u_distance)
            distance.append(u_distance)
            distance_edge.append(u_distance_edge)
            distance_edge.append(u_distance_edge)
            matching.append(u_matching)
            matching.append(u_matching)
            matching_edge.append(u_matching_edge)
            matching_edge.append(u_matching_edge)
            false.append(u_false)
            false.append(u_false)
            false_edge.append(u_false_edge)
            false_edge.append(u_false_edge)
            merge.append(u_merge)
            merge.append(u_merge)
            merge_edge.append(u_merge_edge)
            merge_edge.append(u_merge_edge)
            split.append(u_split)
            split.append(u_split)
            split_edge.append(u_split_edge)
            split_edge.append(u_split_edge)
            gt_and_pred_components.append(u_matching)
            gt_and_pred_components.append(v_matching)
            gt_components.append(u_source)
            gt_components.append(v_source)
            distances.append(0)
            distances.append(0)
            error_vis_edge.append(u_error_vis_edge)
            error_vis_edge.append(v_error_vis_edge)
            error_vis.append(u_error_vis)
            error_vis.append(v_error_vis)
        i += 1

        return neuroglancer.skeleton.Skeleton(
            vertex_positions=vertex_positions,
            edges=edges,
            vertex_attributes=dict(
                source=source,
                source_edge=source_edge,
                distance=distances,
                # distance_edge=distance_edge,
                matching=matching,
                matching_edge=matching_edge,
                false_match=false,
                false_match_edge=false_edge,
                merge=merge,
                merge_edge=merge_edge,
                split=split,
                split_edge=split_edge,
                gt_and_pred_components=gt_and_pred_components,
                gt_components=gt_components,
                error_vis=error_vis,
                error_vis_edge=error_vis_edge,
            ),
            # edge_attribues=dict(distances=distances),
        )


def build_trees(node_ids, locations, edges, node_attrs=None, edge_attrs=None):
    if node_attrs is None:
        node_attrs = {}
    if edge_attrs is None:
        edge_attrs = {}
    trees = nx.DiGraph()
    pbs = {
        int(node_id): node_location
        for node_id, node_location in zip(node_ids, locations)
    }
    for i, row in enumerate(edges):
        u = int(row[0])
        v = int(row[-1])

        e_attrs = {attr: values[i] for attr, values in edge_attrs.items()}

        if u == -1 or v == -1:
            continue

        pos_u = daisy.Coordinate(tuple(pbs[u]))
        pos_v = daisy.Coordinate(tuple(pbs[v]))

        if u not in trees.nodes:
            u_attrs = {attr: values[u] for attr, values in node_attrs}
            trees.add_node(u, location=pos_u, **u_attrs)
        if v not in trees.nodes:
            v_attrs = {attr: values[v] for attr, values in node_attrs}
            trees.add_node(v, location=pos_v, **v_attrs)

        trees.add_edge(u, v, **e_attrs)
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
        mst.add_node(u, location=pos_u + offset)
        mst.add_node(v, location=pos_v + offset)
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


def add_tree(
    s, tree, name, visible=False, color=None, node_attrs=None, edge_attrs=None
):

    dimensions, voxel_offset, voxel_shape = dims_from_guess(tree)

    s.layers.append(
        name="{}".format(name),
        layer=neuroglancer.SegmentationLayer(
            source=[
                neuroglancer.LocalVolume(
                    data=np.ones(voxel_shape, dtype=np.uint32),
                    dimensions=dimensions,
                    voxel_offset=voxel_offset,
                ),
                SkeletonSource(
                    tree,
                    dimensions,
                    voxel_size=[1000, 300, 300],
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                ),
            ],
            skeleton_shader="""
void main() {
  emitRGB(colormapJet(0.5));
}
""",
            selected_alpha=0,
            not_selected_alpha=0,
        ),
    )


def add_trees(
    s,
    trees,
    node_id,
    name,
    dimensions,
    voxel_offset,
    voxel_shape,
    visible=False,
    color=None,
    node_attrs=None,
    edge_attrs=None,
):

    if trees is None:
        return None

    s.layers.append(
        name="{}".format(name),
        layer=neuroglancer.SegmentationLayer(
            source=[
                neuroglancer.LocalVolume(
                    data=np.ones(voxel_shape, dtype=np.uint32),
                    dimensions=dimensions,
                    voxel_offset=voxel_offset,
                ),
                SkeletonSource(
                    trees,
                    dimensions,
                    voxel_size=[1000, 300, 300],
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                ),
            ],
            skeleton_shader="""
void main() {
  emitRGB(colormapJet(0.5));
}
""",
            selected_alpha=0,
            not_selected_alpha=0,
        ),
    )


def add_layer(context, array, name, visible=True, **kwargs):
    array_dims = len(array.shape)
    voxel_size = array.voxel_size
    attrs = {
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
        data=array.data, dimensions=dimensions, voxel_offset=offset
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


def dimensions_from_volume(array):
    voxel_size = array.voxel_size
    spatial_dims = 3
    attrs = {"names": ["z", "y", "x"], "units": "nm", "scales": voxel_size}
    dimensions = neuroglancer.CoordinateSpace(**attrs)
    offset = np.array(array.roi.get_offset())
    offset = offset // attrs["scales"]
    return dimensions, offset, array.shape[-spatial_dims:]


def dimensions_from_guess(dataset, graph):
    locations = dataset[f"points/{graph}-locations"][()]
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


def add_snapshot(
    context,
    snapshot_file,
    name_prefix="",
    volumes=None,
    graphs=None,
    graph_node_attrs=None,
    graph_edge_attrs=None,
):
    with h5py.File(snapshot_file) as dataset:
        if volumes is None:
            volumes = list(dataset.get("volumes", {}).keys())
        if graphs is None:
            graphs = list(dataset.get("points", {}).keys())
            graphs = set([p.split("-")[0] for p in graphs])
        if graph_node_attrs is None:
            graph_node_attrs = []
        if graph_edge_attrs is None:
            graph_edge_attrs = []

        node_id = itertools.count(start=1)

        v = None
        for volume in volumes:
            v = daisy.open_ds(str(snapshot_file.absolute()), f"volumes/{volume}")
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

        # graphs should not be dependent on volumes
        for graph in graphs:
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
            add_trees(
                context,
                components,
                node_id,
                dimensions=dimensions,
                voxel_offset=voxel_offset,
                voxel_shape=voxel_shape,
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
// false
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
    threshold_above=True,
    threshold_inds=None,
    graph_node_attrs=None,
    graph_edge_attrs=None,
):
    mst = tree_from_snapshot(
        mst_snapshot,
        "mst",
        graph_node_attrs=graph_node_attrs,
        graph_edge_attrs=graph_edge_attrs,
    )

    label_matchings, node_matchings, node_labels_mst, node_labels_gt, thresholds = (
        matching_data
    )
    for i, threshold in enumerate(thresholds):
        for (u, v), attrs in list(mst.edges.items()):
            if threshold_above:
                if attrs["distance"] > threshold:
                    mst.remove_edge(u, v)
            else:
                if attrs["distance"] < threshold:
                    mst.remove_edge(u, v)
        for node in list(mst.nodes):
            if mst.degree(node) == 0:
                mst.remove_node(node)

        if i in threshold_inds:
            add_match_layers(
                context,
                copy.deepcopy(mst),
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
