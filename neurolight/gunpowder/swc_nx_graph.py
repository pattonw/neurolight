import networkx as nx
import numpy as np
from gunpowder import Roi, Coordinate
from typing import List, Tuple, Dict
import copy

from .swc_point import SwcPoint


def subgraph_from_points(graph, nodes: List[int], with_neighbors=False) -> nx.DiGraph:
    """
    Creates a subgraph of `graph` that contains the points in `nodes`.
    If `with_neighbors` is True, the subgraph contains all neighbors
    of all points in `nodes` as well.
    """
    sub_g = nx.DiGraph()
    subgraph_nodes = set(nodes)
    subgraph_edges = set()
    for n in nodes:
        for successor in graph.successors(n):
            if with_neighbors or successor in nodes:
                subgraph_nodes.add(successor)
                subgraph_edges.add((n, successor))
        for predecessor in graph.predecessors(n):
            if with_neighbors or predecessor in nodes:
                subgraph_nodes.add(predecessor)
                subgraph_edges.add((predecessor, n))
    for n in subgraph_nodes:
        sub_g.add_node(n, **copy.deepcopy(graph.nodes[n]))
    sub_g.add_edges_from(subgraph_edges)
    return sub_g


def points_to_graph(points: Dict[int, SwcPoint]) -> nx.DiGraph:
    g = nx.DiGraph()
    for point_id, point in points.items():
        g.add_node(
            point_id,
            location=point.location,
            point_type=point.point_type,
            radius=point.radius,
        )
        if (
            point.parent_id is not None
            and point.parent_id != point_id
            and point.parent_id != -1
            and point.parent_id in points
        ):
            g.add_edge(point.parent_id, point_id)

    # check if the temporary graph is tree like
    assert nx.is_directed_acyclic_graph(g), "Malformed Graph!"
    return g


def relabel_connected_components(graph: nx.DiGraph, offset=0):
    temp_g = copy.deepcopy(graph)
    i = -1
    for i, connected_component in enumerate(nx.weakly_connected_components(temp_g)):
        label = i + offset + 1
        for node in connected_component:
            temp_g.nodes[node]["label_id"] = label

    temp_g = nx.convert_node_labels_to_integers(temp_g)
    return temp_g, i


def graph_to_swc_points(graph: nx.DiGraph) -> Dict[int, SwcPoint]:
    graph, _ = relabel_connected_components(graph)
    return {
        node: SwcPoint(
            point_id=node,
            parent_id=list(graph._pred[node].keys())[0]
            if len(graph._pred[node]) == 1
            else -1,
            **graph.nodes[node]
        )
        for node in graph.nodes
    }


def interpolate_points(graph: nx.DiGraph) -> nx.DiGraph:
    n = max([node for node in graph.nodes]) + 1
    temp_g = copy.deepcopy(graph)
    for node in graph.nodes:
        for succ in graph.successors(node):
            temp_g.add_node(
                n,
                location=(graph.nodes[node]["location"] + graph.nodes[succ]["location"])
                // 2,
            )
            temp_g.remove_edge(node, succ)
            temp_g.add_edge(node, n)
            temp_g.add_edge(n, succ)
            n += 1
    return temp_g


def shift_graph(graph: nx.DiGraph, direction: np.ndarray):
    temp_g = copy.deepcopy(graph)
    for node_id, node in temp_g.nodes.items():
        node["location"] += direction.astype(int)
    return temp_g


def crop_graph(graph: nx.DiGraph, roi: Roi, keep_ids=False):
    temp_g = copy.deepcopy(graph)

    one = Coordinate((1, 1, 1))
    zero = Coordinate((0, 0, 0))
    # shrink bottom by 1, since roi already doesn't allow points at the top
    allowable_points = copy.deepcopy(roi)
    allowable_points = allowable_points.grow(-one, zero)
    # bbox on which points may lie
    bbox = copy.deepcopy(roi)
    bbox = bbox.grow(-one, -one)

    to_remove = set()
    predecessors = set()  # u if u is out
    pred_edges = set()
    successors = set()  # v if v is out
    succ_edges = set()
    for u in temp_g.nodes:
        u_in = allowable_points.contains(temp_g.nodes[u]["location"])
        if not u_in:
            to_remove.add(u)
        for v in temp_g.successors(u):
            v_in = allowable_points.contains(temp_g.nodes[v]["location"])
            if not u_in and not v_in:
                to_remove.add(v)
            elif not u_in:
                predecessors.add(u)
                pred_edges.add((u, v))
            elif not v_in:
                successors.add(v)
                succ_edges.add((u, v))

    for node in to_remove.difference(predecessors, successors):
        temp_g.remove_node(node)
    for out_node, in_neighbor in pred_edges:
        new_location = line_box_intercept(
            temp_g.nodes[in_neighbor]["location"],
            temp_g.nodes[out_node]["location"],
            allowable_points,
            bbox,
        )
        if all(np.isclose(new_location, temp_g.nodes[in_neighbor]["location"])):
            temp_g.remove_node(out_node)
        else:
            temp_g.nodes[out_node]["location"] = new_location
    for in_neighbor, out_node in succ_edges:
        new_location = line_box_intercept(
            temp_g.nodes[in_neighbor]["location"],
            temp_g.nodes[out_node]["location"],
            allowable_points,
            bbox,
        )
        if all(np.isclose(new_location, temp_g.nodes[in_neighbor]["location"])):
            temp_g.remove_node(out_node)
        else:
            temp_g.nodes[out_node]["location"] = new_location

    for node in temp_g.nodes:
        assert roi.contains(
            temp_g.nodes[node]["location"]
        ), "node({}) in to_remove: {}, predecessors: {}, successors: {}".format(
            temp_g.nodes[node]["location"],
            node in to_remove,
            node in successors,
            node in predecessors,
        )
    return temp_g


def line_box_intercept(
    inside: np.ndarray, outside: np.ndarray, allowable_points: Roi, bb: Roi
) -> np.ndarray:
    offset = outside - inside
    with np.errstate(divide="ignore", invalid="ignore"):
        # bb_crossings will be 0 if inside is on the bb, 1 if outside is on the bb
        bb_x = np.asarray(
            [
                (np.asarray(bb.get_begin()) - inside) / offset,
                (np.asarray(bb.get_end()) - inside) / offset,
            ],
            dtype=float,
        )

        if np.sum(np.logical_and((bb_x >= 0), (bb_x <= 1))) > 0:
            # all values of bb_x between 0, 1 represent a crossing of a bounding plane
            # the minimum of which is the (normalized) distance to the closest bounding plane
            s = np.min(bb_x[np.logical_and((bb_x > 0), (bb_x <= 1))])
            new_location = np.round(inside + s * offset).astype(int)
            if not allowable_points.contains(new_location):
                raise Exception(
                    "New location {} between {}, {} not contained in {}".format(
                        new_location, inside, outside, allowable_points
                    )
                )
            return new_location
        else:
            raise ValueError(
                (
                    "Could not create a node on the bounding box {} "
                    + "given points (inside:{}, ouside:{})"
                ).format(bb, inside, outside)
            )

