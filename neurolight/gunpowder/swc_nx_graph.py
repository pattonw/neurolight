import networkx as nx
import numpy as np
from gunpowder import Roi, Coordinate, GraphPoint as SwcPoint
from typing import List, Tuple, Dict
import copy
import logging

logger = logging.getLogger(__name__)


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


def relabel_connected_components(graph: nx.DiGraph, offset=0, keep_ids: bool = False):
    temp_g = copy.deepcopy(graph)
    i = -1
    for i, connected_component in enumerate(nx.weakly_connected_components(temp_g)):
        label = i + offset + 1
        for node in connected_component:
            temp_g.nodes[node]["label_id"] = label

    if not keep_ids:
        temp_g = nx.convert_node_labels_to_integers(temp_g)
    return temp_g, i


def graph_to_swc_points(
    graph: nx.DiGraph, keep_ids: bool = False
) -> Dict[int, SwcPoint]:
    graph, _ = relabel_connected_components(graph, keep_ids=keep_ids)
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
    if len(graph.nodes) > 0:
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
    if len(graph.nodes) > 0:
        next_node_id = max([node_id for node_id in temp_g.nodes]) + 1
    else:
        next_node_id = 0

    bbox = copy.deepcopy(roi)

    to_remove = set()
    predecessors = set()  # u if u is out
    pred_edges = set()
    successors = set()  # v if v is out
    succ_edges = set()
    for u in temp_g.nodes:
        u_in = bbox.contains(temp_g.nodes[u]["location"])
        if not u_in:
            to_remove.add(u)
        for v in temp_g.successors(u):
            v_in = bbox.contains(temp_g.nodes[v]["location"])
            if not u_in and not v_in:
                to_remove.add(v)
            elif not u_in:
                predecessors.add(u)
                pred_edges.add((u, v))
            elif not v_in:
                successors.add(v)
                succ_edges.add((u, v))

    # problem: what if there are multiple edges that cross the bbox to one outnode?
    # it would get deleted, then be inaccessable the next time. Thus we must create new nodes
    for out_node, in_neighbor in pred_edges:
        new_location = line_box_intercept(
            temp_g.nodes[in_neighbor]["location"],
            temp_g.nodes[out_node]["location"],
            bbox,
        )
        if not all(np.isclose(new_location, temp_g.nodes[in_neighbor]["location"])):
            temp_g.add_node(next_node_id, **temp_g.nodes[out_node])
            temp_g.nodes[next_node_id]["location"] = new_location
            temp_g.add_edge(next_node_id, in_neighbor)
            next_node_id += 1
    for in_neighbor, out_node in succ_edges:
        new_location = line_box_intercept(
            temp_g.nodes[in_neighbor]["location"],
            temp_g.nodes[out_node]["location"],
            bbox,
        )
        if not all(np.isclose(new_location, temp_g.nodes[in_neighbor]["location"])):
            temp_g.add_node(next_node_id, **temp_g.nodes[out_node])
            temp_g.nodes[next_node_id]["location"] = new_location
            temp_g.add_edge(in_neighbor, next_node_id)
            next_node_id += 1

    for node in to_remove:
        temp_g.remove_node(node)

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


def line_box_intercept(inside: np.ndarray, outside: np.ndarray, bb: Roi) -> np.ndarray:
    offset = outside - inside

    bb_x = np.asarray(
        [
            (np.asarray(bb.get_begin()) - inside) / offset,
            (np.asarray(bb.get_end()) - inside) / offset,
        ],
        dtype=float,
    )

    s = np.min(bb_x[np.logical_and((bb_x >= 0), (bb_x < 1))])
    new_location = np.floor(inside + s * offset).astype(int)
    return new_location

