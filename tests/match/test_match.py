import networkx as nx
import numpy as np
import pytest

from neurolight.match.costs import get_costs
from neurolight.match.preprocess import mouselight_preprocessing
from neurolight.gunpowder.nodes.topological_graph_matching import TopologicalMatcher
from neurolight.gunpowder.nodes.graph_source import GraphSource
from funlib.match.helper_functions import check_gurobi_license

from gunpowder import (
    MergeProvider,
    BatchRequest,
    Roi,
    Coordinate,
    PointsSpec,
    PointsKey,
    build,
)

from pathlib import Path


def skip_gurobi_if_no_license():
    success = check_gurobi_license()
    return pytest.mark.skipif(not success, reason="Requires Gurobi License")


def build_graphs(location_attr: str):
    """
    A-----B-----C
    a b-c   e-f-g
    """
    consensus_nodes = [
        ("A", {location_attr: np.array([1, 1, 1])}),
        ("B", {location_attr: np.array([1, 1, 4])}),
        ("C", {location_attr: np.array([1, 1, 7])}),
    ]
    consensus_edges = [("A", "B"), ("B", "C")]

    skeleton_nodes = [
        ("a", {location_attr: np.array([1, 1, 1])}),
        ("b", {location_attr: np.array([1, 1, 2])}),
        ("c", {location_attr: np.array([1, 1, 3])}),
        ("e", {location_attr: np.array([1, 1, 5])}),
        ("f", {location_attr: np.array([1, 1, 6])}),
        ("g", {location_attr: np.array([1, 1, 7])}),
    ]
    skeleton_edges = [("b", "c"), ("e", "f"), ("f", "g")]

    consensus = nx.DiGraph()
    consensus.add_nodes_from(consensus_nodes)
    consensus.add_edges_from(consensus_edges)

    skeleton = nx.Graph()
    skeleton.add_nodes_from(skeleton_nodes)
    skeleton.add_edges_from(skeleton_edges)

    return skeleton, consensus


valid_examples = [
    example.name
    for example in (Path(__file__).parent / "mouselight_examples" / "valid").iterdir()
]


@pytest.mark.parametrize("example", valid_examples)
@pytest.mark.parametrize(
    "use_gurobi", [pytest.param(True, marks=skip_gurobi_if_no_license()), False]
)
def test_realistic_valid_examples(example, use_gurobi):
    penalty_attr = "penalty"
    location_attr = "location"
    example_dir = Path(__file__).parent / "mouselight_examples" / "valid" / example

    consensus = PointsKey("CONSENSUS")
    skeletonization = PointsKey("SKELETONIZATION")
    matched = PointsKey("MATCHED")

    inf_roi = Roi(Coordinate((None,) * 3), Coordinate((None,) * 3))

    request = BatchRequest()
    request[matched] = PointsSpec(roi=inf_roi)

    pipeline = (
        (
            GraphSource(example_dir / "graph.obj", [skeletonization]),
            GraphSource(example_dir / "tree.obj", [consensus]),
        )
        + MergeProvider()
        + TopologicalMatcher(
            skeletonization,
            consensus,
            matched,
            76,
            48,
            use_gurobi=use_gurobi,
            location_attr=location_attr,
            penalty_attr=penalty_attr,
        )
    )

    with build(pipeline):
        batch = pipeline.request_batch(request)
        assert matched in batch
        assert len(batch[matched].graph.nodes()) > 0


invalid_examples = [
    example.name
    for example in (Path(__file__).parent / "mouselight_examples" / "invalid").iterdir()
]


@pytest.mark.parametrize("example", invalid_examples)
@pytest.mark.parametrize(
    "use_gurobi", [pytest.param(True, marks=skip_gurobi_if_no_license()), False]
)
def test_realistic_invalid_examples(example, use_gurobi):
    penalty_attr = "penalty"
    location_attr = "location"
    example_dir = Path(__file__).parent / "mouselight_examples" / "invalid" / example

    consensus = PointsKey("CONSENSUS")
    skeletonization = PointsKey("SKELETONIZATION")
    matched = PointsKey("MATCHED")

    inf_roi = Roi(Coordinate((None,) * 3), Coordinate((None,) * 3))

    request = BatchRequest()
    request[matched] = PointsSpec(roi=inf_roi)

    pipeline = (
        (
            GraphSource(example_dir / "graph.obj", [skeletonization]),
            GraphSource(example_dir / "tree.obj", [consensus]),
        )
        + MergeProvider()
        + TopologicalMatcher(
            skeletonization,
            consensus,
            matched,
            76,
            48,
            use_gurobi=use_gurobi,
            location_attr=location_attr,
            penalty_attr=penalty_attr,
        )
    )

    with build(pipeline):
        batch = pipeline.request_batch(request)
        assert matched in batch
        assert len(batch[matched].graph.nodes()) == 0


def test_preprocessing():
    voxel_size = [1, 1, 1]
    location_attr = "location"
    penalty_attr = "penalty"
    skeleton, consensus = build_graphs(location_attr)
    mouselight_preprocessing(
        skeleton,
        1.5,
        voxel_size=voxel_size,
        location_attr=location_attr,
        penalty_attr=penalty_attr,
    )

    expected_added_edges = [(("a", "b"), 1)]

    for edge, penalty in expected_added_edges:
        assert skeleton.edges()[edge][penalty_attr] == pytest.approx(penalty)

    mouselight_preprocessing(
        skeleton,
        2.5,
        voxel_size=voxel_size,
        location_attr=location_attr,
        penalty_attr=penalty_attr,
    )

    expected_added_edges = [(("c", "e"), 2)]

    for edge, penalty in expected_added_edges:
        assert skeleton.edges()[edge][penalty_attr] == pytest.approx(penalty)


def test_get_costs():
    location_attr = "position"
    penalty_attr = "extra_cost"

    skeleton, consensus = build_graphs(location_attr)

    node_match_threshold = 2
    edge_match_threshold = 2
    node_balance = 2

    node_matchings, edge_matchings = get_costs(
        skeleton,
        consensus,
        location_attr,
        penalty_attr,
        node_match_threshold,
        edge_match_threshold,
        node_balance,
    )

    expected_node_distances = {
        "a": {"A": 0},
        "b": {"A": 1, "B": 2},
        "c": {"A": 2, "B": 1},
        "e": {"B": 1, "C": 2},
        "f": {"B": 2, "C": 1},
        "g": {"C": 0},
    }

    expected_edge_distances = {
        ("b", "c"): {("A", "B"): 0, ("B", "C"): 3 / 2},
        ("c", "b"): {("A", "B"): 0, ("B", "C"): 3 / 2},
        ("e", "f"): {("A", "B"): 3 / 2, ("B", "C"): 0},
        ("f", "e"): {("A", "B"): 3 / 2, ("B", "C"): 0},
        ("f", "g"): {("B", "C"): 0},
        ("g", "f"): {("B", "C"): 0},
    }

    for skeleton_node, tree_node, cost in node_matchings:
        assert expected_node_distances[skeleton_node][
            tree_node
        ] * node_balance == pytest.approx(cost)

    for skeleton_edge, tree_edge, cost in edge_matchings:
        assert expected_edge_distances[skeleton_edge][tree_edge] == pytest.approx(cost)

