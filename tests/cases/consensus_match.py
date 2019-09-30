import unittest
import neurolight as nl
import networkx as nx
import numpy as np

from neurolight.match_graph_to_tree import GraphToTreeMatcher

class ConsensusMatchTest(unittest.TestCase):
    def test_simple(self):

        # consensus graph:
        #
        # A---->B---->C
        #
        # skeleton graph:
        #
        # a--b--c--d--e
        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("A", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 0, 10])}),
                ("C", {"location": np.array([0, 0, 20])}),
            ]
        )
        consensus.add_edges_from([("A", "B"), ("B", "C")])

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("a", {"location": np.array([0, 0, 0])}),
                ("b", {"location": np.array([0, 0, 5])}),
                ("c", {"location": np.array([0, 0, 10])}),
                ("d", {"location": np.array([0, 0, 15])}),
                ("e", {"location": np.array([0, 0, 20])}),
            ]
        )
        skeleton.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])

        nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        self.assertEqual(skeleton.edges[("a", "b")]["matched_edge"], Edge("A", "B"))
        self.assertEqual(skeleton.edges[("b", "c")]["matched_edge"], Edge("A", "B"))
        self.assertEqual(skeleton.edges[("c", "d")]["matched_edge"], Edge("B", "C"))
        self.assertEqual(skeleton.edges[("d", "e")]["matched_edge"], Edge("B", "C"))

        self.assertEqual(skeleton.edges[("a", "b")]["matched_edge"], ("A", "B"))
        self.assertEqual(skeleton.edges[("b", "c")]["matched_edge"], ("A", "B"))
        self.assertEqual(skeleton.edges[("c", "d")]["matched_edge"], ("B", "C"))
        self.assertEqual(skeleton.edges[("d", "e")]["matched_edge"], ("B", "C"))

    def test_simple_4_way(self):

        # consensus graph:
        #
        #    A
        #     \
        #      \
        # D<----X---->B
        #        \
        #         \
        #          C
        #
        # skeleton graph:
        #
        #    a
        #    |
        #    b
        #    | \
        # c--d--e--f--g
        #        \ |
        #          h
        #          |
        #          i
        #
        # ab should map to A->X
        # bd should map to None
        # be should map to A->X
        # cd should map to X->D
        # de should map to X->D
        # ef should map to X->B
        # eh should map to X->C
        # fg should map to X->B
        # hi should map to X->C
        # fh should map to None

        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("D", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 0, 20])}),
                ("X", {"location": np.array([0, 0, 10])}),
                ("A", {"location": np.array([0, -10, 5])}),
                ("C", {"location": np.array([0, 10, 15])}),
            ]
        )
        consensus.add_edges_from([("A", "X"), ("X", "B"), ("X", "C"), ("X", "D")])

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("c", {"location": np.array([0, 0, 0])}),
                ("d", {"location": np.array([0, 0, 5])}),
                ("g", {"location": np.array([0, 0, 20])}),
                ("f", {"location": np.array([0, 0, 15])}),
                ("e", {"location": np.array([0, 0, 10])}),
                ("b", {"location": np.array([0, -5, 5])}),
                ("a", {"location": np.array([0, -10, 5])}),
                ("h", {"location": np.array([0, 5, 15])}),
                ("i", {"location": np.array([0, 10, 15])}),
            ]
        )
        skeleton.add_edges_from(
            [
                ("a", "b"),
                ("b", "d"),
                ("b", "e"),
                ("c", "d"),
                ("d", "e"),
                ("e", "f"),
                ("e", "h"),
                ("f", "g"),
                ("f", "h"),
                ("h", "i"),
            ]
        )

        nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        self.assertEqual(skeleton.edges[("a", "b")]["matched_edge"], Edge("A", ("X", "A")))
        self.assertEqual(skeleton.edges[("c", "d")]["matched_edge"], Edge("D", ("X", "D")))
        self.assertEqual(skeleton.edges[("f", "g")]["matched_edge"], Edge("B", ("X", "B")))
        self.assertEqual(skeleton.edges[("h", "i")]["matched_edge"], Edge("C", ("X", "C")))

        self.assertEqual(skeleton.edges()[("a", "b")]["matched_edge"], ("A", "X"))
        self.assertEqual(skeleton.edges()[("b", "d")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges()[("b", "e")]["matched_edge"], ("A", "X"))
        self.assertEqual(skeleton.edges()[("c", "d")]["matched_edge"], ("X", "D"))
        self.assertEqual(skeleton.edges()[("d", "e")]["matched_edge"], ("X", "D"))
        self.assertEqual(skeleton.edges()[("e", "f")]["matched_edge"], ("X", "B"))
        self.assertEqual(skeleton.edges()[("e", "h")]["matched_edge"], ("X", "C"))
        self.assertEqual(skeleton.edges()[("f", "g")]["matched_edge"], ("X", "B"))
        self.assertEqual(skeleton.edges()[("f", "h")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges()[("h", "i")]["matched_edge"], ("X", "C"))

    def test_confounding_chain(self):

        # consensus graph:
        #
        # A---->B---->C
        #
        #
        # skeleton graph:
        #
        #  a--b--c--d--e
        #        |
        #        f--h--i
        #
        # the optimal matching should not cheat and assign
        # None to c-f, and BC to f-h, and h-i to reduce cost

        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("A", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 0, 10])}),
                ("C", {"location": np.array([0, 0, 20])}),
            ]
        )
        consensus.add_edges_from([("A", "B"), ("B", "C")])

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("a", {"location": np.array([0, 0, 0])}),
                ("b", {"location": np.array([0, 0, 5])}),
                ("c", {"location": np.array([0, 0, 10])}),
                ("d", {"location": np.array([0, 0, 15])}),
                ("e", {"location": np.array([0, 0, 20])}),
                ("f", {"location": np.array([0, -1, 10])}),
                ("h", {"location": np.array([0, -1, 15])}),
                ("i", {"location": np.array([0, -1, 20])}),
            ]
        )
        skeleton.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("d", "e"),
                ("c", "f"),
                ("f", "h"),
                ("h", "i"),
            ]
        )

        nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        self.assertEqual(skeleton.edges[("a", "b")]["matched_edge"], ("A", "B"))
        self.assertEqual(skeleton.edges[("b", "c")]["matched_edge"], ("A", "B"))
        self.assertEqual(skeleton.edges[("c", "d")]["matched_edge"], ("B", "C"))
        self.assertEqual(skeleton.edges[("d", "e")]["matched_edge"], ("B", "C"))
        self.assertEqual(skeleton.edges[("c", "f")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges[("f", "h")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges[("h", "i")].get("matched_edge"), None)

