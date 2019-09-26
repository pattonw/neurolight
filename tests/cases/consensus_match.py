import unittest
import neurolight as nl
import networkx as nx
import numpy as np


class ConsensusMatchTest(unittest.TestCase):
    def test_simple(self):

        # consensus graph:
        #
        # o-----o-----o
        #
        # skeleton graph:
        #
        # o--o--o--o--o
        consensus = nx.Graph()
        consensus.add_nodes_from(
            [
                (1, {"location": np.array([0, 0, 0])}),
                (2, {"location": np.array([0, 0, 10])}),
                (3, {"location": np.array([0, 0, 20])}),
            ]
        )
        consensus.add_edges_from([(1, 2), (2, 3)])

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                (1, {"location": np.array([0, 0, 0])}),
                (2, {"location": np.array([0, 0, 5])}),
                (3, {"location": np.array([0, 0, 10])}),
                (4, {"location": np.array([0, 0, 15])}),
                (5, {"location": np.array([0, 0, 20])}),
            ]
        )
        skeleton.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])

        nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        self.assertEqual(skeleton.edges[(1, 2)]["matched_edge"] == (1, 2))
        self.assertEqual(skeleton.edges[(2, 3)]["matched_edge"] == (1, 2))
        self.assertEqual(skeleton.edges[(3, 4)]["matched_edge"] == (2, 3))
        self.assertEqual(skeleton.edges[(4, 5)]["matched_edge"] == (2, 3))

    def test_simple_4_way(self):

        # consensus graph:
        #
        #    A
        #     \
        #      \
        # D-----X-----B
        #        \
        #         \
        #          C
        #
        # center node gets replaced with:
        #
        #  tl   XA   tr
        #     / | \
        #  XD---+---XB  h (horizontal) v(vertical)
        #     \ | /
        #  bl   XC   br
        #
        # skeleton graph:
        #
        #    a
        #    |
        #    b
        #    |
        # c--d--e--f--g
        #          |
        #          h
        #          |
        #          i
        #
        # a should map to A-XA
        # b should map to XA-XD
        # c should map to D-XD
        # d should map to XB-XD
        # e should map to XB-XD
        # f should map to B-XB
        # g should map to XB-XC
        # h should map to C-XC

        consensus = nx.Graph()
        consensus.add_nodes_from(
            [
                ("D", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 0, 20])}),
                ("X", {"location": np.array([0, 0, 10])}),
                ("A", {"location": np.array([0, -10, 5])}),
                ("C", {"location": np.array([0, 10, 15])}),
            ]
        )
        consensus.add_edges_from([("A", "X"), ("B", "X"), ("C", "X"), ("D", "X")])

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
                ("c", "d"),
                ("d", "e"),
                ("e", "f"),
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

        self.assertEqual(
            skeleton.edges[("a", "b")]["matched_edge"] == ("A", ("X", "A"))
        )
        self.assertEqual(
            skeleton.edges[("c", "d")]["matched_edge"] == ("D", ("X", "D"))
        )
        self.assertEqual(
            skeleton.edges[("f", "g")]["matched_edge"] == ("B", ("X", "B"))
        )
        self.assertEqual(
            skeleton.edges[("h", "i")]["matched_edge"] == ("C", ("X", "C"))
        )

        self.assertEqual(
            skeleton.edges[("b", "d")]["matched_edge"], (("X", "A"), ("X", "D"))
        )
        self.assertEqual(
            skeleton.edges[("d", "e")]["matched_edge"], (("X", "D"), ("X", "B"))
        )
        self.assertEqual(
            skeleton.edges[("e", "f")]["matched_edge"], (("X", "D"), ("X", "B"))
        )
        self.assertEqual(
            skeleton.edges[("f", "h")]["matched_edge"], (("X", "B"), ("X", "C"))
        )

    def test_confounding_chain(self):

        # consensus graph:
        #
        # A-----B-----C
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

        consensus = nx.Graph()
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
        self.assertEqual(skeleton.edges[("c", "f")]["matched_edge"], None)
        self.assertEqual(skeleton.edges[("f", "h")]["matched_edge"], None)
        self.assertEqual(skeleton.edges[("h", "i")]["matched_edge"], None)

