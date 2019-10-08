import unittest
import neurolight as nl
import networkx as nx
import numpy as np
import itertools


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
                ("a", {"location": np.array([1, 0, 0])}),
                ("b", {"location": np.array([1, 0, 5])}),
                ("c", {"location": np.array([1, 0, 10])}),
                ("d", {"location": np.array([1, 0, 15])}),
                ("e", {"location": np.array([1, 0, 20])}),
            ]
        )
        skeleton.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])

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

    def test_simple_long_chain(self):

        # consensus graph:
        #
        #       A---------->B---------->C
        #
        # skeleton graph:
        #
        # a--b--c--d--e--f--g--h--i--j--k--l--m
        #
        # matching should not have too many or too few edge assignments

        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("A", {"location": np.array([0, 0, 10])}),
                ("B", {"location": np.array([0, 0, 30])}),
                ("C", {"location": np.array([0, 0, 50])}),
            ]
        )
        consensus.add_edges_from([("A", "B"), ("B", "C")])

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("a", {"location": np.array([1, 0, 0])}),
                ("b", {"location": np.array([1, 0, 5])}),
                ("c", {"location": np.array([1, 0, 10])}),
                ("d", {"location": np.array([1, 0, 15])}),
                ("e", {"location": np.array([1, 0, 20])}),
                ("f", {"location": np.array([1, 0, 25])}),
                ("g", {"location": np.array([1, 0, 30])}),
                ("h", {"location": np.array([1, 0, 35])}),
                ("i", {"location": np.array([1, 0, 40])}),
                ("j", {"location": np.array([1, 0, 45])}),
                ("k", {"location": np.array([1, 0, 50])}),
                ("l", {"location": np.array([1, 0, 55])}),
                ("m", {"location": np.array([1, 0, 60])}),
            ]
        )
        skeleton.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("d", "e"),
                ("e", "f"),
                ("f", "g"),
                ("g", "h"),
                ("h", "i"),
                ("i", "j"),
                ("j", "k"),
                ("k", "l"),
                ("l", "m"),
            ]
        )

        nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        self.assertEqual(skeleton.edges[("a", "b")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges[("b", "c")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges[("c", "d")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("d", "e")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("e", "f")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("f", "g")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("g", "h")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("h", "i")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("i", "j")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("j", "k")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("k", "l")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges[("l", "m")].get("matched_edge"), None)

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
        # Matching should be able to match realistic 4 way junction

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
                ("c", {"location": np.array([1, 0, 0])}),
                ("d", {"location": np.array([1, 0, 5])}),
                ("g", {"location": np.array([1, 0, 20])}),
                ("f", {"location": np.array([1, 0, 15])}),
                ("e", {"location": np.array([1, 0, 10])}),
                ("b", {"location": np.array([1, -5, 5])}),
                ("a", {"location": np.array([1, -10, 5])}),
                ("h", {"location": np.array([1, 5, 15])}),
                ("i", {"location": np.array([1, 10, 15])}),
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

        self.assertEqual(skeleton.edges()[("a", "b")].get("matched_edge"), ("A", "X"))
        self.assertEqual(skeleton.edges()[("b", "d")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges()[("b", "e")].get("matched_edge"), ("A", "X"))
        self.assertEqual(skeleton.edges()[("c", "d")].get("matched_edge"), ("X", "D"))
        self.assertEqual(skeleton.edges()[("d", "e")].get("matched_edge"), ("X", "D"))
        self.assertEqual(skeleton.edges()[("e", "f")].get("matched_edge"), ("X", "B"))
        self.assertEqual(skeleton.edges()[("e", "h")].get("matched_edge"), ("X", "C"))
        self.assertEqual(skeleton.edges()[("f", "g")].get("matched_edge"), ("X", "B"))
        self.assertEqual(skeleton.edges()[("f", "h")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges()[("h", "i")].get("matched_edge"), ("X", "C"))

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
        # the optimal matching should not assign anything to extra chain

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
                ("a", {"location": np.array([1, 0, 0])}),
                ("b", {"location": np.array([1, 0, 5])}),
                ("c", {"location": np.array([1, 0, 10])}),
                ("d", {"location": np.array([1, 0, 15])}),
                ("e", {"location": np.array([1, 0, 20])}),
                ("f", {"location": np.array([1, -1, 10])}),
                ("h", {"location": np.array([1, -1, 15])}),
                ("i", {"location": np.array([1, -1, 20])}),
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


    def test_confounding_loop(self):

        # consensus graph:
        #
        # A---->B---->C
        #
        #
        # skeleton graph:
        #
        #  a--b--c--d--e
        #
        #        f--h
        #        | /
        #        i
        #
        # the optimal matching should not create a loop for extra reward

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
                ("a", {"location": np.array([1, 0, 0])}),
                ("b", {"location": np.array([1, 0, 5])}),
                ("c", {"location": np.array([1, 0, 10])}),
                ("d", {"location": np.array([1, 0, 15])}),
                ("e", {"location": np.array([1, 0, 20])}),
                ("f", {"location": np.array([1, -1, 10])}),
                ("h", {"location": np.array([1, -1, 15])}),
                ("i", {"location": np.array([1, -2, 10])}),
            ]
        )
        skeleton.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("d", "e"),
                ("f", "h"),
                ("f", "i"),
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
        self.assertEqual(skeleton.edges[("f", "h")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges[("f", "i")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges[("h", "i")].get("matched_edge"), None)

    def test_cheap_loops(self):

        # consensus graph:
        #
        # A---->B---->C---->D
        #
        #
        # skeleton graph:
        #
        #        l--m--n
        #      /         \
        #     k           o
        #     |           |
        #  a--b--c--d--e--f--g
        #        |     |
        #        h     j
        #         \   /
        #           i
        #
        # the optimal matching should not create a loop for extra reward.

        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("A", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 0, 10])}),
                ("C", {"location": np.array([0, 0, 20])}),
                ("D", {"location": np.array([0, 0, 30])}),
            ]
        )
        consensus.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("a", {"location": np.array([1, 0, 0])}),
                ("b", {"location": np.array([1, 0, 5])}),
                ("c", {"location": np.array([1, 0, 10])}),
                ("d", {"location": np.array([1, 0, 15])}),
                ("e", {"location": np.array([1, 0, 20])}),
                ("f", {"location": np.array([1, 0, 25])}),
                ("g", {"location": np.array([1, 0, 30])}),
                ("h", {"location": np.array([1, 0.4, 10])}),
                ("i", {"location": np.array([1, 10, 15])}),
                ("j", {"location": np.array([1, 0.4, 20])}),
                ("k", {"location": np.array([1, -0.3, 5])}),
                ("l", {"location": np.array([1, -10, 10])}),
                ("m", {"location": np.array([1, -10, 15])}),
                ("n", {"location": np.array([1, -10, 20])}),
                ("o", {"location": np.array([1, -0.3, 25])}),
            ]
        )
        skeleton.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("d", "e"),
                ("e", "f"),
                ("f", "g"),
                ("c", "h"),
                ("e", "j"),
                ("h", "i"),
                ("i", "j"),
                ("b", "k"),
                ("f", "o"),
                ("k", "l"),
                ("l", "m"),
                ("m", "n"),
                ("n", "o"),
            ]
        )

        matcher, optimized_score = nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        expected_tree = {
            ("a", "b"): ("A", "B"),
            ("b", "c"): ("A", "B"),
            ("c", "d"): ("B", "C"),
            ("d", "e"): ("B", "C"),
            ("e", "f"): ("C", "D"),
            ("f", "g"): ("C", "D"),
        }

        expected_nones = {
            ("c", "h"): None,
            ("e", "j"): None,
            ("h", "i"): None,
            ("i", "j"): None,
            ("b", "k"): None,
            ("f", "o"): None,
            ("k", "l"): None,
            ("l", "m"): None,
            ("m", "n"): None,
            ("n", "o"): None,
        }

        matcher.enforce_expected_assignments(expected_tree)
        _, expected_tree_score = matcher.match()

        self.assertLessEqual(expected_tree_score, optimized_score)

        for graph_e, tree_e in itertools.chain(
            expected_tree.items(), expected_nones.items()
        ):
            self.assertEqual(skeleton.edges[graph_e].get("matched_edge"), tree_e)

