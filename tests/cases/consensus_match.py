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
        # the optimal matching should not create a loop around f-h-i

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
                ("i", {"location": np.array([0, -2, 10])}),
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

    def test_better_with_loop(self):

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
        # the optimal matching would be a-b-c-d-e-f-g
        # however a higher score could be achieved by taking
        # the upper path and allowing a loop in the middle

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
                ("a", {"location": np.array([0, 0, 0])}),
                ("b", {"location": np.array([0, 0, 5])}),
                ("c", {"location": np.array([0, 0, 10])}),
                ("d", {"location": np.array([0, 0, 15])}),
                ("e", {"location": np.array([0, 0, 20])}),
                ("f", {"location": np.array([0, 0, 25])}),
                ("g", {"location": np.array([0, 0, 30])}),
                ("h", {"location": np.array([0, 0.4, 10])}),
                ("i", {"location": np.array([0, 10, 15])}),
                ("j", {"location": np.array([0, 0.4, 20])}),
                ("k", {"location": np.array([0, -0.3, 5])}),
                ("l", {"location": np.array([0, -10, 10])}),
                ("m", {"location": np.array([0, -10, 15])}),
                ("n", {"location": np.array([0, -10, 20])}),
                ("o", {"location": np.array([0, -0.3, 25])}),
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

