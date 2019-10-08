import unittest
import numpy as np
import networkx as nx

import neurolight as nl


class IllPosedConsensusMatchTest(unittest.TestCase):
    def test_topological_error_vs_absolute_error(self):

        # consensus graph:
        #
        #     B---->C
        #    /
        #   /
        #  /
        # A---->D---->E
        #  \
        #   \
        #    \
        #     F---->G
        #
        # skeleton graph:
        #
        #   /-b-----c
        #  /
        # |  /--d-----e
        # |/
        # a-----f-----g
        #
        # Should a-f-g map to A-D-E since they are perfect matches
        # or should a-f-g map to A-F-G due to topology?
        # To change this behavior, you may need to fiddle with the cost function

        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("A", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 10, 10])}),
                ("C", {"location": np.array([0, 10, 20])}),
                ("D", {"location": np.array([0, 0, 10])}),
                ("E", {"location": np.array([0, 0, 20])}),
                ("F", {"location": np.array([0, -10, 10])}),
                ("G", {"location": np.array([0, -10, 20])}),
            ]
        )
        consensus.add_edges_from(
            [("A", "B"), ("B", "C"), ("A", "D"), ("D", "E"), ("A", "F"), ("F", "G")]
        )

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("a", {"location": np.array([1, 0, 0])}),
                ("b", {"location": np.array([1, 10, 10])}),
                ("c", {"location": np.array([1, 10, 20])}),
                ("d", {"location": np.array([1, 5, 10])}),
                ("e", {"location": np.array([1, 5, 20])}),
                ("f", {"location": np.array([1, 0, 10])}),
                ("g", {"location": np.array([1, 0, 20])}),
            ]
        )
        skeleton.add_edges_from(
            [("a", "b"), ("b", "c"), ("a", "d"), ("d", "e"), ("a", "f"), ("f", "g")]
        )

        nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        self.assertEqual(skeleton.edges[("a", "b")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("b", "c")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("a", "d")].get("matched_edge"), ("A", "D"))
        self.assertEqual(skeleton.edges[("d", "e")].get("matched_edge"), ("D", "E"))
        self.assertEqual(skeleton.edges[("a", "f")].get("matched_edge"), ("A", "F"))
        self.assertEqual(skeleton.edges[("f", "g")].get("matched_edge"), ("F", "G"))

    def test_edge_sharing(self):

        # consensus graph:
        #
        #     B-------->C--------->F-------->H
        #   /
        # A
        #   \
        #     D-------->E--------->G-------->I
        #
        # skeleton graph:
        #
        #     b       g-----------i----------k
        #   / |     /
        # a   d---e
        #   \ |     \
        #     c       h-----------j----------l
        #
        # should d-e-f be able to match both B-C and D-E?
        # if you want to be able to assign multiple labels per edge,
        # remove the constraint on unique edge assignment

        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("A", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 1, 10])}),
                ("C", {"location": np.array([0, 1, 20])}),
                ("D", {"location": np.array([0, -1, 10])}),
                ("E", {"location": np.array([0, -1, 20])}),
                ("F", {"location": np.array([0, 1, 30])}),
                ("G", {"location": np.array([0, -1, 30])}),
                ("H", {"location": np.array([0, 1, 40])}),
                ("I", {"location": np.array([0, -1, 40])}),
            ]
        )
        consensus.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("A", "D"),
                ("D", "E"),
                ("C", "F"),
                ("E", "G"),
                ("F", "H"),
                ("G", "I"),
            ]
        )

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("a", {"location": np.array([1, 0, 0])}),
                ("b", {"location": np.array([1, 1, 10])}),
                ("c", {"location": np.array([1, -1, 10])}),
                ("d", {"location": np.array([1, 0, 13])}),
                ("e", {"location": np.array([1, 0, 15])}),
                ("g", {"location": np.array([1, 1, 20])}),
                ("h", {"location": np.array([1, -1, 20])}),
                ("i", {"location": np.array([1, 1, 30])}),
                ("j", {"location": np.array([1, -1, 30])}),
                ("k", {"location": np.array([1, 1, 40])}),
                ("l", {"location": np.array([1, -1, 40])}),
            ]
        )
        skeleton.add_edges_from(
            [
                ("a", "b"),
                ("b", "d"),
                ("a", "c"),
                ("c", "d"),
                ("d", "e"),
                ("e", "g"),
                ("e", "h"),
                ("g", "i"),
                ("h", "j"),
                ("i", "k"),
                ("j", "l"),
            ]
        )

        self.assertRaises(
            ValueError,
            nl.match_graph_to_tree,
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        # if this multiple edges assignments were allowed, this would be the expected output
        """
        self.assertEqual(skeleton.edges[("a", "b")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("a", "c")].get("matched_edge"), ("A", "D"))
        self.assertEqual(skeleton.edges[("b", "d")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("c", "d")].get("matched_edge"), ("D", "E"))
        self.assertCountEqual(
            skeleton.edges[("d", "e")].get("matched_edge"), (("B", "C"), ("D", "E"))
        )
        self.assertEqual(skeleton.edges[("e", "g")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("e", "h")].get("matched_edge"), ("D", "E"))
        """

    def test_crossover(self):

        # consensus graph:
        #
        #     B-------->C------->D
        #   /
        # A
        #   \
        #     E-------->F------->G
        #
        # skeleton graph:
        #
        #     b       e----------g
        #   /   \   /
        # a       d
        #   \   /   \
        #     c       f----------h
        #
        # should this be possible or not? Depends on the problem.
        # if you want to be able to represent multiple chains crossing through 1 node,
        # remove the constraint on no matched node degree

        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("A", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 1, 10])}),
                ("C", {"location": np.array([0, 1, 20])}),
                ("D", {"location": np.array([0, 1, 30])}),
                ("E", {"location": np.array([0, -1, 10])}),
                ("F", {"location": np.array([0, -1, 20])}),
                ("G", {"location": np.array([0, -1, 30])}),
            ]
        )
        consensus.add_edges_from(
            [("A", "B"), ("B", "C"), ("C", "D"), ("A", "E"), ("E", "F"), ("F", "G")]
        )

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("a", {"location": np.array([1, 0, 0])}),
                ("b", {"location": np.array([1, 1, 10])}),
                ("c", {"location": np.array([1, -1, 10])}),
                ("d", {"location": np.array([1, 0, 15])}),
                ("e", {"location": np.array([1, 1, 20])}),
                ("f", {"location": np.array([1, -1, 20])}),
                ("g", {"location": np.array([1, 1, 30])}),
                ("h", {"location": np.array([1, -1, 30])}),
            ]
        )
        skeleton.add_edges_from(
            [
                ("a", "b"),
                ("b", "d"),
                ("a", "c"),
                ("c", "d"),
                ("d", "e"),
                ("d", "f"),
                ("e", "g"),
                ("f", "h"),
            ]
        )

        self.assertRaises(
            ValueError,
            nl.match_graph_to_tree,
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        # if crossovers were allowed, this would be the expected result
        """
        self.assertEqual(skeleton.edges[("a", "b")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("a", "c")].get("matched_edge"), ("A", "E"))
        self.assertEqual(skeleton.edges[("b", "d")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("c", "d")].get("matched_edge"), ("E", "F"))
        self.assertEqual(skeleton.edges[("d", "e")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("d", "f")].get("matched_edge"), ("E", "F"))
        self.assertEqual(skeleton.edges[("e", "g")].get("matched_edge"), ("C", "D"))
        self.assertEqual(skeleton.edges[("f", "h")].get("matched_edge"), ("F", "G"))
        """

    def test_bidirected_edges(self):

        # consensus graph:
        #
        #     D E F
        #      \|/
        # A---->B---->C
        #
        # skeleton graph:
        #
        #       e
        #       |
        #    d--b--f
        #       |
        #   a---g---c
        #
        # This is only possible if b->B and
        # a-g-b -> AB and b-g-a -> BC

        consensus = nx.DiGraph()
        consensus.add_nodes_from(
            [
                ("A", {"location": np.array([0, 0, 0])}),
                ("B", {"location": np.array([0, 0, 10])}),
                ("C", {"location": np.array([0, 0, 20])}),
                ("D", {"location": np.array([0, 10, 0])}),
                ("E", {"location": np.array([0, 10, 10])}),
                ("F", {"location": np.array([0, 10, 20])}),
            ]
        )
        consensus.add_edges_from(
            [("A", "B"), ("B", "C"), ("B", "D"), ("B", "E"), ("B", "F")]
        )

        skeleton = nx.Graph()
        skeleton.add_nodes_from(
            [
                ("a", {"location": np.array([1, 0, 0])}),
                ("g", {"location": np.array([1, -1, 10])}),
                ("b", {"location": np.array([1, 0, 10])}),
                ("c", {"location": np.array([1, 0, 20])}),
                ("d", {"location": np.array([1, 10, 0])}),
                ("e", {"location": np.array([1, 10, 10])}),
                ("f", {"location": np.array([1, 10, 20])}),
            ]
        )
        skeleton.add_edges_from(
            [("a", "g"), ("g", "b"), ("g", "c"), ("b", "d"), ("b", "e"), ("b", "f")]
        )

        self.assertRaises(
            ValueError,
            nl.match_graph_to_tree,
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        # expected output if bidirectional edges were allowed:
        """
        self.assertEqual(skeleton.edges[("a", "g")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("g", "c")].get("matched_edge"), ("B", "C"))
        self.assertCountEqual(
            skeleton.edges[("g", "b")].get("matched_edge"), (("A", "B"), ("B", "C"))
        )
        self.assertEqual(skeleton.edges[("b", "d")].get("matched_edge"), ("B", "D"))
        self.assertEqual(skeleton.edges[("b", "e")].get("matched_edge"), ("B", "E"))
        self.assertEqual(skeleton.edges[("b", "f")].get("matched_edge"), ("B", "F"))
        """

    def test_multi_edge_match(self):

        # consensus graph:
        #
        # A---->B---->C
        #
        # skeleton graph:
        #
        # a--b--c--d--e
        #
        # should it be possible to map both a-b and b-c to A-B?
        # If you want to match subgraph isomorphisms exactly,
        # make sure the unique edge matching constraint is on

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

        self.assertEqual(skeleton.edges[("a", "b")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("b", "c")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("c", "d")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("d", "e")].get("matched_edge"), ("B", "C"))

        # expected output if multi_edge_matches weren't allowed:
        """
        nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_distance_threshold=100,
            match_attribute="matched_edge",
        )

        self.assertEqual(skeleton.edges[("a", "b")].get("matched_edge"), None)
        self.assertEqual(skeleton.edges[("b", "c")].get("matched_edge"), ("A", "B"))
        self.assertEqual(skeleton.edges[("c", "d")].get("matched_edge"), ("B", "C"))
        self.assertEqual(skeleton.edges[("d", "e")].get("matched_edge"), None)
        """


