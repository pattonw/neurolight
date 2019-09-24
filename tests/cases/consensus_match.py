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
        consensus.add_nodes_from([
            (1, {'location': np.array([0, 0, 0])}),
            (2, {'location': np.array([0, 0, 10])}),
            (3, {'location': np.array([0, 0, 20])})])
        consensus.add_edges_from([
            (1, 2),
            (2, 3)])

        skeleton = nx.Graph()
        skeleton.add_nodes_from([
            (1, {'location': np.array([0, 0, 0])}),
            (2, {'location': np.array([0, 0, 5])}),
            (3, {'location': np.array([0, 0, 10])}),
            (4, {'location': np.array([0, 0, 15])}),
            (5, {'location': np.array([0, 0, 20])})])
        skeleton.add_edges_from([
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5)])

        nl.match_graph_to_tree(
            skeleton,
            consensus,
            match_attribute='matched_edge')

        self.assertEqual(skeleton.edges[(1, 2)]['matched_edge'] == (1, 2))
        self.assertEqual(skeleton.edges[(2, 3)]['matched_edge'] == (1, 2))
        self.assertEqual(skeleton.edges[(3, 4)]['matched_edge'] == (2, 3))
        self.assertEqual(skeleton.edges[(4, 5)]['matched_edge'] == (2, 3))
