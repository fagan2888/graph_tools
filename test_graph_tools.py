"""
Filename: test_graph_tools.py
Author: Daisuke Oyama

Tests for graph_tools.py

"""
import sys
import numpy as np
import nose
from nose.tools import eq_, ok_

from graph_tools import Digraph


class Graphs:
    """Setup graphs for the tests"""

    def __init__(self):
        self.strongly_connected_graph_dicts = []
        self.not_strongly_connected_graph_dicts = []

        graph_dict = {
            'A': np.array([[1, 0], [0, 1]]),
            'comm_classes': [[0], [1]],
            'rec_classes': [[0], [1]],
            'is_strongly_connected': False,
            }
        self.not_strongly_connected_graph_dicts.append(graph_dict)

        graph_dict = {
            'A': np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]]),
            'comm_classes': [[0], [1], [2]],
            'rec_classes': [[0], [2]],
            'is_strongly_connected': False,
            }
        self.not_strongly_connected_graph_dicts.append(graph_dict)

        graph_dict = {
            'A': np.array([[0, 1], [1, 0]]),
            'comm_classes': [list(range(2))],
            'rec_classes': [list(range(2))],
            'is_strongly_connected': True,
            'period': 2,
            }
        self.strongly_connected_graph_dicts.append(graph_dict)

        # Degenrate graph with no edge
        graph_dict = {
            'A': np.array([[0]]),
            'comm_classes': [list(range(1))],
            'rec_classes': [list(range(1))],
            'is_strongly_connected': True,
            'period': 0,
            }
        self.strongly_connected_graph_dicts.append(graph_dict)

        # Degenrate graph with self loop
        graph_dict = {
            'A': np.array([[1]]),
            'comm_classes': [list(range(1))],
            'rec_classes': [list(range(1))],
            'is_strongly_connected': True,
            'period': 1,
            }
        self.strongly_connected_graph_dicts.append(graph_dict)

        self.graph_dicts = \
            self.strongly_connected_graph_dicts + \
            self.not_strongly_connected_graph_dicts


class TestDigraph:
    """Test the methods in Digraph"""

    def setUp(self):
        """Setup Digraph instances"""
        self.graphs = Graphs()
        for graph_dict in self.graphs.graph_dicts:
            graph_dict['A'] = Digraph(graph_dict['A'])

    def test_comm_classes(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(sorted(graph_dict['A'].comm_classes()),
                sorted(graph_dict['comm_classes']))

    def test_num_comm_classes(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['A'].num_comm_classes,
                len(graph_dict['comm_classes']))

    def test_rec_classes(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(sorted(graph_dict['A'].rec_classes()),
                sorted(graph_dict['rec_classes']))

    def test_num_rec_classes(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['A'].num_rec_classes,
                len(graph_dict['rec_classes']))

    def test_is_strongly_connected(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['A'].is_strongly_connected,
                graph_dict['is_strongly_connected'])

    def test_period(self):
        for graph_dict in self.graphs.graph_dicts:
            try:
                period, cyc_classes_labels = graph_dict['A'].period()
                eq_(period, graph_dict['period'])
            except NotImplementedError:
                eq_(graph_dict['A'].is_strongly_connected, False)


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
