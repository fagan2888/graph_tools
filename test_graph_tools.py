"""
Filename: test_graph_tools.py
Author: Daisuke Oyama

Tests for graph_tools.py

"""
import sys
import numpy as np
from numpy.testing import assert_array_equal
import nose
from nose.tools import eq_, ok_

from graph_tools import DiGraph


class Graphs:
    """Setup graphs for the tests"""

    def __init__(self):
        self.strongly_connected_graph_dicts = []
        self.not_strongly_connected_graph_dicts = []

        graph_dict = {
            'A': np.array([[1, 0], [0, 1]]),
            'strongly_connected_components':
                [np.array([0]), np.array([1])],
            'sink_strongly_connected_components':
                [np.array([0]), np.array([1])],
            'is_strongly_connected': False,
        }
        self.not_strongly_connected_graph_dicts.append(graph_dict)

        graph_dict = {
            'A': np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]]),
            'strongly_connected_components':
                [np.array([0]), np.array([1]), np.array([2])],
            'sink_strongly_connected_components':
                [np.array([0]), np.array([2])],
            'is_strongly_connected': False,
        }
        self.not_strongly_connected_graph_dicts.append(graph_dict)

        graph_dict = {
            'A': np.array([[0, 1], [1, 0]]),
            'strongly_connected_components': [np.arange(2)],
            'sink_strongly_connected_components': [np.arange(2)],
            'is_strongly_connected': True,
            'period': 2,
            'is_aperiodic': False,
            'cyclic_components': [np.array([0]), np.array([1])],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        # Degenrate graph with no edge
        graph_dict = {
            'A': np.array([[0]]),
            'strongly_connected_components': [np.arange(1)],
            'sink_strongly_connected_components': [np.arange(1)],
            'is_strongly_connected': True,
            'period': 1,
            'is_aperiodic': True,
            'cyclic_components': [np.array([0])],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        # Degenrate graph with self loop
        graph_dict = {
            'A': np.array([[1]]),
            'strongly_connected_components': [np.arange(1)],
            'sink_strongly_connected_components': [np.arange(1)],
            'is_strongly_connected': True,
            'period': 1,
            'is_aperiodic': True,
            'cyclic_components': [np.array([0])],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        self.graph_dicts = \
            self.strongly_connected_graph_dicts + \
            self.not_strongly_connected_graph_dicts


class TestDiGraph:
    """Test the methods in Digraph"""

    def setUp(self):
        """Setup Digraph instances"""
        self.graphs = Graphs()
        for graph_dict in self.graphs.graph_dicts:
            graph_dict['DiGraph'] = DiGraph(graph_dict['A'])

    def test_strongly_connected_components(self):
        for graph_dict in self.graphs.graph_dicts:
            assert_array_equal(
                sorted(graph_dict['DiGraph'].strongly_connected_components),
                sorted(graph_dict['strongly_connected_components']))

    def test_num_strongly_connected_components(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['DiGraph'].num_strongly_connected_components,
                len(graph_dict['strongly_connected_components']))

    def test_sink_strongly_connected_components(self):
        for graph_dict in self.graphs.graph_dicts:
            assert_array_equal(
                sorted(graph_dict['DiGraph'].sink_strongly_connected_components),
                sorted(graph_dict['sink_strongly_connected_components']))

    def test_num_sink_strongly_connected_components(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['DiGraph'].num_sink_strongly_connected_components,
                len(graph_dict['sink_strongly_connected_components']))

    def test_is_strongly_connected(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['DiGraph'].is_strongly_connected,
                graph_dict['is_strongly_connected'])

    def test_period(self):
        for graph_dict in self.graphs.graph_dicts:
            try:
                eq_(graph_dict['DiGraph'].period, graph_dict['period'])
            except NotImplementedError:
                eq_(graph_dict['DiGraph'].is_strongly_connected, False)

    def test_is_aperiodic(self):
        for graph_dict in self.graphs.graph_dicts:
            try:
                eq_(graph_dict['DiGraph'].is_aperiodic,
                    graph_dict['is_aperiodic'])
            except NotImplementedError:
                eq_(graph_dict['DiGraph'].is_strongly_connected, False)

    def test_cyclic_components(self):
        for graph_dict in self.graphs.graph_dicts:
            try:
                assert_array_equal(
                    sorted(graph_dict['DiGraph'].cyclic_components),
                    sorted(graph_dict['cyclic_components']))
            except NotImplementedError:
                eq_(graph_dict['DiGraph'].is_strongly_connected, False)


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
