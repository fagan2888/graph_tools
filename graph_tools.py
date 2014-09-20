"""
Filename: graph_tools.py

Author: Daisuke Oyama

Tools for dealing with a graph (preliminary).

TODO:
* Modify the docstrings
* Change the names to fit in the context of graph theory

"""
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from fractions import gcd


class Digraph:
    r"""
    Class for directed graphs. In particular, implement methods that
    find communication classes and reccurent classes.

    Parameters
    ----------
    input_array : array_like(float, ndim=2)
        Array representing a stochastic matrix. Must be of shape n x n.

    Attributes
    ----------
    is_irreducible : bool
        Indicate whether the array is an irreducible matrix.

    num_comm_classes : int
        Number of communication classes.

    num_rec_classes : int
        Number of recurrent classes.

    """

    def __init__(self, input):
        self.csr_matrix = sparse.csr_matrix(input)

        m, n = self.csr_matrix.shape
        if n != m:
            raise ValueError('matrix must be square')

        self.n = n

        self._num_comm_classes = None
        self._comm_classes_proj = None
        self._rec_classes_labels = None

    def _find_comm_classes(self):
        """
        Set ``self._num_comm_classes`` and ``self._comm_classes_proj``
        by calling ``scipy.sparse.csgraph.connected_components``:
        * docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
        * github.com/scipy/scipy/blob/master/scipy/sparse/csgraph/_traversal.pyx

        ``self._comm_classes_proj`` is a list of length `n` that assigns
        to each index the label of the communication class to which it belongs.

        """
        # Find the communication classes (strongly connected components)
        self._num_comm_classes, self._comm_classes_proj = \
            csgraph.connected_components(self.csr_matrix, connection='strong')

    @property
    def num_comm_classes(self):
        if self._num_comm_classes is None:
            self._find_comm_classes()
        return self._num_comm_classes

    @property
    def comm_classes_proj(self):
        if self._comm_classes_proj is None:
            self._find_comm_classes()
        return self._comm_classes_proj

    @property
    def is_irreducible(self):
        return (self.num_comm_classes == 1)

    def _find_rec_classes(self):
        """
        Set self._rec_classes_labels, which is a list containing the labels of
        the recurrent communication classes.

        """
        # Directed graph on the quotient set (the set of communication classes)
        # represented by dictionary of sets
        digraph_quo = {k: set() for k in range(self.num_comm_classes)}

        comm_classes_proj = self.comm_classes_proj
        for state_from, state_to in _csr_matrix_indices(self.csr_matrix):
            comm_class_from, comm_class_to = \
                comm_classes_proj[state_from], comm_classes_proj[state_to]
            if comm_class_from != comm_class_to:
                digraph_quo[comm_class_from].add(comm_class_to)

        # A recurrent class is a communication class such that none of
        # its members communicates with states in other classes
        self._rec_classes_labels = \
            [k for k, val in digraph_quo.items() if len(val) == 0]

    @property
    def rec_classes_labels(self):
        if self._rec_classes_labels is None:
            self._find_rec_classes()
        return self._rec_classes_labels

    @property
    def num_rec_classes(self):
        return len(self.rec_classes_labels)

    def comm_classes(self):
        r"""
        Return the communication classes (strongly connected components).

        Returns
        -------
        list(list(int))
            List of lists containing the communication classes

        """
        if self.is_irreducible:
            return [list(range(self.n))]
        else:
            return [np.where(self.comm_classes_proj == k)[0].tolist()
                    for k in range(self.num_comm_classes)]

    def rec_classes(self):
        r"""
        Return the recurrent classes (closed communication classes).

        Returns
        -------
        list(list(int))
            List of lists containing the recurrent classes

        """
        if self.is_irreducible:
            return [list(range(self.n))]
        else:
            return [np.where(self.comm_classes_proj == k)[0].tolist()
                    for k in self.rec_classes_labels]

    def period(self):
        if not self.is_irreducible:
            raise NotImplementedError(
                'period is not defined for a reducible graph'
                )

        if np.any(self.csr_matrix.diagonal() > 0):
            return 1, np.zeros(self.n)

        # Construct a breadth first tree rooted at 0
        node_order, predecessors = \
            csgraph.breadth_first_order(self.csr_matrix, i_start=0)
        tree_csr = \
            csgraph.reconstruct_path(self.csr_matrix, predecessors).astype(bool)

        # Edges not belonging to tree_csr
        non_tree_csr = self.csr_matrix - tree_csr
        non_tree_csr.eliminate_zeros()

        # Distance to 0
        level = np.zeros(self.n, dtype=int)
        for i in range(1, self.n):
            level[node_order[i]] = level[predecessors[node_order[i]]] + 1

        # Determine the period
        d = 0
        for edge_from, edge_to in _csr_matrix_indices(non_tree_csr):
            value = level[edge_from] - level[edge_to] + 1
            d = gcd(d, value)

        return d, level % d


def _csr_matrix_indices(S):
    """
    Generate the indices of nonzero entries of a csr_matrix S
    """
    m, n = S.shape

    for i in range(m):
        for j in range(S.indptr[i], S.indptr[i+1]):
            row_index, col_index = i, S.indices[j]
            yield row_index, col_index
