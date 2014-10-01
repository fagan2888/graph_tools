"""
Filename: graph_tools.py

Author: Daisuke Oyama

Tools for dealing with a graph (preliminary).

TODO:
* Elaborate the docstrings

"""
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from fractions import gcd


class DiGraph:
    r"""
    Class for directed graphs. In particular, implement methods that
    find strongly connected components.

    Parameters
    ----------
    adj_matrix : array_like(ndim=2)
        Adjacency matrix representing a directed graph. Must be of shape
        n x n.

    weighted : bool

    Attributes
    ----------
    csgraph : csr_matrix
        Compressed sparse representation of the digraph.

    is_strongly_connected : bool
        Indicate whether the digraph is strongly connected.

    num_strongly_connected_components : int
        The number of the strongly connected components.

    strongly_connected_components : list(ndarray(int))
        List of numpy arrays containing the strongly connected
        components.

    num_sink_strongly_connected_components : int
        The number of the sink strongly connected components.

    sink_strongly_connected_components : list(ndarray(int))
        List of numpy arrays containing the sink strongly connected
        components.

    is_aperiodic : bool
        Indicate whether the digraph is aperiodic.

    period : int
        The period of the digraph. Defined only for a strongly connected
        digraph.

    cyclic_components : list(ndarray(int))
        List of numpy arrays containing the cyclic components.

    """

    def __init__(self, adj_matrix, weighted=False):
        if weighted:
            dtype = None
        else:
            dtype = bool
        self.csgraph = sparse.csr_matrix(adj_matrix, dtype=dtype)

        m, n = self.csgraph.shape
        if n != m:
            raise ValueError('input matrix must be square')

        self.n = n  # Number of nodes

        self._num_scc = None
        self._scc_proj = None
        self._sink_scc_labels = None

        self._period = None

    def subgraph(self, nodes):
        subgraph_csr = self.csgraph[nodes, :][:, nodes]
        h = self.__class__(subgraph_csr)
        return h

    def _find_scc(self):
        """
        Set ``self._num_scc`` and ``self._scc_proj``
        by calling ``scipy.sparse.csgraph.connected_components``:
        * docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
        * github.com/scipy/scipy/blob/master/scipy/sparse/csgraph/_traversal.pyx

        ``self._scc_proj`` is a list of length `n` that assigns to each node
        the label of the strongly connected component to which it belongs.

        """
        # Find the strongly connected components
        self._num_scc, self._scc_proj = \
            csgraph.connected_components(self.csgraph, connection='strong')

    @property
    def num_strongly_connected_components(self):
        if self._num_scc is None:
            self._find_scc()
        return self._num_scc

    @property
    def scc_proj(self):
        if self._scc_proj is None:
            self._find_scc()
        return self._scc_proj

    @property
    def is_strongly_connected(self):
        return (self.num_strongly_connected_components == 1)

    def _condensation_lil(self):
        """
        Return the sparse matrix representation of the condensation digraph
        in lil format
        """
        condensation_lil = sparse.lil_matrix(
            (self.num_strongly_connected_components,
             self.num_strongly_connected_components), dtype=bool
        )

        scc_proj = self.scc_proj
        for node_from, node_to in _csr_matrix_indices(self.csgraph):
            scc_from, scc_to = scc_proj[node_from], scc_proj[node_to]
            if scc_from != scc_to:
                condensation_lil[scc_from, scc_to] = True

        return condensation_lil

    def _find_sink_scc(self):
        """
        Set self._sink_scc_labels, which is a list containing the labels of
        the strongly connected components.

        """
        condensation_lil = self._condensation_lil()

        # A sink SCC is a SCC such that none of its members is strongly
        # connected to nodes in other SCCs
        # Those k's such that graph_condensed_lil.rows[k] == []
        self._sink_scc_labels = \
            np.where(np.logical_not(condensation_lil.rows))[0]

    @property
    def sink_scc_labels(self):
        if self._sink_scc_labels is None:
            self._find_sink_scc()
        return self._sink_scc_labels

    @property
    def num_sink_strongly_connected_components(self):
        return len(self.sink_scc_labels)

    @property
    def strongly_connected_components(self):
        if self.is_strongly_connected:
            return [np.arange(self.n)]
        else:
            return [np.where(self.scc_proj == k)[0]
                    for k in range(self.num_strongly_connected_components)]

    @property
    def sink_strongly_connected_components(self):
        if self.is_strongly_connected:
            return [np.arange(self.n)]
        else:
            return [np.where(self.scc_proj == k)[0]
                    for k in self.sink_scc_labels.tolist()]

    def _compute_period(self):
        r"""
        Set ``self._period`` and ``self._cyclic_components_proj``.

        """
        # Degenerate graph with a single node (which is strongly connected)
        # csgraph.reconstruct_path would raise an exception
        if self.n == 1:
            if self.csgraph[0, 0] == 0:  # No edge: "trivial graph"
                self._period = 1  # Any universally accepted definition?
                self._cyclic_components_proj = np.zeros(self.n, dtype=int)
                return None
            else:  # Self loop
                self._period = 1
                self._cyclic_components_proj = np.zeros(self.n, dtype=int)
                return None

        if not self.is_strongly_connected:
            raise NotImplementedError(
                'Not defined for a non strongly-connected digraph'
            )

        if np.any(self.csgraph.diagonal() > 0):
            self._period = 1
            self._cyclic_components_proj = np.zeros(self.n, dtype=int)
            return None

        # Construct a breadth-first search tree rooted at 0
        node_order, predecessors = \
            csgraph.breadth_first_order(self.csgraph, i_start=0)
        bfs_tree_csr = \
            csgraph.reconstruct_path(self.csgraph, predecessors)

        # Edges not belonging to tree_csr
        non_bfs_tree_csr = self.csgraph - bfs_tree_csr
        non_bfs_tree_csr.eliminate_zeros()

        # Distance to 0
        level = np.zeros(self.n, dtype=int)
        for i in range(1, self.n):
            level[node_order[i]] = level[predecessors[node_order[i]]] + 1

        # Determine the period
        d = 0
        for node_from, node_to in _csr_matrix_indices(non_bfs_tree_csr):
            value = level[node_from] - level[node_to] + 1
            d = gcd(d, value)
            if d == 1:
                self._period = 1
                self._cyclic_components_proj = np.zeros(self.n, dtype=int)
                return None

        self._period = d
        self._cyclic_components_proj = level % d

    @property
    def period(self):
        if self._period is None:
            self._compute_period()
        return self._period

    @property
    def is_aperiodic(self):
        return (self.period == 1)

    @property
    def cyclic_components(self):
        if self.is_aperiodic:
            return [np.arange(self.n)]
        else:
            return [np.where(self._cyclic_components_proj == k)[0]
                    for k in range(self.period)]


def _csr_matrix_indices(S):
    """
    Generate the indices of nonzero entries of a csr_matrix S

    """
    m, n = S.shape

    for i in range(m):
        for j in range(S.indptr[i], S.indptr[i+1]):
            row_index, col_index = i, S.indices[j]
            yield row_index, col_index
