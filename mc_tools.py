"""
Authors: Chase Coleman, Spencer Lyon, Daisuke Oyama, Tom Sargent,
         John Stachurski

Filename: mc_tools.py

This file contains some useful objects for handling a finite-state
discrete-time Markov chain.  It contains code written by several people
and was ultimately compiled into a single file to take advantage of the
pros of each.

"""
from __future__ import division
import numpy as np
import sys
#from .discrete_rv import DiscreteRV
from discrete_rv import DiscreteRV
#from .graph_tools import DiGraph
from graph_tools import DiGraph
#from .gth_solve import gth_solve
from gth_solve import gth_solve
from warnings import warn


class MarkovChain(object):
    """
    Class for a finite-state discrete-time Markov chain. It stores
    useful information such as the stationary distributions and allows
    simulation of state transitions.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        The transition matrix.  Must be of shape n x n.

    Attributes
    ----------
    P : array_like(float, ndim=2)
        The transition matrix.

    stationary_distributions : array_like(float, ndim=2)
        Array containing the stationary distributions as rows.

    is_irreducible : bool
        Indicate whether the Markov chain is irreducible.

    num_communication_classes : int
        The number of the communication classes.

    communication_classes : list(ndarray(int))
        List of numpy arrays containing the communication classes.

    num_recurrent_classes : int
        The number of the recurrent classes.

    recurrent_classes : list(ndarray(int))
        List of numpy arrays containing the recurrent classes.

    is_aperiodic : bool
        Indicate whether the Markov chain is aperiodic. Defined only
        when the Markov chain is irreducible.

    period : int
        The period of the Markov chain. Defined only when the Markov
        chain is irreducible.

    cyclic_classes : list(ndarray(int))
        List of numpy arrays containing the cyclic classes. Defined only
        when the Markov chain is irreducible.

    Methods
    -------
    simulate : Simulates the markov chain for a given initial
        state or distribution

    """

    def __init__(self, P):
        self.P = np.asarray(P)
        n, m = self.P.shape

        # Check Properties
        # double check that P is a square matrix
        if n != m:
            raise ValueError('P must be square')

        # Double check that the rows of P sum to one
        if not np.allclose(np.sum(self.P, axis=1), np.ones(self.P.shape[0])):
            raise ValueError('The rows of P must sum to 1')

        # The number of states
        self.n = n

        # To analyze the structure of P as a directed graph
        self.digraph = DiGraph(P)

        self._stationary_dists = None

    def __repr__(self):
        msg = "Markov chain with transition matrix \nP = \n{0}"

        if self._stationary_dists is None:
            return msg.format(self.P)
        else:
            msg = msg + "\nand stationary distributions \n{1}"
            return msg.format(self.P, self._stationary_dists)

    def __str__(self):
        return str(self.__repr__)

    @property
    def is_irreducible(self):
        return self.digraph.is_strongly_connected

    @property
    def num_communication_classes(self):
        return self.digraph.num_strongly_connected_components

    @property
    def communication_classes(self):
        return self.digraph.strongly_connected_components

    @property
    def num_recurrent_classes(self):
        return self.digraph.num_sink_strongly_connected_components

    @property
    def recurrent_classes(self):
        return self.digraph.sink_strongly_connected_components

    @property
    def is_aperiodic(self):
        if not self.is_irreducible:
            raise NotImplementedError(
                'Not defined for a reducible Markov chain'
            )
        else:
            return self.digraph.is_aperiodic

    @property
    def period(self):
        if not self.is_irreducible:
            raise NotImplementedError(
                'Not defined for a reducible Markov chain'
            )
        else:
            return self.digraph.period

    @property
    def cyclic_classes(self):
        if not self.is_irreducible:
            raise NotImplementedError(
                'Not defined for a reducible Markov chain'
            )
        else:
            return self.digraph.cyclic_components

    def _compute_stationary(self):
        """
        Store the stationary distributions in self._stationary_distributions.

        """
        if self.is_irreducible:
            stationary_dists = gth_solve(self.P).reshape(1, self.n)
        else:
            rec_classes = self.recurrent_classes
            stationary_dists = np.zeros((len(rec_classes), self.n))
            for i, rec_class in enumerate(rec_classes):
                stationary_dists[i, rec_class] = \
                    gth_solve(self.P[rec_class, :][:, rec_class])

        self._stationary_dists = stationary_dists

    @property
    def stationary_distributions(self):
        if self._stationary_dists is None:
            self._compute_stationary()
        return self._stationary_dists

    def simulate(self, init=0, sample_size=1000):
        sim = mc_sample_path(self.P, init, sample_size)

        return sim


def mc_compute_stationary(P):
    """
    Computes the stationary distributions of P. These are the left
    eigenvectors that correspond to the unit eigenvalues of the
    matrix P' (They satisfy x = x P).

    Returns
    -------
    stationary_dists : array_like(float, ndim=2)
        Array containing the stationary distributions as its rows.

    """
    return MarkovChain(P).stationary_distributions


def mc_sample_path(P, init=0, sample_size=1000):
    # === set up array to store output === #
    X = np.empty(sample_size, dtype=int)
    if isinstance(init, int):
        X[0] = init
    else:
        X[0] = DiscreteRV(init).draw()

    # === turn each row into a distribution === #
    # In particular, let P_dist[i] be the distribution corresponding to the
    # i-th row P[i,:]
    n = len(P)
    P_dist = [DiscreteRV(P[i,:]) for i in range(n)]

    # === generate the sample path === #
    for t in range(sample_size - 1):
        X[t+1] = P_dist[X[t]].draw()

    return X


#---------------------------------------------------------------------#
# Set up the docstrings for the functions
#---------------------------------------------------------------------#

# For drawing a sample path
_sample_path_docstr = \
"""
Generates one sample path from a finite Markov chain with (n x n)
transition matrix P on state space S = {{0,...,n-1}}.

Parameters
----------
{p_arg}init : array_like(float ndim=1) or scalar(int)
    If init is an array_like then it is treated as the initial
    distribution across states.  If init is a scalar then it
    treated as the deterministic initial state.

sample_size : scalar(int), optional(default=1000)
    The length of the sample path.

Returns
-------
X : array_like(int, ndim=1)
    The simulation of states

"""

# set docstring for functions
mc_sample_path.__doc__ = _sample_path_docstr.format(p_arg=
"""P : array_like(float, ndim=2)
    A discrete Markov transition matrix

""")

# set docstring for methods

if sys.version_info[0] == 3:
    MarkovChain.simulate.__doc__ = _sample_path_docstr.format(p_arg="")
elif sys.version_info[0] == 2:
    MarkovChain.simulate.__func__.__doc__ = _sample_path_docstr.format(p_arg="")
