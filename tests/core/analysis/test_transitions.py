"""Test transition estimates, stationary laws, spectral gaps, relaxation times, and detailed balance."""

import math

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import (
    detailed_balance,
    relaxation_times,
    spectral_gap,
    stationary_distribution,
    transition_counts,
    transition_matrix,
)
from skcomplex.core.base import AnalysisError


def _chain(matrix: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Return a path of a discrete-time chain with a transition matrix, started in state 0."""
    cumulative = np.cumsum(matrix, axis=1)
    draws = rng.random(steps)
    states = np.zeros(steps, dtype=int)
    for t in range(1, steps):
        states[t] = int(np.searchsorted(cumulative[states[t - 1]], draws[t]))
    return states


def test_stationary_distribution_ehrenfest_urn_and_ring_walk():
    """Test the binomial law of the Ehrenfest urn and the spectral gap of the lazy walk on a cycle of 11 nodes."""
    balls = 20
    rates = np.zeros((balls + 1, balls + 1))
    for count in range(balls + 1):
        if count > 0:
            rates[count, count - 1] = count
        if count < balls:
            rates[count, count + 1] = balls - count
    binomial = np.asarray([math.comb(balls, count) for count in range(balls + 1)]) / 2**balls
    np.testing.assert_allclose(stationary_distribution(rates, rates=True), binomial, atol=1e-10)
    adjacency = nx.to_numpy_array(nx.cycle_graph(11))
    walk = 0.5 * np.eye(11) + 0.5 * adjacency / 2
    assert spectral_gap(walk) == pytest.approx((1 - math.cos(2 * math.pi / 11)) / 2, abs=1e-10)
    assert relaxation_times(walk)[0] == pytest.approx(-1 / math.log(1 - spectral_gap(walk)), rel=1e-9)


def test_spectral_gap_of_rates():
    """Test the gap and relaxation time of a two-state rate matrix with rates 1 and 2."""
    rates = [[0.0, 1.0], [2.0, 0.0]]
    assert spectral_gap(rates, rates=True) == pytest.approx(3)
    assert relaxation_times(rates, rates=True).tolist() == pytest.approx([1 / 3])


def test_transition_matrix_estimates_a_known_chain():
    """Test that a long path recovers its transition matrix and counts transitions at a lag."""
    matrix = np.asarray([[0.9, 0.1], [0.3, 0.7]])
    path = _chain(matrix, 50_000, np.random.default_rng(0))
    np.testing.assert_allclose(transition_matrix(path), matrix, atol=0.02)
    assert transition_counts(path, lag=2).sum() == path.size - 2
    assert transition_matrix([0, 0], n_states=2)[1].tolist() == [0.0, 1.0]


def test_detailed_balance_of_reversible_and_driven_chains():
    """Test that a birth-death chain passes detailed balance and a driven cycle fails with a nonzero current."""
    rng = np.random.default_rng(1)
    size = 5
    birth_death = np.zeros((size, size))
    for state in range(size):
        if state < size - 1:
            birth_death[state, state + 1] = 0.3
        if state > 0:
            birth_death[state, state - 1] = 0.4
        birth_death[state, state] = 1 - birth_death[state].sum()
    cycle = np.asarray([[0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.6, 0.2, 0.2]])
    reversible = detailed_balance(_chain(birth_death, 50_000, rng))
    driven = detailed_balance(_chain(cycle, 50_000, rng))
    assert reversible.balanced
    assert not driven.balanced
    assert driven.currents[0, 1] == pytest.approx(0.4 / 3, abs=0.01)


def test_stationary_distribution_rejects_non_square_matrices():
    """Test that a matrix that is not square raises an error."""
    with pytest.raises(AnalysisError, match='square'):
        stationary_distribution([[1.0, 0.0]])
