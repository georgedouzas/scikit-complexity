"""Test extreme-value fits, record counts, stable fits, and goodness of fit to named laws."""

import numpy as np
import pytest

from skcomplex.core.analysis import compare_with_law, fit_extreme_values, fit_stable, record_counts, snapshots_of
from skcomplex.core.base import AnalysisError
from skcomplex.models import run


def _random_exchange_economy(rng: np.random.Generator, agents: int, exchanges: int) -> np.ndarray:
    """Return the money of agents after random pairs pool their money and split it at a uniform fraction."""
    money = np.ones(agents)
    for _ in range(exchanges):
        order = rng.permutation(agents)
        first, second = order[: agents // 2], order[agents // 2 :]
        pooled = money[first] + money[second]
        share = rng.random(first.size)
        money[first], money[second] = share * pooled, (1 - share) * pooled
    return money


def test_fit_extreme_values_classes():
    """Test that block maxima of exponential, Pareto, and uniform samples fall in the three extreme-value classes."""
    rng = np.random.default_rng(0)
    size, block = 1_000_000, 1000
    assert fit_extreme_values(rng.exponential(size=size), block).shape == pytest.approx(0, abs=0.05)
    assert fit_extreme_values((1 - rng.random(size)) ** (-1 / 2), block).shape == pytest.approx(0.5, abs=0.1)
    assert fit_extreme_values(rng.random(size), block).shape == pytest.approx(-1, abs=0.1)
    with pytest.raises(AnalysisError, match='at least 10 blocks'):
        fit_extreme_values(np.zeros(50), 10)


def test_record_counts_of_independent_values():
    """Test that 10,000 sequences of 1000 independent values hold the harmonic number of records on average."""
    counts = record_counts(np.random.default_rng(1).random((10_000, 1000)))
    harmonic = np.sum(1 / np.arange(1, 1001))
    assert counts.mean() == pytest.approx(harmonic, rel=0.01)


def test_fit_stable_of_cauchy_sums():
    """Test that normalized sums of Cauchy values fit a stable law of index 1."""
    sums = np.random.default_rng(2).standard_cauchy((2000, 10)).sum(axis=1) / 10
    assert fit_stable(sums).alpha == pytest.approx(1, abs=0.15)


def test_compare_with_law_boltzmann_law_on_two_domains():
    """Test that gas energies and exchanged money are exponential at the energy per molecule and money per agent."""
    snapshots = snapshots_of(run('ideal_gas').trajectory_)
    energies = 0.5 * np.sum(snapshots.velocities**2, axis=-1)
    gas = compare_with_law(energies[-1], fixed={'loc': 0})
    assert gas.p_value > 0.05
    assert gas.parameters['scale'] == pytest.approx(energies[0].mean(), rel=0.02)
    money = _random_exchange_economy(np.random.default_rng(3), 2000, 400)
    economy = compare_with_law(money, fixed={'loc': 0})
    assert economy.p_value > 0.05
    assert economy.parameters['scale'] == pytest.approx(1, rel=0.02)
    with pytest.raises(AnalysisError, match='Unknown law'):
        compare_with_law(money, 'zipf')
