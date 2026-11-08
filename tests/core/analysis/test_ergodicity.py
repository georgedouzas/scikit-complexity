"""Test the ergodicity analysis of ensembles of runs and of invariant densities."""

import numpy as np
import pytest

from skcomplex.core.analysis import ErgodicityResult, ergodicity
from skcomplex.core.errors import AnalysisError

RUNS = 200
STEPS = 20_000
RELAXATION = 10.0


def _ergodicity(kind: str, source: object, **settings: object) -> ErgodicityResult:
    """Return the ergodicity analysis of a source for one kind, run."""
    return ergodicity(source, kind=kind, **settings)


def _ornstein_uhlenbeck(rng: np.random.Generator) -> np.ndarray:
    """Return stationary Ornstein-Uhlenbeck paths of unit variance, one row per run, sampled at unit steps."""
    decay = np.exp(-1 / RELAXATION)
    kicks = rng.normal(0.0, np.sqrt(1 - decay**2), (RUNS, STEPS))
    paths = np.empty((RUNS, STEPS))
    paths[:, 0] = rng.normal(size=RUNS)
    for step in range(1, STEPS):
        paths[:, step] = decay * paths[:, step - 1] + kicks[:, step]
    return paths


def _frozen(rng: np.random.Generator) -> np.ndarray:
    """Return paths that each stay in one of two wells at -1 and 1 with small noise, one row per run."""
    wells = rng.choice([-1.0, 1.0], size=(RUNS, 1))
    return wells + 0.1 * rng.normal(size=(RUNS, STEPS // 10))


def test_ergodicity_breaking_ergodic_diffusion():
    """Test that Ornstein-Uhlenbeck paths much longer than their relaxation are ergodic with a 1/t metric."""
    paths = _ornstein_uhlenbeck(np.random.default_rng(0))
    assert _ergodicity('breaking', paths).ergodicity_breaking < 0.01
    metric = _ergodicity('metric', paths)
    assert metric.exponent == pytest.approx(-1.0, abs=0.1)
    assert metric.metric.shape == (STEPS,)
    assert _ergodicity('distance', paths).ergodicity_distance < 0.1


def test_ergodicity_breaking_frozen_disorder():
    """Test that paths frozen in one of two wells break ergodicity and keep a level metric."""
    paths = _frozen(np.random.default_rng(1))
    assert _ergodicity('breaking', paths).ergodicity_breaking > 0.5
    assert _ergodicity('metric', paths).exponent == pytest.approx(0.0, abs=0.1)
    assert _ergodicity('distance', paths).ergodicity_distance > 0.4
    frozen = _ergodicity('metric', np.repeat([[1.0], [-1.0]], 100, axis=1))
    assert (float(frozen.metric[-1]), round(frozen.exponent, 6)) == (1.0, 0.0)


def test_ergodicity_metric_times_the_columns_by_the_step():
    """Test that the runs of an ensemble, one row per run, are timed by the step between columns."""
    paths = _frozen(np.random.default_rng(2))[:, :50]
    np.testing.assert_allclose(_ergodicity('metric', paths, dt=0.5).times, np.arange(1, 51) * 0.5)


def test_ergodicity_breaking_rejects_a_single_run():
    """Test that the diagnostics ask for several runs."""
    with pytest.raises(AnalysisError, match='two runs'):
        _ergodicity('breaking', np.ones((1, 10)))
    with pytest.raises(AnalysisError, match='needs the edges'):
        ergodicity([0, 1, 0, 1], kind='ulam')
    with pytest.raises(AnalysisError, match='does not read samples'):
        ergodicity([0, 1, 0, 1], kind='breaking', samples=5)


def test_invariant_density_and_ulam_chain_of_the_logistic_map():
    """Test that a long run and the Ulam chain on 100 cells both match the arcsine density in total variation."""
    edges = np.linspace(0, 1, 101)
    exact = np.diff(2 / np.pi * np.arcsin(np.sqrt(edges)))
    values = np.empty(1_000_000)
    value = 0.3
    for index in range(values.size):
        value = 4 * value * (1 - value)
        values[index] = value
    density = _ergodicity('invariant-density', values, bins=edges).density
    chain = _ergodicity('ulam', lambda x: 4 * x * (1 - x), edges=edges, samples=1000)
    assert 0.5 * np.sum(np.abs(density * 0.01 - exact)) < 0.05
    assert 0.5 * np.sum(np.abs(chain.density * 0.01 - exact)) < 0.05
    np.testing.assert_allclose(chain.transition_matrix.sum(axis=1), 1)
    two_bins = _ergodicity('invariant-density', [0.1, 0.2, 0.6, 0.7], bins=2, value_range=(0.0, 1.0))
    assert two_bins.density.tolist() == [1.0, 1.0]
