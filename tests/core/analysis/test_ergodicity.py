"""Test the ergodicity-breaking parameter, the fluctuation metric, and the distance of time and ensemble laws."""

import numpy as np
import pytest

from skcomplex.core.analysis import Ensemble, ergodicity_breaking, ergodicity_distance, fluctuation_metric
from skcomplex.core.base import AnalysisError

RUNS = 200
STEPS = 20_000
RELAXATION = 10.0


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
    assert ergodicity_breaking(paths) < 0.01
    metric = fluctuation_metric(paths)
    assert metric.exponent == pytest.approx(-1.0, abs=0.1)
    assert metric.metric.shape == (STEPS,)
    assert ergodicity_distance(paths) < 0.1


def test_ergodicity_breaking_frozen_disorder():
    """Test that paths frozen in one of two wells break ergodicity and keep a level metric."""
    paths = _frozen(np.random.default_rng(1))
    assert ergodicity_breaking(paths) > 0.5
    assert fluctuation_metric(paths).exponent == pytest.approx(0.0, abs=0.1)
    assert ergodicity_distance(paths) > 0.4


def test_ergodicity_breaking_reads_an_ensemble():
    """Test that an ensemble of an experiment gives the same diagnostics as its values and uses its times."""
    paths = _frozen(np.random.default_rng(2))[:, :50]
    times = np.arange(50) * 0.5
    ensemble = Ensemble(times, paths, [{}] * RUNS, np.arange(RUNS))
    assert ergodicity_breaking(ensemble) == pytest.approx(ergodicity_breaking(paths))
    np.testing.assert_allclose(fluctuation_metric(ensemble).times, times)


def test_ergodicity_breaking_rejects_a_single_run():
    """Test that the diagnostics ask for several runs."""
    with pytest.raises(AnalysisError, match='two runs'):
        ergodicity_breaking(np.ones((1, 10)))
