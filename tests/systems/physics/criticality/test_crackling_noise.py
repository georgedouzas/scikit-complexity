"""Test the `_crackling_noise` module of `skcomplex.systems.physics.criticality`."""

import numpy as np
import pytest

from skcomplex.core.analysis import FitLawResult, events, fit_law
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.physics.criticality import crackling_noise

DURATIONS = (20, 40, 80)
MAX_GENERATIONS = 5000


def _exponent(values: np.ndarray, minimum: float) -> FitLawResult:
    """Return the discrete power law fitted above a fixed lower cutoff."""
    return fit_law(values, law='power', minimum=minimum, discrete=True, bootstrap=0)


def _galton_watson(rng: np.random.Generator, trees: int) -> np.ndarray:
    """Return the activity of critical Galton-Watson trees with Poisson offspring of mean 1, one generation a step.

    The trees follow one another, each followed by a silent step, so that every tree is one avalanche.
    """
    alive, active = np.arange(trees), np.ones(trees, dtype=np.int64)
    durations = np.zeros(trees, dtype=np.int64)
    history: list[tuple[np.ndarray, np.ndarray]] = []
    while alive.size and len(history) < MAX_GENERATIONS:
        history.append((alive, active))
        durations[alive] += 1
        active = rng.poisson(active)
        alive, active = alive[active > 0], active[active > 0]
    starts = np.concatenate([[0], np.cumsum(durations + 1)[:-1]])
    activity = np.zeros(int(np.sum(durations + 1)))
    for generation, (ids, members) in enumerate(history):
        activity[starts[ids] + generation] = members
    return activity


def test_crackling_noise_finds_the_excursions():
    """Test that the avalanches of the crackling-noise analysis are the excursions of the activity."""
    activity = _galton_watson(np.random.default_rng(0), 1000)
    crackling = crackling_noise(activity, min_count=1)
    excursions = events(activity, kind='excursions')
    np.testing.assert_array_equal(crackling.sizes, excursions.sizes)
    np.testing.assert_array_equal(crackling.durations, excursions.durations)


def test_crackling_noise_branching_process_scaling():
    """Test that the mean size grows as the squared duration, agrees with the laws, and the shapes collapse."""
    found = events(_galton_watson(np.random.default_rng(1), 100_000), kind='excursions')
    crackling = crackling_noise(
        _galton_watson(np.random.default_rng(2), 1_000_000),
        durations=DURATIONS,
        min_duration=10,
        size_exponent=_exponent(found.sizes, 10).exponent,
        duration_exponent=_exponent(found.durations, 10).exponent,
    )
    assert crackling.exponent == pytest.approx(2, abs=0.1)
    assert crackling.exponent == pytest.approx(crackling.predicted, abs=0.15)
    assert crackling.shape_spread < 0.1
    assert sorted(crackling.mean_shapes) == list(DURATIONS)
    with pytest.raises(AnalysisError, match='Fewer than 2 durations'):
        crackling_noise([0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(AnalysisError, match='min_count=0'):
        crackling_noise([0, 1, 0, 1], min_count=0)
