"""Test the `_crackling_noise` module of `skcomplex.models.physics.criticality`."""

import numpy as np
import pytest

from skcomplex.analysis import EventAnalysis, FitAnalysis
from skcomplex.errors import AnalysisError
from skcomplex.models.physics.criticality import CracklingNoiseAnalysis

DURATIONS = (20, 40, 80)
MAX_GENERATIONS = 5000


def _exponent(values: np.ndarray, minimum: float) -> FitAnalysis:
    """Return the discrete power law fitted above a fixed lower cutoff."""
    return FitAnalysis('power', minimum=minimum, discrete=True, bootstrap=0).prepare(values).run()


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
    crackling = CracklingNoiseAnalysis(min_count=1).prepare(activity).run()
    excursions = EventAnalysis('excursions').prepare(activity).run()
    np.testing.assert_array_equal(crackling.sizes_, excursions.sizes_)
    np.testing.assert_array_equal(crackling.durations_, excursions.durations_)


def test_crackling_noise_branching_process_scaling():
    """Test that the mean size grows as the squared duration, agrees with the laws, and the shapes collapse."""
    found = EventAnalysis('excursions').prepare(_galton_watson(np.random.default_rng(1), 100_000)).run()
    crackling = CracklingNoiseAnalysis(
        durations=DURATIONS,
        min_duration=10,
        size_exponent=_exponent(found.sizes_, 10).exponent_,
        duration_exponent=_exponent(found.durations_, 10).exponent_,
    )
    crackling.prepare(_galton_watson(np.random.default_rng(2), 1_000_000)).run()
    assert crackling.exponent_ == pytest.approx(2, abs=0.1)
    assert crackling.exponent_ == pytest.approx(crackling.predicted_, abs=0.15)
    assert crackling.shape_spread_ < 0.1
    assert sorted(crackling.mean_shapes_) == list(DURATIONS)
    with pytest.raises(AnalysisError, match='Fewer than 2 durations'):
        CracklingNoiseAnalysis().prepare([0, 1, 0, 1, 0, 1, 0]).run()
    with pytest.raises(AnalysisError, match='min_count=0'):
        CracklingNoiseAnalysis(min_count=0).prepare()
