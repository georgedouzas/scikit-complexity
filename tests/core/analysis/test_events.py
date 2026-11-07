"""Test the event analysis of avalanches, event counts in windows, laminar phases, excursions, and branching."""

import numpy as np
import pytest

from skcomplex.core.analysis import EventAnalysis, FitLawResult, distribution, fit_law
from skcomplex.core.errors import AnalysisError
from skcomplex.systems import create_simulation

MAX_GENERATIONS = 5000


def _events(kind: str, source: object, **settings: object) -> EventAnalysis:
    """Return the event analysis of a source for one kind, run."""
    return EventAnalysis(kind, **settings).prepare(source).run()


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


def _branching_activity(rng: np.random.Generator, ratio: float, drive: float, steps: int) -> np.ndarray:
    """Return the activity of a driven branching process, each active unit triggering Poisson descendants."""
    activity = np.zeros(steps)
    activity[0] = drive / (1 - ratio)
    for t in range(1, steps):
        activity[t] = rng.poisson(ratio * activity[t - 1] + drive)
    return activity


def test_avalanche_sizes():
    """Test that an avalanche's size is the sum of topplings over its consecutive nonzero steps."""
    events = _events('detect', [0, 0, 3, 5, 1, 0, 2, 0, 0, 0, 7, 7])
    assert events.sizes_.tolist() == [9.0, 2.0, 14.0]
    assert events.durations_.tolist() == [3.0, 1.0, 2.0]
    assert events.waiting_times_.tolist() == [1.0, 3.0]
    assert events.starts_.tolist() == [2.0, 6.0, 10.0]
    assert events.peaks_.tolist() == [5.0, 2.0, 7.0]


def test_events_of_a_trajectory_use_its_times():
    """Test that the events of a trajectory start at its times."""
    trajectory = create_simulation('poisson_process', n_processes=1).prepare().run(until=50.0).trajectory_
    events = _events('detect', trajectory, variables=['counters.count'])
    assert set(events.starts_) <= set(trajectory.times)


def test_event_counts_and_dispersion_of_event_times():
    """Test that the event times of a Poisson process give counts per window with a dispersion index near 1."""
    times = np.cumsum(np.random.default_rng(1).exponential(1.0, 100_000))
    counts = _events('counts', times, window=10.0).counts_
    assert counts.sum() <= times.size
    assert distribution(counts, kind='dispersion-index').index[0] == pytest.approx(1, abs=0.1)


def test_laminar_durations_of_on_off_intermittency():
    """Test that the laminar phases of on-off intermittency have a tail of exponent 3/2."""
    rng = np.random.default_rng(1)
    rng.choice([-1, 1], 2_000_000)
    size = 2_000_000
    values = np.empty(size)
    values[0] = 0.5
    factors = np.e * rng.random(size)
    for t in range(1, size):
        values[t] = max(factors[t] * values[t - 1] * (1 - values[t - 1]), 1e-12)
    durations = _events('laminar-durations', values, threshold=1e-3).durations_
    fit = fit_law(durations, minimum=10, discrete=True, bootstrap=0)
    assert fit.exponent == pytest.approx(1.5, abs=0.1)
    assert _events('laminar-durations', [1, 0, 0, 1, 0, 1], threshold=0.5).durations_.tolist() == [2.0, 1.0]


def test_unknown_kind_ignored_setting_and_missing_window():
    """Test that an unknown kind, an ignored setting, a missing window, and several series raise errors."""
    with pytest.raises(AnalysisError, match='detect, counts, laminar-durations'):
        EventAnalysis('unknown').prepare()
    with pytest.raises(AnalysisError, match='does not read window'):
        EventAnalysis('detect', window=1.0).prepare()
    with pytest.raises(AnalysisError, match='positive window'):
        EventAnalysis('counts').prepare()
    with pytest.raises(AnalysisError, match='Pick one'):
        _events('detect', np.zeros((5, 2)))


def test_excursions_of_critical_branching():
    """Test that critical Galton-Watson excursions have size exponent 3/2 and duration exponent 2."""
    found = _events('excursions', _galton_watson(np.random.default_rng(0), 100_000))
    assert _exponent(found.sizes_, 10).exponent == pytest.approx(1.5, abs=0.05)
    assert _exponent(found.durations_, 10).exponent == pytest.approx(2, abs=0.1)


def test_excursions_of_an_activity_series():
    """Test that runs of activity above a threshold are excursions with their profiles."""
    found = _events('excursions', np.asarray([0, 2, 3, 0, 1, 0]), threshold=0.5)
    assert found.sizes_.tolist() == [4.0, 0.5]
    assert [shape.size for shape in found.shapes_] == [2, 1]


def test_branching_ratio_subsampled_network():
    """Test that a branching network with ratio 0.98 gives 0.98 from all units and from 5 percent of them."""
    rng = np.random.default_rng(2)
    activity = _branching_activity(rng, 0.98, 20.0, 100_000)
    assert _events('branching', activity).branching_ratio_ == pytest.approx(0.98, abs=0.01)
    subsampled = rng.binomial(activity.astype(int), 0.05)
    assert _events('branching', subsampled).branching_ratio_ == pytest.approx(0.98, abs=0.01)


def test_branching_ratio_neuronal_and_epidemic_activity():
    """Test a critical stochastic neural network and a subcritical epidemic with reproduction number 0.9."""
    rng = np.random.default_rng(3)
    neurons, targets = 10_000, 10
    active = 100
    spikes = []
    for _ in range(20_000):
        hits = rng.binomial(active * targets, 1 / targets)
        active = np.unique(rng.integers(0, neurons, hits)).size + rng.binomial(neurons, 1e-4)
        spikes.append(active)
    assert _events('branching', np.asarray(spikes, dtype=float)).branching_ratio_ == pytest.approx(1, abs=0.02)
    cases = _branching_activity(rng, 0.9, 5.0, 50_000)
    assert _events('branching', cases).branching_ratio_ == pytest.approx(0.9, abs=0.03)


def test_excursions_and_branching_reject_sources():
    """Test that the branching ratio takes no records, excursions take one series, and branching reads its steps."""
    with pytest.raises(AnalysisError, match='does not read max_step'):
        EventAnalysis('excursions', max_step=10).prepare()
    with pytest.raises(AnalysisError, match='reads one series'):
        _events('excursions', np.ones((5, 2)))
    with pytest.raises(AnalysisError, match='Use positive values'):
        EventAnalysis('branching', max_step=0).prepare()
