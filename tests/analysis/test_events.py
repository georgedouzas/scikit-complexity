"""Test the event analysis of avalanches, event counts in windows, and laminar phases."""

import numpy as np
import pytest

from skcomplex.analysis import EventAnalysis, FluctuationAnalysis, fit_power_law
from skcomplex.core.errors import AnalysisError
from skcomplex.models import run


def _events(kind: str, source: object, **settings: object) -> EventAnalysis:
    """Return the event analysis of a source for one kind, run."""
    return EventAnalysis(kind, **settings).check().prepare(source).run()


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
    trajectory = run('poisson_process', n_processes=1, settings={'until': 50.0, 'random_state': 0}).trajectory_
    events = _events('detect', trajectory, variables=['counters.count'])
    assert set(events.starts_) <= set(trajectory.times)


def test_event_counts_and_fano_of_event_times():
    """Test that the event times of a Poisson process give counts per window with a Fano factor near 1."""
    times = np.cumsum(np.random.default_rng(1).exponential(1.0, 100_000))
    counts = _events('counts', times, window=10.0).counts_
    assert counts.sum() <= times.size
    assert FluctuationAnalysis('fano').check().prepare(counts).run().factor_[0] == pytest.approx(1, abs=0.1)


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
    assert fit_power_law(durations, xmin=10, discrete=True, bootstrap=0).alpha == pytest.approx(1.5, abs=0.1)
    assert _events('laminar-durations', [1, 0, 0, 1, 0, 1], threshold=0.5).durations_.tolist() == [2.0, 1.0]


def test_unknown_kind_ignored_setting_and_missing_window():
    """Test that an unknown kind, an ignored setting, a missing window, and several series raise errors."""
    with pytest.raises(AnalysisError, match='detect, counts, laminar-durations'):
        EventAnalysis('unknown').check()
    with pytest.raises(AnalysisError, match='does not read window'):
        EventAnalysis('detect', window=1.0).check()
    with pytest.raises(AnalysisError, match='positive window'):
        EventAnalysis('counts').check()
    with pytest.raises(AnalysisError, match='Pick one'):
        _events('detect', np.zeros((5, 2)))
