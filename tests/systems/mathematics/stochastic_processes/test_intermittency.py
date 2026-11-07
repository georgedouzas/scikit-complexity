"""Test the `_intermittency` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.systems.mathematics.stochastic_processes import OnOffIntermittencySystem


def _laminar_durations(values, threshold, dt):
    """Return the durations of the complete phases below a threshold of every column of a series."""
    durations = []
    for column in values.T:
        edges = np.diff(np.r_[0, (column < threshold).astype(int), 0])
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        durations.extend(((ends - starts) * dt)[1:-1])
    return np.array(durations)


def test_on_off_intermittency_laminar_phases():
    """Test that the laminar phases at the bifurcation have durations distributed as tau^(-3/2)."""
    system = OnOffIntermittencySystem(50, epsilon=0.0).prepare().system_
    simulation = (
        Simulation(system, dt=0.1, solver=EulerMaruyamaSolver(dt=0.02), random_state=0).prepare().run(until=500)
    )
    values = simulation.trajectory_.values('variables.x')
    durations = _laminar_durations(values, 0.05, 0.1)
    edges = np.logspace(0, 2, 9)
    counts, _ = np.histogram(durations, edges)
    slope = np.polyfit(np.log(np.sqrt(edges[1:] * edges[:-1])), np.log(counts / np.diff(edges)), 1)[0]
    assert slope == pytest.approx(-1.5, abs=0.15)
    assert np.mean(values < 0.05) > 0.5
