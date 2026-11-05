"""Test the `_traps` module of `skcomplex.models.physics.soft_matter`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.physics.soft_matter import (
    biased_trap_walk,
)


def test_biased_trap_walk_subdiffusion():
    """Test that walkers among traps of exponent 1/2 travel as t^(1/2) with the renewal prefactor."""
    alpha = 0.5
    simulation = (
        Simulation(
            built := biased_trap_walk(100, n_sites=200, alpha=alpha),
            dt=10,
            random_state=0,
            observables=describe('biased_trap_walk').observables_of(built),
        )
        .prepare()
        .run(until=2000)
    )
    times, position = simulation.trajectory_.times, simulation.trajectory_.values('position')
    late = times >= 50
    slope = np.polyfit(np.log(times[late]), np.log(position[late]), 1)[0]
    assert slope == pytest.approx(alpha, abs=0.1)
    prediction = times[-1] ** alpha * math.sin(math.pi * alpha) / (math.pi * alpha)
    assert position[-1] == pytest.approx(prediction, rel=0.2)
