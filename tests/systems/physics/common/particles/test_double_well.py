"""Test the `_double_well` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import DoubleWellElement, DoubleWellInteraction


def test_double_well_relaxes_to_the_nearest_minimum_without_noise():
    """Test that particles without noise or forcing settle at the minimum `x = -1` or `x = 1` on their side."""
    particles = DoubleWellElement('particles', x=np.array([-0.5, 0.5, 2.0]), size=3)
    well = DoubleWellInteraction('well', target='particles', amplitude=0.0, D=0.0)
    system = System(elements=[particles], interactions=[well]).prepare()
    assert system.simulate(until=30.0, dt=30.0).get_values('particles.x')[-1] == pytest.approx([-1.0, 1.0, 1.0])


def test_double_well_noise_moves_particles_between_wells():
    """Test that strong noise moves particles that start in the left well into the right well."""
    particles = DoubleWellElement('particles', x=-1.0, size=200)
    well = DoubleWellInteraction('well', target='particles', amplitude=0.0, D=0.3)
    system = System(elements=[particles], interactions=[well], solver=EulerMaruyamaSolver(dt=0.01)).prepare()
    positions = system.simulate(until=50.0, dt=50.0, random_state=0).get_values('particles.x')[-1]
    assert 0.3 < (positions > 0).mean() < 0.7
