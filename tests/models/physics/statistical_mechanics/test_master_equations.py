"""Test the `_master_equations` module of `skcomplex.models.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.statistical_mechanics import (
    discrete_master_equation,
    landau_relaxation,
    two_phase_zone_exchange,
)


def test_discrete_master_equation_relaxes_to_boltzmann():
    """Test that the master equation conserves probability and reaches `exp(-beta U) / Z`."""
    trajectory = Simulation(discrete_master_equation(), random_state=0).check().run(until=3000).trajectory_
    probability = trajectory.lattice('states.P')
    weights = np.exp(-trajectory.lattice('states.U')[-1])
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.allclose(probability[-1], weights / weights.sum(), atol=1e-6)


def test_two_phase_zone_exchange_is_binomial():
    """Test that the number of particles in a zone settles on the binomial mean and variance."""
    trajectory = Simulation(two_phase_zone_exchange(300), dt=0.5, random_state=0).check().run(until=12).trajectory_
    numbers = trajectory.values('zones.N')[trajectory.times >= 8]
    assert numbers.mean() == pytest.approx(10.0, rel=0.05)
    assert numbers.var() == pytest.approx(9.0, rel=0.15)


@pytest.mark.parametrize(('temperature', 'order'), [(0.5, np.sqrt(0.5)), (2.0, 0.0)])
def test_landau_relaxation_reaches_the_mean_field_order(temperature, order):
    """Test that the order parameter relaxes to `sqrt(a (Tc - T) / b)` below `Tc` and to 0 above it."""
    trajectory = Simulation(landau_relaxation(T=temperature), dt=0.1).check().run(until=50).trajectory_
    assert trajectory.values('magnet.phi')[-1] == pytest.approx(order, abs=1e-6)
