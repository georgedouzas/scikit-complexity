"""Test the `_master_equations` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import (
    DiscreteMasterEquationSystem,
    LandauRelaxationSystem,
    TwoPhaseZoneExchangeSystem,
)


def test_discrete_master_equation_relaxes_to_boltzmann():
    """Test that the master equation conserves probability and reaches `exp(-beta U) / Z`."""
    trajectory = simulate(DiscreteMasterEquationSystem(), until=3000, random_state=0)
    probability = trajectory.lattice('states.P')
    weights = np.exp(-trajectory.lattice('states.U')[-1])
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.allclose(probability[-1], weights / weights.sum(), atol=1e-6)


def test_two_phase_zone_exchange_is_binomial():
    """Test that the number of particles in a zone settles on the binomial mean and variance."""
    trajectory = simulate(TwoPhaseZoneExchangeSystem(300), until=12, random_state=0)
    numbers = trajectory.values('zones.N')[trajectory.times >= 8]
    assert numbers.mean() == pytest.approx(10.0, rel=0.05)
    assert numbers.var() == pytest.approx(9.0, rel=0.15)


@pytest.mark.parametrize(('temperature', 'order'), [(0.5, np.sqrt(0.5)), (2.0, 0.0)])
def test_landau_relaxation_reaches_the_mean_field_order(temperature, order):
    """Test that the order parameter relaxes to `sqrt(a (Tc - T) / b)` below `Tc` and to 0 above it."""
    trajectory = simulate(LandauRelaxationSystem(T=temperature), until=50)
    assert trajectory.values('magnet.phi')[-1] == pytest.approx(order, abs=1e-6)
