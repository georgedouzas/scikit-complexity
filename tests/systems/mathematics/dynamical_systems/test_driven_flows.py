"""Test the `_driven_flows` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems import describe
from skcomplex.systems.mathematics.dynamical_systems import (
    OUDrivenRosslerSystem,
)


def test_ou_driven_rossler_control_has_the_stationary_variance():
    """Test that the control is an Ornstein-Uhlenbeck process of variance rho^2 / (2 tau) and the flow stays bounded."""
    solver = describe('ou_driven_rossler').defaults['solver']
    system = OUDrivenRosslerSystem(tau=1.0, rho=0.2)
    trajectory = simulate(system, until=2000.0, dt=0.1, solver=solver, random_state=0)
    assert trajectory.values('rossler.eta').var() == pytest.approx(0.2**2 / 2, rel=0.1)
    assert np.abs(trajectory.values('rossler.x')).max() < 20
