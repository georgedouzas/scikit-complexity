"""Test the `_driven_flows` module of `skcomplex.models.mathematics.dynamical_systems`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.mathematics.dynamical_systems import (
    ou_driven_rossler,
)


def test_ou_driven_rossler_control_has_the_stationary_variance():
    """Test that the control is an Ornstein-Uhlenbeck process of variance rho^2 / (2 tau) and the flow stays bounded."""
    solver = describe('ou_driven_rossler').defaults['solver']
    system = ou_driven_rossler(tau=1.0, rho=0.2)
    trajectory = Simulation(system, dt=0.1, solver=solver, random_state=0).run(until=2000.0).trajectory_
    assert trajectory['rossler.eta'].var() == pytest.approx(0.2**2 / 2, rel=0.1)
    assert np.abs(trajectory['rossler.x']).max() < 20
