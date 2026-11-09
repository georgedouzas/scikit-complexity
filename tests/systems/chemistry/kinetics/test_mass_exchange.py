"""Test the `_mass_exchange` module of `skcomplex.systems.chemistry.kinetics`."""

import numpy as np
import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.chemistry.kinetics import MassExchangeInteraction

from ....conftest import NodeElement


def test_state_follows_a_moving_goal():
    """Test that a mass exchange towards the goal `t` follows `t - time (1 - exp(-t / time))`."""
    system = CompositeSystem(
        elements=[NodeElement('cell', x=0.0)],
        interactions=[MassExchangeInteraction('relaxation', target='cell', goal=lambda s: s.t, time=2.0)],
    )
    trajectory = simulate(system, until=3.0, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=0.5)
    expected = trajectory.times - 2.0 * (1 - np.exp(-trajectory.times / 2.0))
    np.testing.assert_allclose(trajectory.values('cell.x'), expected, rtol=1e-7, atol=1e-9)


def test_check_rejects_a_goal_that_is_not_a_function():
    """Test that `check` rejects a goal that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='goal'):
        MassExchangeInteraction('relaxation', target='cell', goal='zero').prepare()
