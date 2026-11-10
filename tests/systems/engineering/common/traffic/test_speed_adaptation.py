"""Test the `_speed_adaptation` module of `skcomplex.systems.engineering.common.traffic`."""

import numpy as np
import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.engineering.common.traffic import SpeedAdaptationInteraction

from .....conftest import NodeElement


def test_speed_adapts_towards_a_moving_desired_speed():
    """Test that adapting towards the desired speed `t` follows `t - time (1 - exp(-t / time))`."""
    system = CompositeSystem(
        elements=[NodeElement('car', x=0.0)],
        interactions=[SpeedAdaptationInteraction('adaptation', target='car', goal=lambda s: s.t, time=2.0)],
    )
    trajectory = simulate(system, until=3.0, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=0.5)
    expected = trajectory.times - 2.0 * (1 - np.exp(-trajectory.times / 2.0))
    np.testing.assert_allclose(trajectory.get_values('car.x'), expected, rtol=1e-7, atol=1e-9)


def test_check_rejects_a_time_that_is_not_a_function():
    """Test that `check` rejects an adaptation time that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='time'):
        SpeedAdaptationInteraction('adaptation', target='car', time='slow').prepare()
