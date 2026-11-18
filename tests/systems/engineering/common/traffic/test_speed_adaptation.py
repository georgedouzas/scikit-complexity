"""Test the `_speed_adaptation` module of `skcomplex.systems.engineering.common.traffic`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.engineering.common.traffic import CarElement, SpeedAdaptationInteraction


def test_speed_adapts_towards_a_moving_desired_speed():
    """Test that adapting towards the desired speed `t` follows `t - time (1 - exp(-t / time))`."""
    system = System(
        elements=[Element('car', values={'x': 0.0})],
        interactions=[SpeedAdaptationInteraction('adaptation', target='car', goal=lambda s: s.t, time=2.0)],
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=3.0, dt=0.5)
    expected = trajectory.times - 2.0 * (1 - np.exp(-trajectory.times / 2.0))
    np.testing.assert_allclose(trajectory.get_values('car.x'), expected, rtol=1e-7, atol=1e-9)


def test_speed_of_a_car_adapts_as_the_derivative_of_its_position():
    """Test that the speed `x'` of a car relaxes to the goal and its position advances at that speed."""
    adaptation = SpeedAdaptationInteraction('adaptation', target='car', state="x'", goal=1.0, time=0.5)
    system = System(elements=[CarElement('car')], interactions=[adaptation]).prepare()
    trajectory = system.simulate(until=10.0, dt=1.0)
    assert system.states_ == ('car.x', "car.x'")
    assert trajectory.get_values("car.x'")[-1] == pytest.approx(1.0, abs=1e-6)
    assert trajectory.get_values('car.x')[-1] == pytest.approx(10.0 - 0.5, abs=1e-4)


def test_prepare_rejects_a_time_that_is_not_a_function():
    """Test that `prepare` rejects an adaptation time that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='time'):
        SpeedAdaptationInteraction('adaptation', target='car', time='slow').prepare()
