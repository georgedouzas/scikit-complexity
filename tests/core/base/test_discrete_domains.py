"""Test boolean, categorical, and angle domains of states."""

import numpy as np
import pytest

from skcomplex.core.base import Domain, ValidationError, parameter, state, wrap_angle
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Rate, Update
from skcomplex.core.runtime import Simulation
from skcomplex.core.system import System

HEALTH = Domain.categorical('susceptible', 'infected', 'recovered')

Person = Element.define('Person', states={'health': state('susceptible', domain=HEALTH)})
Progression = Interaction.define(
    'Progression',
    roles=['target'],
    parameters={'step': parameter(1.0)},
    laws=lambda s: [Update(s.target.health, s.target.health + s.step)],
)
Rotor = Element.define('Rotor', states={'theta': state(0.0, domain=Domain.ANGLE)}, parameters={'omega': 1.0})
Spin = Interaction.define('Spin', roles=['target'], laws=lambda s: [Rate(s.target.theta, s.target.omega)])
Switch = Element.define('Switch', states={'on': state(False, domain=Domain.BOOLEAN)})


def test_categorical_values_round_trip_as_labels():
    """Test that categorical states take labels as values and return them as labels."""
    system = System(components=[Person('ann', health='infected')], interactions=[Progression('p', target='ann')])
    trajectory = Simulation(system).run(until=1).trajectory_
    assert trajectory['ann.health'].tolist() == ['infected', 'recovered']


def test_categorical_rejects_unknown_label():
    """Test that an initial value outside the alphabet is rejected."""
    with pytest.raises(ValidationError, match='categorical'):
        Simulation(System(components=[Person('ann', health='dead')])).run(until=1)


def test_checked_mode_catches_values_outside_alphabet():
    """Test that checked mode names the state when a law leaves the alphabet."""
    system = System(components=[Person('ann', health='recovered')], interactions=[Progression('p', target='ann')])
    with pytest.raises(ValidationError, match=r'ann\.health .*categorical'):
        Simulation(system, checked=True).run(until=1)
    unchecked = Simulation(system).run(until=1).trajectory_
    assert unchecked.states[-1, 0] == 3.0


def test_angle_wraps_into_one_turn():
    """Test that an angle growing past π lies in (-π, π] after every frame."""
    system = System(components=[Rotor('r', theta=6.0, omega=1.0)], interactions=[Spin('s', target='r')])
    values = Simulation(system, dt=0.5).run(until=3).trajectory_['r.theta']
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(9.0 - 2 * np.pi, abs=1e-6)


def test_angle_swings_around_zero_without_jumping():
    """Test that an angle decreasing through zero stays near zero instead of jumping a full turn."""
    system = System(components=[Rotor('r', theta=0.1, omega=-1.0)], interactions=[Spin('s', target='r')])
    values = Simulation(system, dt=0.1).run(until=1).trajectory_['r.theta']
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(-0.9, abs=1e-6)


def test_angle_encode_wraps_into_half_open_turn():
    """Test that encoding an angle wraps it into (-π, π], keeping π and moving -π to π."""
    encoded = Domain.ANGLE.encode([3 * np.pi / 2, -np.pi, np.pi])
    np.testing.assert_allclose(encoded, [-np.pi / 2, np.pi, np.pi])


def test_wrap_angle_maps_into_half_open_turn():
    """Test that wrap_angle maps any real value into (-π, π]."""
    values = wrap_angle(np.array([0.1, -0.1, 7.0, -7.0, np.pi, -np.pi]))
    np.testing.assert_allclose(values, [0.1, -0.1, 7.0 - 2 * np.pi, -7.0 + 2 * np.pi, np.pi, np.pi])


def test_boolean_values_return_as_booleans():
    """Test that boolean states return booleans and reject other numbers."""
    trajectory = Simulation(System(components=[Switch('s', on=True)]), dt=1).run(until=1).trajectory_
    assert trajectory['s.on'].tolist() == [True, True]
    with pytest.raises(ValidationError, match='boolean'):
        Simulation(System(components=[Switch('s', on=0.5)])).run(until=1)


def test_wrap_angle_keeps_angles_inside_exactly():
    """Test that wrap_angle returns angles already in (-π, π] bit for bit, so boundary laws see no change."""
    values = np.linspace(-3.14, np.pi, 1001)
    assert np.array_equal(wrap_angle(values), values)
