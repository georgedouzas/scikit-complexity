"""Test boolean, categorical, and angle domains of states."""

import numpy as np
import pytest

from skcomplex.core.base import Domain, ValidationError, parameter, state
from skcomplex.core.model import Element, Interaction, Rate, System, Update
from skcomplex.core.runtime import Simulation

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
    system = System(elements=[Person('ann', health='infected')], interactions=[Progression('p', target='ann')])
    trajectory = Simulation(system).run(until=1).trajectory_
    assert trajectory['ann.health'].tolist() == ['infected', 'recovered']


def test_categorical_rejects_unknown_label():
    """Test that an initial value outside the alphabet is rejected."""
    with pytest.raises(ValidationError, match='categorical'):
        Simulation(System(elements=[Person('ann', health='dead')])).run(until=1)


def test_checked_mode_catches_values_outside_alphabet():
    """Test that checked mode names the state when a law leaves the alphabet."""
    system = System(elements=[Person('ann', health='recovered')], interactions=[Progression('p', target='ann')])
    with pytest.raises(ValidationError, match=r'ann\.health .*categorical'):
        Simulation(system, checked=True).run(until=1)
    unchecked = Simulation(system).run(until=1).trajectory_
    assert unchecked.states[-1, 0] == 3.0


def test_angle_wraps_into_one_turn():
    """Test that an angle growing past 2π lies in [0, 2π) after every frame."""
    system = System(elements=[Rotor('r', theta=6.0, omega=1.0)], interactions=[Spin('s', target='r')])
    values = Simulation(system, dt=0.5).run(until=3).trajectory_['r.theta']
    assert np.all((values >= 0) & (values < 2 * np.pi))
    assert values[-1] == pytest.approx(np.mod(9.0, 2 * np.pi), abs=1e-6)


def test_boolean_values_return_as_booleans():
    """Test that boolean states return booleans and reject other numbers."""
    trajectory = Simulation(System(elements=[Switch('s', on=True)]), dt=1).run(until=1).trajectory_
    assert trajectory['s.on'].tolist() == [True, True]
    with pytest.raises(ValidationError, match='boolean'):
        Simulation(System(elements=[Switch('s', on=0.5)])).run(until=1)
