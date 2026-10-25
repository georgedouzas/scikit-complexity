"""Test the `_domains` module of `skcomplex.core.base`."""

import numpy as np
import pytest

from skcomplex.core.base import (
    NumericDomain,
    ValidationError,
    state,
    wrap_angle,
)
from skcomplex.core.components import ElementComponent
from skcomplex.core.interactions import BaseInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation

from .conftest import PersonComponent, ProgressionInteraction

RotorComponent = ElementComponent.define(
    'RotorComponent',
    states={'theta': state(0.0, domain=NumericDomain.ANGLE)},
    parameters={'omega': 1.0},
)


SpinInteraction = BaseInteraction.define(
    'SpinInteraction',
    roles=['target'],
    laws=lambda s: [RateLaw(s.target.theta, s.target.omega)],
)


SwitchComponent = ElementComponent.define('SwitchComponent', states={'on': state(False, domain=NumericDomain.BOOLEAN)})


@pytest.mark.parametrize(
    ('domain', 'value', 'expected'),
    [
        (NumericDomain.REAL, -1.5, True),
        (NumericDomain.POSITIVE, 0.0, False),
        (NumericDomain.NONNEGATIVE, 0.0, True),
        (NumericDomain.INTEGER, 2.5, False),
        (NumericDomain.NONNEGATIVE_INTEGER, [0, 3], True),
        (NumericDomain.REAL, float('nan'), False),
        (NumericDomain.REAL, 'm', False),
    ],
)
def test_domain_contains_checks_values(domain, value, expected):
    """Test that domains accept and reject values."""
    assert domain.contains(value) is expected


def test_domain_maps_to_sage_domain():
    """Test that domains give the Sage domains of their symbols, which have no nonnegative domains."""
    assert NumericDomain.NONNEGATIVE_INTEGER.sage_domain == 'integer'


def test_categorical_values_round_trip_as_labels():
    """Test that categorical states take labels as values and return them as labels."""
    system = System(
        components=[PersonComponent('ann', health='infected')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    trajectory = Simulation(system).run(until=1).trajectory_
    assert trajectory['ann.health'].tolist() == ['infected', 'recovered']


def test_categorical_rejects_unknown_label():
    """Test that an initial value outside the alphabet is rejected."""
    with pytest.raises(ValidationError, match='categorical'):
        Simulation(System(components=[PersonComponent('ann', health='dead')])).run(until=1)


def test_angle_wraps_into_one_turn():
    """Test that an angle growing past π lies in (-π, π] after every frame."""
    system = System(
        components=[RotorComponent('r', theta=6.0, omega=1.0)],
        interactions=[SpinInteraction('s', target='r')],
    )
    values = Simulation(system, dt=0.5).run(until=3).trajectory_['r.theta']
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(9.0 - 2 * np.pi, abs=1e-6)


def test_angle_swings_around_zero_without_jumping():
    """Test that an angle decreasing through zero stays near zero instead of jumping a full turn."""
    system = System(
        components=[RotorComponent('r', theta=0.1, omega=-1.0)],
        interactions=[SpinInteraction('s', target='r')],
    )
    values = Simulation(system, dt=0.1).run(until=1).trajectory_['r.theta']
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(-0.9, abs=1e-6)


def test_angle_encode_wraps_into_half_open_turn():
    """Test that encoding an angle wraps it into (-π, π], keeping π and moving -π to π."""
    encoded = NumericDomain.ANGLE.encode([3 * np.pi / 2, -np.pi, np.pi])
    np.testing.assert_allclose(encoded, [-np.pi / 2, np.pi, np.pi])


def test_wrap_angle_maps_into_half_open_turn():
    """Test that wrap_angle maps any real value into (-π, π]."""
    values = wrap_angle(np.array([0.1, -0.1, 7.0, -7.0, np.pi, -np.pi]))
    np.testing.assert_allclose(values, [0.1, -0.1, 7.0 - 2 * np.pi, -7.0 + 2 * np.pi, np.pi, np.pi])


def test_boolean_values_return_as_booleans():
    """Test that boolean states return booleans and reject other numbers."""
    trajectory = Simulation(System(components=[SwitchComponent('s', on=True)]), dt=1).run(until=1).trajectory_
    assert trajectory['s.on'].tolist() == [True, True]
    with pytest.raises(ValidationError, match='boolean'):
        Simulation(System(components=[SwitchComponent('s', on=0.5)])).run(until=1)


def test_wrap_angle_keeps_angles_inside_exactly():
    """Test that wrap_angle returns angles already in (-π, π] bit for bit, so boundary laws see no change."""
    values = np.linspace(-3.14, np.pi, 1001)
    assert np.array_equal(wrap_angle(values), values)
