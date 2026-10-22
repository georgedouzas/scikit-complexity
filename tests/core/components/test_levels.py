"""Test the conversions between the members of a group and a distribution over activities."""

import numpy as np
import pytest

from skcomplex.core.base import Domain, ValidationError, state
from skcomplex.core.components import Element, Group, distribution_from, members_from
from skcomplex.core.laws import moment
from skcomplex.core.runtime import Simulation
from skcomplex.core.space import ContinuousTime, LatticeSpace
from skcomplex.core.system import Observable, System

CLASSES = np.arange(9.0)


class Trader(Element):
    """Agent with a wealth class."""

    wealth: float = state(default=0.0)


class Density(Element):
    """Density of particles at one activity value."""

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)


def density(s):
    """Return the density of a distribution."""
    return moment(s.f, 0, CLASSES)


def mean_class(s):
    """Return the mean class of a distribution."""
    return moment(s.f, 1, CLASSES, normalized=True)


def _traders():
    """Return 1000 traders with random wealth classes."""
    wealth = np.random.default_rng(0).integers(0, 9, size=1000).astype(float)
    return Trader.group('traders', 1000, wealth=wealth)


def test_agents_bin_into_a_distribution():
    """Test that 1000 agents' wealth bins into a distribution of density 1 whose mean is the agents' mean class."""
    traders = _traders()
    distribution = distribution_from(traders, 'wealth', CLASSES, element=Density, label='wealth')
    assert distribution.on == 'sites'
    assert distribution.size == 9
    system = System(components=[distribution], space=LatticeSpace(9, boundary='open'), time=ContinuousTime())
    observables = [Observable('density', density, group='wealth'), Observable('mean', mean_class, group='wealth')]
    trajectory = Simulation(system, observables=observables, dt=1.0).run(until=0).trajectory_
    assert trajectory['density'][0] == pytest.approx(1.0)
    assert trajectory['mean'][0] == pytest.approx(traders.values['wealth'].mean())


def test_values_bin_to_the_nearest_activity():
    """Test that values between activities count at the nearest one, and values outside at the closest end."""
    distribution = distribution_from([-1.0, 0.4, 0.6, 1.5, 9.0], 'wealth', [0.0, 1.0, 2.0], element=Density)
    np.testing.assert_allclose(distribution.values['f'], [0.4, 0.4, 0.2])
    assert distribution.label == 'members_distribution'


def test_distribution_samples_members():
    """Test that members drawn from a distribution take its activities with its frequencies."""
    profile = np.array([0.1, 0.0, 0.2, 0.0, 0.4, 0.0, 0.2, 0.0, 0.1])
    group = Group('wealth', element=Density, size=9, on='sites', values={'f': profile})
    members = members_from(group, Trader, 20_000, activities=CLASSES, random_state=1)
    assert members.variable
    assert members.label == 'wealth_members'
    assert set(np.unique(members.values['wealth'])) <= set(CLASSES[profile > 0])
    assert members.values['wealth'].mean() == pytest.approx(profile @ CLASSES, rel=0.02)
    again = distribution_from(members, 'wealth', CLASSES, element=Density)
    np.testing.assert_allclose(again.values['f'], profile, atol=0.01)


@pytest.mark.parametrize(
    ('call', 'message'),
    [
        (lambda: distribution_from([1.0], 'wealth', [1.0, 0.0], element=Density), 'increasing'),
        (lambda: distribution_from([], 'wealth', [0.0, 1.0], element=Density), 'no member'),
        (lambda: distribution_from(_traders(), 'income', CLASSES, element=Density), 'Name the state'),
        (lambda: members_from([0.0, 0.0], Trader, 3, activities=[0.0, 1.0]), 'not all zero'),
        (lambda: members_from([1.0], Trader, 3, activities=[0.0, 1.0]), 'one nonnegative density'),
    ],
)
def test_level_conversions_reject_bad_input(call, message):
    """Test that the conversions reject bad activities, empty groups, unknown states, and bad densities."""
    with pytest.raises(ValidationError, match=message):
        call()
