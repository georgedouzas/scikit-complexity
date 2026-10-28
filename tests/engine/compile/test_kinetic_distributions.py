"""Test the conversions between the members of a group and a distribution over activities."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws.utils import moment
from skcomplex.core.spaces import ContinuousTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import distribution_from, members_from
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import ObservableSpec, StateSpec, VisualSpec

CLASSES = np.arange(9.0)


class TraderElement(Element):
    """Agent with a wealth class."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth


class DensityElement(Element):
    """Density of particles at one activity value."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        f: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.f = f


def density(s):
    """Return the density of a distribution."""
    return moment(s.f, 0, CLASSES)


def mean_class(s):
    """Return the mean class of a distribution."""
    return moment(s.f, 1, CLASSES, normalized=True)


def _traders():
    """Return 1000 traders with random wealth classes."""
    wealth = np.random.default_rng(0).integers(0, 9, size=1000).astype(float)
    return GroupElement('traders', element=TraderElement, size=1000, values={'wealth': wealth})


def test_agents_bin_into_a_distribution():
    """Test that 1000 agents' wealth bins into a distribution of density 1 whose mean is the agents' mean class."""
    traders = _traders()
    distribution = distribution_from(traders, 'wealth', CLASSES, element=DensityElement, label='wealth')
    assert distribution.on == 'sites'
    assert distribution.size == 9
    system = System(elements=[distribution], space=LatticeSpace(9, boundary='fixed'), time=ContinuousTime())
    observables = [
        ObservableSpec('density', density, group='wealth'),
        ObservableSpec('mean', mean_class, group='wealth'),
    ]
    trajectory = Simulation(system, observables=observables, dt=1.0).check().run(until=0).trajectory_
    assert trajectory.values('density')[0] == pytest.approx(1.0)
    assert trajectory.values('mean')[0] == pytest.approx(traders.values['wealth'].mean())


def test_values_bin_to_the_nearest_activity():
    """Test that values between activities count at the nearest one, and values outside at the closest end."""
    distribution = distribution_from([-1.0, 0.4, 0.6, 1.5, 9.0], 'wealth', [0.0, 1.0, 2.0], element=DensityElement)
    np.testing.assert_allclose(distribution.values['f'], [0.4, 0.4, 0.2])
    assert distribution.label == 'members_distribution'


def test_distribution_samples_members():
    """Test that members drawn from a distribution take its activities with its frequencies."""
    profile = np.array([0.1, 0.0, 0.2, 0.0, 0.4, 0.0, 0.2, 0.0, 0.1])
    group = GroupElement('wealth', element=DensityElement, size=9, on='sites', values={'f': profile})
    members = members_from(group, TraderElement, 20_000, activities=CLASSES, random_state=1)
    assert members.variable
    assert members.label == 'wealth_members'
    assert set(np.unique(members.values['wealth'])) <= set(CLASSES[profile > 0])
    assert members.values['wealth'].mean() == pytest.approx(profile @ CLASSES, rel=0.02)
    again = distribution_from(members, 'wealth', CLASSES, element=DensityElement)
    np.testing.assert_allclose(again.values['f'], profile, atol=0.01)


@pytest.mark.parametrize(
    ('call', 'message'),
    [
        (lambda: distribution_from([1.0], 'wealth', [1.0, 0.0], element=DensityElement), 'increasing'),
        (lambda: distribution_from([], 'wealth', [0.0, 1.0], element=DensityElement), 'no member'),
        (lambda: distribution_from(_traders(), 'income', CLASSES, element=DensityElement), 'Name the state'),
        (lambda: members_from([0.0, 0.0], TraderElement, 3, activities=[0.0, 1.0]), 'not all zero'),
        (lambda: members_from([1.0], TraderElement, 3, activities=[0.0, 1.0]), 'one nonnegative density'),
    ],
)
def test_level_conversions_reject_bad_input(call, message):
    """Test that the conversions reject bad activities, empty groups, unknown states, and bad densities."""
    with pytest.raises(ValidationError, match=message):
        call()
