"""Test the `_moments` module of `skcomplex.core.components.laws.utils`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import bilinear, moment
from skcomplex.core.components.laws.utils._reductions import reductions_in
from skcomplex.core.components.spaces import BaseSpace, LatticeSpace
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

CLASSES = 9
ACTIVITIES = np.linspace(0.0, 1.0, CLASSES)
PROFILE = np.array([0.05, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02])
CLASSES_SPACE = LatticeSpace(CLASSES, boundary='fixed')


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount, at every site of its space."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x


class UpwardInteraction(Interaction):
    """Encounters that move a candidate one class up with probability one half."""

    def __init__(
        self,
        label: str,
        *,
        population: Role,
        eta: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.population = population
        self.eta = eta

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the encounter."""
        super()._prepare(context)
        s = context
        table = np.zeros((CLASSES, CLASSES, CLASSES))
        for candidate in range(CLASSES):
            table[candidate, :, min(candidate + 1, CLASSES - 1)] += 0.5
            table[candidate, :, candidate] += 0.5
        self.laws_ = [
            RateLaw(s.population.x, bilinear(s.eta, table, s.population.x, s.population.x, conserved=True)),
        ]


class ReadMeanInteraction(Interaction):
    """Rate of a recorder equal to the mean activity of a distribution, a moment read inside a law."""

    def __init__(
        self,
        label: str,
        *,
        recorder: Role,
        population: Role,
    ) -> None:
        super().__init__(label)
        self.recorder = recorder
        self.population = population

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate."""
        super()._prepare(context)
        s = context
        self.laws_ = [RateLaw(s.recorder.y, moment(s.population.x, 1, ACTIVITIES, normalized=True))]


def density(s):
    """Return the density of a distribution."""
    return moment(s.x, 0, ACTIVITIES)


def mean_activity(s):
    """Return the mean activity of a distribution."""
    return moment(s.x, 1, ACTIVITIES, normalized=True)


def activity_variance(s):
    """Return the variance of the activity of a distribution."""
    return moment(s.x, 2, ACTIVITIES, central=True)


OBSERVABLES = [
    Observable('density', density, group='wealth'),
    Observable('mean_activity', mean_activity, group='wealth'),
    Observable('variance', activity_variance, group='wealth'),
]


def _system():
    """Return a distribution of wealth over 9 classes as a site group on a lattice of 9 sites."""
    return System(
        elements=[AmountElement('wealth', x=PROFILE, space=CLASSES_SPACE)],
        interactions=[UpwardInteraction('upward', population='wealth')],
    )


def test_moment_of_order_zero_is_the_total():
    """Test that the order-0 moment is the total and higher orders weight the members by their coordinates."""
    f = var('f')
    assert str(moment(f, 0, ACTIVITIES)) == 'total(f)'
    assert 'weights' in str(moment(f, 2, ACTIVITIES))


@pytest.mark.parametrize(('order', 'coordinates'), [(-1, [0.0, 1.0]), (1.5, [0.0, 1.0]), (1, [[0.0, 1.0]]), (1, [])])
def test_moment_rejects_bad_orders_and_coordinates(order, coordinates):
    """Test that a moment rejects orders that are not nonnegative integers and coordinates that are not a vector."""
    with pytest.raises(ValidationError):
        moment(var('x'), order, coordinates)


def test_normalised_start_holds_the_profile():
    """Test that a distribution starts at its profile and its order-0 moment equals the profile's sum."""
    trajectory = _system().prepare().simulate(until=0, observables=OBSERVABLES, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('wealth.x')[0], PROFILE)
    assert trajectory.get_values('density')[0] == pytest.approx(PROFILE.sum())


def test_mean_activity_is_recorded_at_every_frame():
    """Test that the mean activity moment is the activity-weighted sum divided by the density at every frame."""
    trajectory = _system().prepare().simulate(until=5, observables=OBSERVABLES, dt=0.5)
    states = np.asarray(trajectory.get_values('wealth.x'))
    expected = states @ ACTIVITIES / states.sum(axis=1)
    np.testing.assert_allclose(trajectory.get_values('mean_activity'), expected, rtol=1e-12)
    variance = states @ ACTIVITIES**2 / states.sum(axis=1) - expected**2
    np.testing.assert_allclose(trajectory.get_values('variance'), variance, rtol=1e-9, atol=1e-12)
    assert trajectory.get_values('mean_activity')[-1] > trajectory.get_values('mean_activity')[0]


def test_law_reads_a_moment():
    """Test that a rate law reads the mean activity moment of a distribution."""
    system = System(
        elements=[AmountElement('wealth', x=PROFILE, space=CLASSES_SPACE), Element('recorder', values={'y': 0.0})],
        interactions=[ReadMeanInteraction('read', recorder='recorder', population='wealth')],
        backend=NumpyBackend(),
    )
    problem = system.prepare()._problem
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = float(PROFILE @ ACTIVITIES / PROFILE.sum())
    assert rates[problem.states.get_entry('recorder.y').start] == pytest.approx(expected)


def test_moment_inside_a_group_law_is_shared():
    """Test that a moment read by every member of the reduced group gives every member the same value."""

    class RelaxInteraction(Interaction):
        """Relaxation of every class toward the mean activity, a moment that every member reads."""

        def __init__(
            self,
            label: str,
            *,
            population: Role,
        ) -> None:
            super().__init__(label)
            self.population = population

        def _prepare(self, context: SimpleNamespace) -> None:
            """Set the rate."""
            super()._prepare(context)
            s = context
            self.laws_ = [RateLaw(s.population.x, moment(s.population.x, 1, ACTIVITIES))]

    system = System(
        elements=[AmountElement('wealth', x=PROFILE, space=CLASSES_SPACE)],
        interactions=[RelaxInteraction('relax', population='wealth')],
        backend=NumpyBackend(),
    )
    problem = system.prepare()._problem
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    np.testing.assert_allclose(rates, np.full(CLASSES, PROFILE @ ACTIVITIES))


def test_normalised_moment_divides_a_weighted_total_by_the_total():
    """Test that a normalised moment holds one weighted and one plain total."""
    found = reductions_in(moment(var('f'), 1, [0.0, 1.0, 2.0], normalized=True))
    assert sorted(item.weights is None for item in found) == [False, True]


def test_moment_over_an_axis_sums_the_powers_of_its_coordinate():
    """Test that a moment over one axis of a phase space sums the field times the power of the coordinate."""
    assert str(moment(var('f'), 2, axes=('v',))) == 'reduce_axes(_coordinate_v^2*f, _coordinate_v)'
