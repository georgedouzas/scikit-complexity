"""Test the `_moments` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EncounterLaw, RateLaw
from skcomplex.core.components.laws.utils import moment
from skcomplex.core.components.spaces import ContinuousTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec

CLASSES = 9
ACTIVITIES = np.linspace(0.0, 1.0, CLASSES)
PROFILE = np.array([0.05, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02])


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


class RecorderElement(Element):
    """Element whose rate is the mean activity of a distribution."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y


class UpwardInteraction(Interaction):
    """Encounters that move a candidate one class up with probability one half."""

    def __init__(
        self,
        label: str,
        *,
        population: Annotated[str, RoleSpec()],
        eta: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.population = population
        self.eta = eta

    def _get_laws(self, s):
        """Return the encounter."""
        table = np.zeros((CLASSES, CLASSES, CLASSES))
        for candidate in range(CLASSES):
            table[candidate, :, min(candidate + 1, CLASSES - 1)] += 0.5
            table[candidate, :, candidate] += 0.5
        return [EncounterLaw(s.population.f, s.population.f, s.eta, table)]


class ReadMeanInteraction(Interaction):
    """Rate of a recorder equal to the mean activity of a distribution, a moment read inside a law."""

    def __init__(
        self,
        label: str,
        *,
        recorder: Annotated[str, RoleSpec()],
        population: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.recorder = recorder
        self.population = population

    def _get_laws(self, s):
        """Return the rate."""
        return [RateLaw(s.recorder.y, moment(s.population.f, 1, ACTIVITIES, normalized=True))]


def density(s):
    """Return the density of a distribution."""
    return moment(s.f, 0, ACTIVITIES)


def mean_activity(s):
    """Return the mean activity of a distribution."""
    return moment(s.f, 1, ACTIVITIES, normalized=True)


def activity_variance(s):
    """Return the variance of the activity of a distribution."""
    return moment(s.f, 2, ACTIVITIES, central=True)


OBSERVABLES = [
    ObservableSpec('density', density, group='wealth'),
    ObservableSpec('mean_activity', mean_activity, group='wealth'),
    ObservableSpec('variance', activity_variance, group='wealth'),
]


def _system():
    """Return a distribution of wealth over 9 classes as a site group on a lattice of 9 sites."""
    return System(
        elements=[GroupElement('wealth', element=DensityElement, on='sites', values={'f': PROFILE})],
        interactions=[UpwardInteraction('upward', population='wealth')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
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
        moment(var('f'), order, coordinates)


def test_normalised_start_holds_the_profile():
    """Test that a distribution starts at its profile and its order-0 moment equals the profile's sum."""
    simulation = Simulation(_system(), observables=OBSERVABLES, dt=1.0).prepare().run(until=0)
    np.testing.assert_allclose(simulation.trajectory_.values('wealth.f')[0], PROFILE)
    assert simulation.trajectory_.values('density')[0] == pytest.approx(PROFILE.sum())


def test_mean_activity_is_recorded_at_every_frame():
    """Test that the mean activity moment is the activity-weighted sum divided by the density at every frame."""
    trajectory = Simulation(_system(), observables=OBSERVABLES, dt=0.5).prepare().run(until=5).trajectory_
    states = np.asarray(trajectory.values('wealth.f'))
    expected = states @ ACTIVITIES / states.sum(axis=1)
    np.testing.assert_allclose(trajectory.values('mean_activity'), expected, rtol=1e-12)
    variance = states @ ACTIVITIES**2 / states.sum(axis=1) - expected**2
    np.testing.assert_allclose(trajectory.values('variance'), variance, rtol=1e-9, atol=1e-12)
    assert trajectory.values('mean_activity')[-1] > trajectory.values('mean_activity')[0]


def test_law_reads_a_moment():
    """Test that a rate law reads the mean activity moment of a distribution."""
    system = System(
        elements=[
            GroupElement('wealth', element=DensityElement, on='sites', values={'f': PROFILE}),
            RecorderElement('recorder'),
        ],
        interactions=[ReadMeanInteraction('read', recorder='recorder', population='wealth')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = compile_system(system, backend=NumpyBackend())
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = float(PROFILE @ ACTIVITIES / PROFILE.sum())
    assert rates[problem.states.entry('recorder.y').start] == pytest.approx(expected)


def test_moment_inside_a_group_law_is_shared():
    """Test that a moment read by every member of the reduced group gives every member the same value."""

    class RelaxInteraction(Interaction):
        """Relaxation of every class toward the mean activity, a moment that every member reads."""

        def __init__(
            self,
            label: str,
            *,
            population: Annotated[str, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.population = population

        def _get_laws(self, s):
            """Return the rate."""
            return [RateLaw(s.population.f, moment(s.population.f, 1, ACTIVITIES))]

    system = System(
        elements=[GroupElement('wealth', element=DensityElement, on='sites', values={'f': PROFILE})],
        interactions=[RelaxInteraction('relax', population='wealth')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = compile_system(system, backend=NumpyBackend())
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    np.testing.assert_allclose(rates, np.full(CLASSES, PROFILE @ ACTIVITIES))
