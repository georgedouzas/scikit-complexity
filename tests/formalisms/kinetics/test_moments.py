"""Test the moments of distributions held by site groups over grids of activity values."""

import numpy as np
import pytest

from skcomplex.core.base import NumericDomain, ValidationError, parameter, role, state, symbol
from skcomplex.core.components import ElementComponent, GroupComponent
from skcomplex.core.interactions import BaseInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.space import ContinuousTime, LatticeSpace
from skcomplex.core.system import Observable, System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.kinetics import EncounterLaw, moment

CLASSES = 9
ACTIVITIES = np.linspace(0.0, 1.0, CLASSES)
PROFILE = np.array([0.05, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02])


class DensityComponent(ElementComponent):
    """Density of particles at one activity value."""

    f: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE)


class RecorderComponent(ElementComponent):
    """Element whose rate is the mean activity of a distribution."""

    y: float = state(default=0.0)


class UpwardInteraction(BaseInteraction):
    """Encounters that move a candidate one class up with probability one half."""

    population: str = role()
    eta: float = parameter(default=1.0)

    def laws(self, s):
        """Return the encounter."""
        table = np.zeros((CLASSES, CLASSES, CLASSES))
        for candidate in range(CLASSES):
            table[candidate, :, min(candidate + 1, CLASSES - 1)] += 0.5
            table[candidate, :, candidate] += 0.5
        return [EncounterLaw(s.population.f, s.population.f, s.eta, table)]


class ReadMeanInteraction(BaseInteraction):
    """Rate of a recorder equal to the mean activity of a distribution, a moment read inside a law."""

    recorder: str = role()
    population: str = role()

    def laws(self, s):
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
    Observable('density', density, group='wealth'),
    Observable('mean_activity', mean_activity, group='wealth'),
    Observable('variance', activity_variance, group='wealth'),
]


def _system():
    """Return a distribution of wealth over 9 classes as a site group on a lattice of 9 sites."""
    return System(
        components=[GroupComponent('wealth', element=DensityComponent, on='sites', values={'f': PROFILE})],
        interactions=[UpwardInteraction('upward', population='wealth')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )


def test_moment_of_order_zero_is_the_total():
    """Test that the order-0 moment is the total and higher orders weight the members by their coordinates."""
    f = symbol('f')
    assert str(moment(f, 0, ACTIVITIES)) == 'total(f)'
    assert 'weights' in str(moment(f, 2, ACTIVITIES))


@pytest.mark.parametrize(('order', 'coordinates'), [(-1, [0.0, 1.0]), (1.5, [0.0, 1.0]), (1, [[0.0, 1.0]]), (1, [])])
def test_moment_rejects_bad_orders_and_coordinates(order, coordinates):
    """Test that a moment rejects orders that are not nonnegative integers and coordinates that are not a vector."""
    with pytest.raises(ValidationError):
        moment(symbol('f'), order, coordinates)


def test_normalised_start_holds_the_profile():
    """Test that a distribution starts at its profile and its order-0 moment equals the profile's sum."""
    simulation = Simulation(_system(), observables=OBSERVABLES, dt=1.0).run(until=0)
    np.testing.assert_allclose(simulation.trajectory_['wealth.f'][0], PROFILE)
    assert simulation.trajectory_['density'][0] == pytest.approx(PROFILE.sum())


def test_mean_activity_is_recorded_at_every_frame():
    """Test that the mean activity moment is the activity-weighted sum divided by the density at every frame."""
    trajectory = Simulation(_system(), observables=OBSERVABLES, dt=0.5).run(until=5).trajectory_
    states = np.asarray(trajectory['wealth.f'])
    expected = states @ ACTIVITIES / states.sum(axis=1)
    np.testing.assert_allclose(trajectory['mean_activity'], expected, rtol=1e-12)
    variance = states @ ACTIVITIES**2 / states.sum(axis=1) - expected**2
    np.testing.assert_allclose(trajectory['variance'], variance, rtol=1e-9, atol=1e-12)
    assert trajectory['mean_activity'][-1] > trajectory['mean_activity'][0]


def test_law_reads_a_moment():
    """Test that a rate law reads the mean activity moment of a distribution."""
    system = System(
        components=[
            GroupComponent('wealth', element=DensityComponent, on='sites', values={'f': PROFILE}),
            RecorderComponent('recorder'),
        ],
        interactions=[ReadMeanInteraction('read', recorder='recorder', population='wealth')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = compile_system(system, backend='numpy')
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = float(PROFILE @ ACTIVITIES / PROFILE.sum())
    assert rates[problem.states.entry('recorder.y').start] == pytest.approx(expected)


def test_moment_inside_a_group_law_is_shared():
    """Test that a moment read by every member of the reduced group gives every member the same value."""

    class RelaxInteraction(BaseInteraction):
        """Relaxation of every class toward the mean activity, a moment that every member reads."""

        population: str = role()

        def laws(self, s):
            """Return the rate."""
            return [RateLaw(s.population.f, moment(s.population.f, 1, ACTIVITIES))]

    system = System(
        components=[GroupComponent('wealth', element=DensityComponent, on='sites', values={'f': PROFILE})],
        interactions=[RelaxInteraction('relax', population='wealth')],
        space=LatticeSpace(CLASSES, boundary='fixed'),
        time=ContinuousTime(),
    )
    problem = compile_system(system, backend='numpy')
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    np.testing.assert_allclose(rates, np.full(CLASSES, PROFILE @ ACTIVITIES))
