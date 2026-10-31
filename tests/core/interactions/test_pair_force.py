"""Test the `_pair_force` module of `skcomplex.core.interactions`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import exp

from skcomplex.core.elements import GroupElement, ParticleElement
from skcomplex.core.interactions import PairForceInteraction
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.errors import ValidationError


def _morse(namespace, r):
    """Return the Morse energy of depth 2, width 1.5, and minimum at 1."""
    return 2 * (1 - exp(-1.5 * (r - 1))) ** 2


def _morse_energy(first, second):
    """Return the numeric Morse energy of two members at their positions."""
    r = np.linalg.norm(first - second)
    return 2 * (1 - np.exp(-1.5 * (r - 1))) ** 2


def _pair_accelerations(positions, masses, cutoff=3.0):
    """Return the compiled accelerations of a pair of particles under the Morse energy."""
    group = GroupElement('pair', element=ParticleElement, size=2, values={'x': positions, 'm': masses})
    system = System(
        elements=[group],
        interactions=[PairForceInteraction('morse', 'pair', 'pair', cutoff=cutoff, energy=_morse)],
        space=ContinuousSpace(n_dim=2),
    )
    problem = compile_system(system)
    entry = problem.states.entry('pair.v')
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    return drift[entry.start : entry.start + entry.size].reshape(entry.rows, entry.columns)


def test_pair_force_equals_minus_the_gradient_of_the_energy():
    """Test that the force on each member is minus the gradient of the energy, divided by its mass."""
    positions, masses = np.array([[0.0, 0.0], [1.2, 0.9]]), np.array([1.0, 2.0])
    step = 1e-6
    shifts = step * np.eye(2)
    gradient = np.array(
        [
            (_morse_energy(positions[0] + shift, positions[1]) - _morse_energy(positions[0] - shift, positions[1]))
            / (2 * step)
            for shift in shifts
        ],
    )
    accelerations = _pair_accelerations(positions, masses)
    np.testing.assert_allclose(accelerations[0], -gradient / masses[0], rtol=1e-6)
    np.testing.assert_allclose(accelerations[1], gradient / masses[1], rtol=1e-6)


def test_pair_force_vanishes_beyond_the_cutoff():
    """Test that members farther apart than the cutoff feel no force."""
    accelerations = _pair_accelerations(np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([1.0, 1.0]), cutoff=1.5)
    np.testing.assert_allclose(accelerations, 0.0)


def test_pair_force_holds_members_at_the_energy_minimum():
    """Test that two members at the minimum of the energy stay at rest."""
    group = GroupElement('pair', element=ParticleElement, size=2, values={'x': [[0.0, 0.0], [1.0, 0.0]]})
    system = System(
        elements=[group],
        interactions=[PairForceInteraction('morse', 'pair', 'pair', cutoff=3.0, energy=_morse)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = Simulation(system, solver=ScipyODESolver(), dt=0.1).check().prepare().run(until=1).trajectory_
    np.testing.assert_allclose(trajectory.values('pair.v'), 0.0, atol=1e-12)


def test_check_rejects_an_energy_that_is_not_callable():
    """Test that `check` rejects an energy that is not a callable."""
    with pytest.raises(ValidationError, match='callable'):
        PairForceInteraction('morse', 'pair', 'pair', energy=1.0).check()
