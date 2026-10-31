"""Test the `_euler_lagrange` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, sin, var
from scipy.linalg import expm

from skcomplex.core.elements import Element
from skcomplex.core.interactions import EulerLagrangeInteraction
from skcomplex.core.laws import EquationLaw
from skcomplex.core.laws.utils import derivative
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.errors import ValidationError
from skcomplex.specs import RoleSpec, StateSpec

DAMPING, FORCE, OMEGA = 0.3, 0.5, 1.3


class AngleElement(Element):
    """Element with an angle of order 2."""

    def __init__(self, label: str, *, q: Annotated[object, StateSpec(order=2)] = 1.0, q_d1: object = 0.0) -> None:
        super().__init__(label)
        self.q = q
        self.q_d1 = q_d1


class PendulumElement(AngleElement):
    """Damped and driven pendulum written as an equation by hand."""

    def _get_laws(self, s):
        """Return the equation of the pendulum."""
        return [EquationLaw(derivative(s.q, 2) == -sin(s.q) - DAMPING * derivative(s.q, 1) + FORCE * cos(OMEGA * s.t))]


class PairElement(Element):
    """Element with two coordinates of order 2."""

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[object, StateSpec(order=2)] = 1.0,
        a_d1: object = 0.0,
        b: Annotated[object, StateSpec(order=2)] = 0.0,
        b_d1: object = 0.0,
    ) -> None:
        super().__init__(label)
        self.a = a
        self.a_d1 = a_d1
        self.b = b
        self.b_d1 = b_d1


def _run(system, until):
    """Return the trajectory of a system solved to a tight tolerance."""
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    return Simulation(system, solver=solver, dt=0.1).check().prepare().run(until=until).trajectory_


def test_run_matches_the_equation_of_a_damped_driven_pendulum():
    """Test that the Lagrangian, the dissipation, and the forcing of a pendulum give its equation of motion."""
    pendulum = EulerLagrangeInteraction(
        'pendulum',
        target='bob',
        states='q',
        lagrangian=lambda s: derivative(s.target.q, 1) ** 2 / 2 + cos(s.target.q),
        dissipation=lambda s: DAMPING * derivative(s.target.q, 1) ** 2 / 2,
        forcing=lambda s: FORCE * cos(OMEGA * s.t),
    )
    by_lagrangian = _run(System(elements=[AngleElement('bob')], interactions=[pendulum]), 10.0)
    by_hand = _run(System(elements=[PendulumElement('bob')]), 10.0)
    np.testing.assert_allclose(by_lagrangian.values('bob.q'), by_hand.values('bob.q'), atol=1e-7)


def test_run_solves_a_mass_matrix_of_several_coordinates():
    """Test that coordinates coupled through their velocities follow `M q'' = -q` for the mass matrix `M`."""
    coupled = EulerLagrangeInteraction(
        'coupled',
        target='pair',
        states=('a', 'b'),
        lagrangian=lambda s: (
            derivative(s.target.a, 1) ** 2 / 2
            + derivative(s.target.b, 1) ** 2 / 2
            + derivative(s.target.a, 1) * derivative(s.target.b, 1) / 2
            - (s.target.a**2 + s.target.b**2) / 2
        ),
    )
    trajectory = _run(System(elements=[PairElement('pair')], interactions=[coupled]), 5.0)
    inverse = np.linalg.inv(np.array([[1.0, 0.5], [0.5, 1.0]]))
    generator = np.block([[np.zeros((2, 2)), np.eye(2)], [-inverse, np.zeros((2, 2))]])
    expected = np.array([expm(generator * time) @ [1.0, 0.0, 0.0, 0.0] for time in trajectory.times])
    np.testing.assert_allclose(trajectory.values('pair.a'), expected[:, 0], atol=1e-7)
    np.testing.assert_allclose(trajectory.values('pair.b'), expected[:, 1], atol=1e-7)


def test_run_rejects_a_forcing_of_another_size():
    """Test that a forcing with more forces than coordinates is rejected."""
    q, t = var('q t')
    interaction = EulerLagrangeInteraction(
        'swing',
        target='bob',
        states='q',
        lagrangian=lambda s: q**2,
        forcing=lambda s: [1, 2],
    )
    with pytest.raises(ValidationError, match='2 forces for 1 coordinates'):
        interaction.check().prepare().run(SimpleNamespace(target=SimpleNamespace(q=q), t=t))


def test_check_rejects_a_lagrangian_that_is_not_callable():
    """Test that `check` rejects a Lagrangian that is not a function."""
    with pytest.raises(ValidationError, match='lagrangian'):
        EulerLagrangeInteraction('swing', target='bob', lagrangian='kinetic').check()


class SpringInteraction(EulerLagrangeInteraction):
    """Spring between the angles of two elements, whose Lagrangian holds only the energy of the spring."""

    def __init__(self, label: str, *, first: Annotated[str, RoleSpec()], second: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=first, states='q')
        self.first = first
        self.second = second

    def _get_lagrangian(self, s):
        """Return minus the energy of the spring."""
        return -((s.second.q - s.first.q) ** 2) / 2


def test_run_adds_the_equations_of_one_coordinate_from_several_interactions():
    """Test that a coupling Lagrangian adds to the free ones: two equal masses exchange the energy of their modes."""
    free = [
        EulerLagrangeInteraction(
            f'free_{label}',
            target=label,
            states='q',
            lagrangian=lambda s: derivative(s.target.q, 1) ** 2 / 2 - s.target.q**2 / 2,
        )
        for label in ('left', 'right')
    ]
    system = System(
        elements=[AngleElement('left', q=1.0), AngleElement('right', q=1.0)],
        interactions=[*free, SpringInteraction('spring', first='left', second='right')],
    )
    simulation = Simulation(system, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=1.0).check().prepare()
    trajectory = simulation.run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory.values('left.q')[-1], np.cos(1.0), atol=1e-7)
    np.testing.assert_allclose(trajectory.values('right.q')[-1], np.cos(1.0), atol=1e-7)
