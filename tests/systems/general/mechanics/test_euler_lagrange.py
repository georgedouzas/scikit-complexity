"""Test the `_euler_lagrange` module of `skcomplex.systems.general.mechanics`."""

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, function, sin, var
from scipy.linalg import expm

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.general.mechanics import EulerLagrangeInteraction

DAMPING, FORCE, OMEGA = 0.3, 0.5, 1.3


class PendulumInteraction(Interaction):
    """Damped and driven pendulum written as an equation by hand."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        """Set the equation of the pendulum."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            q, t = context.target.q, context.t
            rate = -sin(q) - DAMPING * q.diff(t) + FORCE * cos(OMEGA * t)
            self.laws_ = [RateLaw(q.diff(t, 2), rate)]


def _run(elements, interactions, until, dt=0.1):
    """Return the trajectory of a system solved to a tight tolerance."""
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    system = System(elements=elements, interactions=interactions, solver=solver)
    return system.prepare().simulate(until=until, dt=dt)


def test_simulate_matches_the_equation_of_a_damped_driven_pendulum():
    """Test that the Lagrangian, the dissipation, and the forcing of a pendulum give its equation of motion."""
    pendulum = EulerLagrangeInteraction(
        'pendulum',
        target='bob',
        states='q',
        lagrangian=lambda s: s.target.q.diff(s.t) ** 2 / 2 + cos(s.target.q),
        dissipation=lambda s: DAMPING * s.target.q.diff(s.t) ** 2 / 2,
        forcing=lambda s: FORCE * cos(OMEGA * s.t),
    )
    by_lagrangian = _run([Element('bob', values={'q': 1.0})], [pendulum], 10.0)
    by_hand = _run([Element('bob', values={'q': 1.0})], [PendulumInteraction('pendulum', target='bob')], 10.0)
    np.testing.assert_allclose(by_lagrangian.get_values('bob.q'), by_hand.get_values('bob.q'), atol=1e-7)


def test_simulate_solves_a_mass_matrix_of_several_coordinates():
    """Test that coordinates coupled through their velocities follow `M q'' = -q` for the mass matrix `M`."""
    coupled = EulerLagrangeInteraction(
        'coupled',
        target='pair',
        states=('a', 'b'),
        lagrangian=lambda s: (
            s.target.a.diff(s.t) ** 2 / 2
            + s.target.b.diff(s.t) ** 2 / 2
            + s.target.a.diff(s.t) * s.target.b.diff(s.t) / 2
            - (s.target.a**2 + s.target.b**2) / 2
        ),
    )
    trajectory = _run([Element('pair', values={'a': 1.0, 'b': 0.0})], [coupled], 5.0)
    inverse = np.linalg.inv(np.array([[1.0, 0.5], [0.5, 1.0]]))
    generator = np.block([[np.zeros((2, 2)), np.eye(2)], [-inverse, np.zeros((2, 2))]])
    expected = np.array([expm(generator * time) @ [1.0, 0.0, 0.0, 0.0] for time in trajectory.times])
    np.testing.assert_allclose(trajectory.get_values('pair.a'), expected[:, 0], atol=1e-7)
    np.testing.assert_allclose(trajectory.get_values('pair.b'), expected[:, 1], atol=1e-7)


def test_prepare_rejects_a_forcing_of_another_size():
    """Test that a forcing with more forces than coordinates is rejected."""
    t = var('t')
    q = function('q')(t)
    interaction = EulerLagrangeInteraction(
        'swing',
        target='bob',
        states='q',
        lagrangian=lambda s: s.target.q**2,
        forcing=lambda s: [1, 2],
    )
    with pytest.raises(ValidationError, match='2 forces for 1 coordinates'):
        interaction.prepare(SimpleNamespace(target=SimpleNamespace(q=q), t=t))


def test_prepare_rejects_a_lagrangian_that_is_not_callable():
    """Test that `prepare` rejects a Lagrangian that is not a function."""
    with pytest.raises(ValidationError, match='lagrangian'):
        EulerLagrangeInteraction('swing', target='bob', lagrangian='kinetic').prepare()


class SpringInteraction(EulerLagrangeInteraction):
    """Spring between the angles of two elements, whose Lagrangian holds only the energy of the spring."""

    def __init__(self, label: str, *, first: Role, second: Role) -> None:
        super().__init__(label, target=first, states='q')
        self.first = first
        self.second = second

    def _get_lagrangian(self, s):
        """Return minus the energy of the spring."""
        return -((s.second.q - s.first.q) ** 2) / 2


def test_simulate_adds_the_equations_of_one_coordinate_from_several_interactions():
    """Test that a coupling Lagrangian adds to the free ones: two equal masses exchange the energy of their modes."""
    free = [
        EulerLagrangeInteraction(
            f'free_{label}',
            target=label,
            states='q',
            lagrangian=lambda s: s.target.q.diff(s.t) ** 2 / 2 - s.target.q**2 / 2,
        )
        for label in ('left', 'right')
    ]
    elements = [Element('left', values={'q': 1.0}), Element('right', values={'q': 1.0})]
    trajectory = _run(elements, [*free, SpringInteraction('spring', first='left', second='right')], 1.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('left.q')[-1], np.cos(1.0), atol=1e-7)
    np.testing.assert_allclose(trajectory.get_values('right.q')[-1], np.cos(1.0), atol=1e-7)
