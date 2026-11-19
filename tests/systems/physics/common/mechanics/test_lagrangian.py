"""Test the `_lagrangian` module of `skcomplex.systems.physics.common.mechanics`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.physics.common.mechanics import LagrangianInteraction

SOLVER = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)


class DampedSpringInteraction(LagrangianInteraction):
    """Mass on a spring with viscous damping and a constant force."""

    def __init__(self, label: str, *, target: Role, c: float = 0.0, F: float = 0.0) -> None:
        super().__init__(label)
        self.target = target
        self.c = c
        self.F = F

    def _get_coordinates(self, context):
        return [context.target.x]

    def _get_lagrangian(self, context, positions, velocities):
        return context.target.m * velocities[0] ** 2 / 2 - context.target.k * positions[0] ** 2 / 2

    def _get_dissipation(self, context, positions, velocities):
        return context.c * velocities[0] ** 2 / 2

    def _get_forces(self, context, positions, velocities):
        return [context.F]


class CartesianPairInteraction(LagrangianInteraction):
    """Two coordinates whose kinetic energy couples their velocities."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _get_coordinates(self, context):
        return [context.target.x, context.target.y]

    def _get_lagrangian(self, context, positions, velocities):
        u, v = velocities
        x, y = positions
        return u**2 + u * v + v**2 - (x**2 + y**2) / 2


def test_lagrangian_interaction_damps_and_forces_a_spring():
    """Test that a damped spring under a constant force follows the exact solution about its shifted rest point."""
    mass = Element('mass', values={'x': 0.0, 'm': 1.0, 'k': 4.0})
    motion = DampedSpringInteraction('motion', target='mass', c=0.4, F=2.0)
    trajectory = System(elements=[mass], interactions=[motion], solver=SOLVER).prepare().simulate(until=5.0, dt=0.1)
    t = trajectory.times
    omega = np.sqrt(4.0 - 0.2**2)
    expected = 0.5 - 0.5 * np.exp(-0.2 * t) * (np.cos(omega * t) + 0.2 / omega * np.sin(omega * t))
    np.testing.assert_allclose(trajectory.get_values('mass.x'), expected, atol=1e-8)


def test_lagrangian_interaction_conserves_the_energy_of_coupled_velocities():
    """Test that coordinates with coupled velocities keep the energy `u² + u v + v² + (x² + y²) / 2`."""
    body = Element('body', values={'x': 1.0, 'y': 0.0})
    system = System(elements=[body], interactions=[CartesianPairInteraction('motion', target='body')], solver=SOLVER)
    trajectory = system.prepare().simulate(until=10.0, dt=0.1, initial_state={"body.y'": 0.5})
    x, y, u, v = (trajectory.get_values(name) for name in ('body.x', 'body.y', "body.x'", "body.y'"))
    energy = u**2 + u * v + v**2 + (x**2 + y**2) / 2
    assert np.ptp(energy) < 1e-8
    assert np.ptp(y) > 0.1
