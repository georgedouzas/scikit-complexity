"""Test the `_hamiltonian` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, var, vector

from skcomplex.core.elements import Element
from skcomplex.core.interactions import HamiltonianInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.errors import ValidationError
from skcomplex.specs import RoleSpec, StateSpec


class OscillatorElement(Element):
    """Element with a canonical pair of order 1."""

    def __init__(
        self,
        label: str,
        *,
        q: Annotated[object, StateSpec()] = 1.0,
        p: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.q = q
        self.p = p


class SecondOrderElement(Element):
    """Element with a coordinate of order 2."""

    def __init__(
        self,
        label: str,
        *,
        q: Annotated[object, StateSpec(order=2)] = 1.0,
        q_d1: object = 0.0,
        p: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.q = q
        self.q_d1 = q_d1
        self.p = p


class RotorElement(Element):
    """Element with an angle and its momentum."""

    def __init__(
        self,
        label: str,
        *,
        q: Annotated[object, StateSpec(period=2 * np.pi)] = 0.0,
        p: Annotated[object, StateSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.q = q
        self.p = p


class CouplingInteraction(HamiltonianInteraction):
    """Spring between the coordinates of two oscillators, whose Hamiltonian holds only the coupling energy."""

    def __init__(self, label: str, *, first: Annotated[str, RoleSpec()], second: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=first)
        self.first = first
        self.second = second

    def _get_hamiltonian(self, s):
        """Return the energy of the spring."""
        return 0.5 * (s.first.q - s.second.q) ** 2


def test_run_returns_the_rate_laws_of_an_oscillator():
    """Test that the Hamiltonian of an oscillator gives the rates `p / m` and `-k q`."""
    q, p, m, k, t = var('q p m k t')
    interaction = HamiltonianInteraction(
        'oscillator',
        target='mass',
        hamiltonian=lambda s: s.target.p**2 / (2 * m) + k * s.target.q**2 / 2,
    )
    laws = interaction.check().prepare().run(SimpleNamespace(target=SimpleNamespace(q=q, p=p), t=t)).laws_
    assert [type(law) for law in laws] == [RateLaw, RateLaw]
    assert laws[0].target is q
    assert bool(laws[0].value == p / m)
    assert laws[1].target is p
    assert bool(laws[1].value == -k * q)


def test_run_differentiates_every_component_of_a_vector_state():
    """Test that a vector coordinate and momentum give the rates of every component."""
    q1, q2, p1, p2, t = var('q1 q2 p1 p2 t')
    interaction = HamiltonianInteraction(
        'kepler',
        target='body',
        hamiltonian=lambda s: s.target.p.dot_product(s.target.p) / 2 + s.target.q[0] * s.target.q[1],
    )
    namespace = SimpleNamespace(target=SimpleNamespace(q=vector([q1, q2]), p=vector([p1, p2])), t=t)
    laws = interaction.check().prepare().run(namespace).laws_
    assert list(laws[0].value) == [p1, p2]
    assert list(laws[1].value) == [-q2, -q1]


def test_run_sums_the_hamiltonians_of_a_coupling():
    """Test that a coupling Hamiltonian adds its rates to those of the free oscillators."""
    free = [
        HamiltonianInteraction(
            f'free_{label}',
            target=label,
            hamiltonian=lambda s: s.target.p**2 / 2 + s.target.q**2 / 2,
        )
        for label in ('left', 'right')
    ]
    system = System(
        elements=[OscillatorElement('left', q=1.0), OscillatorElement('right', q=1.0)],
        interactions=[*free, CouplingInteraction('spring', first='left', second='right')],
    )
    simulation = Simulation(system, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=1.0).check().prepare()
    trajectory = simulation.run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory.values('left.q')[-1], np.cos(1.0), atol=1e-7)
    np.testing.assert_allclose(trajectory.values('left.p')[-1], -np.sin(1.0), atol=1e-7)


def test_run_wraps_an_angle_coordinate():
    """Test that an angle coordinate of a rotor wraps around its period."""

    rotor = HamiltonianInteraction('rotor', target='wheel', hamiltonian=lambda s: s.target.p**2 / 2 - cos(s.target.q))
    system = System(elements=[RotorElement('wheel')], interactions=[rotor])
    trajectory = Simulation(system, dt=0.01).check().prepare().run(until=20.0).trajectory_
    values = trajectory.values('wheel.q')
    assert np.abs(values).max() <= np.pi
    assert np.abs(np.diff(values)).max() > np.pi


def test_check_rejects_a_coordinate_without_a_momentum():
    """Test that two coordinates with one momentum name the unpaired coordinate."""
    interaction = HamiltonianInteraction('pairs', target='mass', coordinates=('x', 'y'), momenta='px')
    with pytest.raises(ValidationError, match='y has no partner'):
        interaction.check()


def test_check_rejects_a_hamiltonian_that_is_not_callable():
    """Test that `check` rejects a Hamiltonian that is not a function."""
    with pytest.raises(ValidationError, match='hamiltonian'):
        HamiltonianInteraction('energy', target='mass', hamiltonian='kinetic').check()


def test_run_rejects_a_momentum_of_another_size():
    """Test that a coordinate and a momentum of different sizes are rejected."""
    q, p1, p2, t = var('q p1 p2 t')
    interaction = HamiltonianInteraction('energy', target='mass', hamiltonian=lambda s: q)
    namespace = SimpleNamespace(target=SimpleNamespace(q=q, p=vector([p1, p2])), t=t)
    with pytest.raises(ValidationError, match='differ in size'):
        interaction.check().prepare().run(namespace)


def test_prepare_reports_a_coordinate_of_order_2():
    """Test that a coordinate of order 2 is one error at the path of the interaction with the fixes."""
    energy = HamiltonianInteraction('energy', target='mass', hamiltonian=lambda s: s.target.p**2 + s.target.q**2)
    system = System(elements=[SecondOrderElement('mass')], interactions=[energy])
    errors = [issue for issue in PassCompiler().check().prepare(system).report_.issues if issue.severity == 'error']
    assert len(errors) == 1
    assert errors[0].path == 'energy'
    assert 'StateSpec(order=1)' in errors[0].message
    assert 'EulerLagrangeInteraction' in errors[0].message
