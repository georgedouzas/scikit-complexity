"""Test the `_hamiltonian` module of `skcomplex.systems.physics.common.mechanics`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, var, vector

from skcomplex.core.components.elements import Element, Period
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.physics.common.mechanics import HamiltonianInteraction


class OscillatorElement(Element):
    """Element with a canonical pair."""

    def __init__(self, label: str, *, q: object = 1.0, p: object = 0.0) -> None:
        super().__init__(label)
        self.q = q
        self.p = p


class RotorElement(Element):
    """Element with an angle and its momentum."""

    def __init__(
        self,
        label: str,
        *,
        q: Annotated[object, Period(2 * np.pi)] = 0.0,
        p: object = 2.0,
    ) -> None:
        super().__init__(label)
        self.q = q
        self.p = p


class SpringCouplingInteraction(HamiltonianInteraction):
    """Spring between the coordinates of two oscillators, whose Hamiltonian holds only the coupling energy."""

    def __init__(self, label: str, *, first: Role, second: Role) -> None:
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
    laws = interaction.prepare().run(SimpleNamespace(target=SimpleNamespace(q=q, p=p), t=t)).laws_
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
    laws = interaction.prepare().run(namespace).laws_
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
        interactions=[*free, SpringCouplingInteraction('spring', first='left', second='right')],
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=1.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('left.q')[-1], np.cos(1.0), atol=1e-7)
    np.testing.assert_allclose(trajectory.get_values('left.p')[-1], -np.sin(1.0), atol=1e-7)


def test_run_wraps_an_angle_coordinate():
    """Test that an angle coordinate of a rotor wraps around its period."""
    rotor = HamiltonianInteraction('rotor', target='wheel', hamiltonian=lambda s: s.target.p**2 / 2 - cos(s.target.q))
    system = System(elements=[RotorElement('wheel')], interactions=[rotor])
    trajectory = system.prepare().simulate(until=20.0, dt=0.01)
    values = trajectory.get_values('wheel.q')
    assert np.abs(values).max() <= np.pi
    assert np.abs(np.diff(values)).max() > np.pi


def test_check_rejects_a_coordinate_without_a_momentum():
    """Test that two coordinates with one momentum name the unpaired coordinate."""
    interaction = HamiltonianInteraction('pairs', target='mass', coordinates=('x', 'y'), momenta='px')
    with pytest.raises(ValidationError, match='y has no partner'):
        interaction.prepare()


def test_check_rejects_a_hamiltonian_that_is_not_callable():
    """Test that `check` rejects a Hamiltonian that is not a function."""
    with pytest.raises(ValidationError, match='hamiltonian'):
        HamiltonianInteraction('energy', target='mass', hamiltonian='kinetic').prepare()


def test_run_rejects_a_momentum_of_another_size():
    """Test that a coordinate and a momentum of different sizes are rejected."""
    q, p1, p2, t = var('q p1 p2 t')
    interaction = HamiltonianInteraction('energy', target='mass', hamiltonian=lambda s: q)
    namespace = SimpleNamespace(target=SimpleNamespace(q=q, p=vector([p1, p2])), t=t)
    with pytest.raises(ValidationError, match='differ in size'):
        interaction.prepare().run(namespace)


def test_prepare_reports_a_coordinate_of_order_2():
    """Test that a coordinate that another law gives a second derivative is rejected with both orders and the fix."""
    energy = HamiltonianInteraction('energy', target='mass', hamiltonian=lambda s: s.target.p**2 + s.target.q**2)
    push = Interaction('push', roles={'target': 'mass'}, laws=lambda s: [RateLaw(s.target.q.diff(s.t, 2), 0.0)])
    mass = Element('mass', values={'q': 1.0, "q'": 0.0, 'p': 0.0})
    system = System(elements=[mass], interactions=[energy, push])
    with pytest.raises(ValidationError, match=r'mass\.q has rate laws of order 1 and of order 2') as error:
        system.prepare()
    assert 'Write each law on the same derivative' in str(error.value)
