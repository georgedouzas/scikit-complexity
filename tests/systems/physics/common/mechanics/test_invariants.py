"""Test the `_invariants` module of `skcomplex.systems.physics.common.mechanics`."""

from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.analysis import invariants
from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import StateSpec
from skcomplex.systems.physics.celestial_mechanics import SitnikovSystem, TwoBodyOrbitSystem
from skcomplex.systems.physics.common.mechanics import EulerLagrangeInteraction, SymmetrySpec, noether, poisson_brackets
from skcomplex.systems.physics.nonlinear_oscillators import HenonHeilesSystem

MU = 0.01215


class LagrangianBodyElement(Element):
    """Light body of the rotating frame with coordinates of order 2."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.5,
        x_d1: float = 0.0,
        y: Annotated[float, StateSpec(order=2)] = 0.0,
        y_d1: float = 1.15,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1
        self.y = y
        self.y_d1 = y_d1


def _distances(x, y):
    """Return the distances of a point from the two primaries."""
    return sqrt((x + MU) ** 2 + y**2), sqrt((x - 1 + MU) ** 2 + y**2)


def _rotating_lagrangian(s):
    """Return the Lagrangian of the rotating frame."""
    body = s.target
    vx, vy = derivative(body.x, 1), derivative(body.y, 1)
    first, second = _distances(body.x, body.y)
    kinetic = (vx**2 + vy**2) / 2 + body.x * vy - body.y * vx + (body.x**2 + body.y**2) / 2
    return kinetic + (1 - MU) / first + MU / second


def _jacobi(s):
    """Return the Jacobi constant of the light body."""
    body = s.body
    first, second = _distances(body.x, body.y)
    return body.x**2 + body.y**2 + 2 * (1 - MU) / first + 2 * MU / second - body.x_d1**2 - body.y_d1**2


def _lagrangian_three_body() -> CompositeSystem:
    """Return the restricted three-body problem written with an Euler-Lagrange interaction."""
    motion = EulerLagrangeInteraction('motion', target='body', states=('x', 'y'), lagrangian=_rotating_lagrangian)
    return CompositeSystem(elements=[LagrangianBodyElement('body')], interactions=[motion])


def _kepler_expressions() -> list[Observable]:
    """Return the energy, the angular momentum, and the first momentum of the Kepler problem."""
    return [
        Observable(
            'energy',
            lambda s: (s.planet.px**2 + s.planet.py**2) / 2
            - s.planet.G
            * s.planet.mass
            / sqrt(
                s.planet.x**2 + s.planet.y**2,
            ),
        ),
        Observable('angular_momentum', lambda s: s.planet.x * s.planet.py - s.planet.y * s.planet.px),
        Observable('px', lambda s: s.planet.px),
    ]


def test_poisson_brackets_bracket_the_components_of_the_angular_momentum():
    """Test that the bracket of a coordinate with its momentum is one and with the angular momentum is `-y`."""
    system = TwoBodyOrbitSystem().prepare().system_
    expressions = [
        Observable('x', lambda s: s.planet.x),
        Observable('px', lambda s: s.planet.px),
        Observable('angular_momentum', lambda s: s.planet.x * s.planet.py - s.planet.y * s.planet.px),
    ]
    brackets = poisson_brackets(system, expressions)
    namespace = invariants(system).namespace.planet
    assert brackets[('x', 'px')] == 1
    assert bool(brackets[('x', 'angular_momentum')] == -namespace.y)


def test_poisson_brackets_reject_a_system_without_canonical_pairs():
    """Test that brackets of a system without a Hamiltonian interaction raise an error naming it."""
    with pytest.raises(AnalysisError, match='HamiltonianInteraction'):
        poisson_brackets(SitnikovSystem().prepare().system_, [Observable('z', lambda s: s.body.z)])


def test_noether_confirms_the_rotation_invariance_of_kepler_with_its_charge():
    """Test that the rotation of the plane keeps the Kepler Hamiltonian, with the angular momentum as charge."""
    system = TwoBodyOrbitSystem().prepare().system_
    rotation = SymmetrySpec('rotation', lambda s: {'planet.x': -s.planet.y, 'planet.y': s.planet.x})
    result = noether(system, rotation)
    namespace = invariants(system).namespace.planet
    assert result.invariant
    assert result.remainder == 0
    assert bool((result.charge - (namespace.x * namespace.py - namespace.y * namespace.px)).simplify_full() == 0)


def test_noether_reports_the_rotation_charge_of_a_lagrangian():
    """Test that the rotating-frame Lagrangian is not invariant under rotation, since the primaries break it."""
    system = _lagrangian_three_body()
    rotation = SymmetrySpec('rotation', lambda s: {'body.x': -s.body.y, 'body.y': s.body.x})
    result = noether(system, rotation)
    namespace = invariants(system).namespace.body
    assert not result.invariant
    expected = namespace.x * namespace.y_d1 - namespace.y * namespace.x_d1 + namespace.x**2 + namespace.y**2
    assert bool((result.charge - expected).simplify_full() == 0)


def test_noether_reports_a_broken_translation_of_henon_heiles():
    """Test that a translation of the first coordinate changes the Hénon-Heiles Hamiltonian by `x + 2 x y`."""
    system = HenonHeilesSystem().prepare().system_
    result = noether(system, SymmetrySpec('translation', lambda s: {'star.x': 1}))
    namespace = invariants(system).namespace.star
    assert not result.invariant
    assert bool((result.remainder - (namespace.x + 2 * namespace.x * namespace.y)).simplify_full() == 0)
    assert bool(result.charge == namespace.px)


def test_noether_rejects_a_symmetry_that_is_not_a_spec():
    """Test that a symmetry given as something else than a `SymmetrySpec` raises an error naming it."""
    with pytest.raises(AnalysisError, match='SymmetrySpec'):
        noether(HenonHeilesSystem().prepare().system_, None)
