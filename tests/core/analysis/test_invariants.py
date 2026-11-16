"""Test the `_invariants` module of `skcomplex.core.analysis`."""

from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import SR, sqrt

from skcomplex.core.analysis import invariants
from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import StateSpec
from skcomplex.systems.general.mechanics import EulerLagrangeInteraction
from skcomplex.systems.physics.celestial_mechanics import TwoBodyOrbitSystem
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


def test_run_reports_the_energy_and_angular_momentum_of_kepler_conserved():
    """Test that the energy and angular momentum of the Kepler problem are conserved and a momentum is not."""
    analysis = invariants(TwoBodyOrbitSystem().prepare().system_, expressions=_kepler_expressions())
    assert analysis.conserved == {'energy': True, 'angular_momentum': True, 'px': False}
    assert analysis.methods['energy'] == 'symbolic'
    namespace = analysis.namespace.planet
    expected = -namespace.G * namespace.mass * namespace.x / (namespace.x**2 + namespace.y**2) ** SR(3 / 2)
    assert bool((analysis.rates['px'] - expected).simplify_full() == 0)


def test_run_reports_the_jacobi_integral_of_a_lagrangian_system_conserved():
    """Test that the Jacobi integral of the Euler-Lagrange restricted three-body problem has a zero Lie derivative."""
    expressions = [Observable('jacobi', _jacobi), Observable('x', lambda s: s.body.x)]
    analysis = invariants(_lagrangian_three_body(), expressions=expressions, random_state=0)
    assert analysis.conserved == {'jacobi': True, 'x': False}


def test_run_rejects_an_expression_of_no_system():
    """Test that an expression that is not an observable of the system raises an error."""
    with pytest.raises(AnalysisError, match='not an expression of the system'):
        invariants(HenonHeilesSystem().prepare().system_, expressions=['x'])
