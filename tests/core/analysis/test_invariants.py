"""Test the `_invariants` module of `skcomplex.core.analysis`."""

from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import SR, sqrt

from skcomplex.core.analysis import InvariantAnalysis
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import EulerLagrangeInteraction
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.system import System
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import ObservableSpec, StateSpec, SymmetrySpec
from skcomplex.systems.physics.celestial_mechanics import sitnikov, two_body_orbit
from skcomplex.systems.physics.nonlinear_oscillators import henon_heiles

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


def _lagrangian_three_body() -> System:
    """Return the restricted three-body problem written with an Euler-Lagrange interaction."""
    motion = EulerLagrangeInteraction('motion', target='body', states=('x', 'y'), lagrangian=_rotating_lagrangian)
    return System(elements=[LagrangianBodyElement('body')], interactions=[motion])


def _kepler_expressions() -> list[ObservableSpec]:
    """Return the energy, the angular momentum, and the first momentum of the Kepler problem."""
    return [
        ObservableSpec(
            'energy',
            lambda s: (s.planet.px**2 + s.planet.py**2) / 2
            - s.planet.G
            * s.planet.mass
            / sqrt(
                s.planet.x**2 + s.planet.y**2,
            ),
        ),
        ObservableSpec('angular_momentum', lambda s: s.planet.x * s.planet.py - s.planet.y * s.planet.px),
        ObservableSpec('px', lambda s: s.planet.px),
    ]


def test_run_reports_the_energy_and_angular_momentum_of_kepler_conserved():
    """Test that the energy and angular momentum of the Kepler problem are conserved and a momentum is not."""
    analysis = InvariantAnalysis(expressions=_kepler_expressions()).prepare(two_body_orbit()).run()
    assert analysis.conserved_ == {'energy': True, 'angular_momentum': True, 'px': False}
    assert analysis.methods_['energy'] == 'symbolic'
    namespace = analysis.namespace_.planet
    expected = -namespace.G * namespace.mass * namespace.x / (namespace.x**2 + namespace.y**2) ** SR(3 / 2)
    assert bool((analysis.rates_['px'] - expected).simplify_full() == 0)


def test_run_reports_the_jacobi_integral_of_a_lagrangian_system_conserved():
    """Test that the Jacobi integral of the Euler-Lagrange restricted three-body problem has a zero Lie derivative."""
    expressions = [ObservableSpec('jacobi', _jacobi), ObservableSpec('x', lambda s: s.body.x)]
    analysis = InvariantAnalysis(expressions=expressions, random_state=0).prepare(_lagrangian_three_body())
    analysis.run()
    assert analysis.conserved_ == {'jacobi': True, 'x': False}


def test_run_brackets_the_components_of_the_angular_momentum():
    """Test that the bracket of the coordinates with their momenta is one and of the momentum with x is y-free."""
    expressions = [
        ObservableSpec('x', lambda s: s.planet.x),
        ObservableSpec('px', lambda s: s.planet.px),
        ObservableSpec('angular_momentum', lambda s: s.planet.x * s.planet.py - s.planet.y * s.planet.px),
    ]
    analysis = InvariantAnalysis('brackets', expressions=expressions).prepare(two_body_orbit()).run()
    namespace = analysis.namespace_.planet
    assert analysis.brackets_[('x', 'px')] == 1
    assert bool(analysis.brackets_[('x', 'angular_momentum')] == -namespace.y)


def test_run_rejects_brackets_without_canonical_pairs():
    """Test that brackets of a system without a Hamiltonian interaction raise an error naming it."""
    analysis = InvariantAnalysis('brackets', expressions=[ObservableSpec('z', lambda s: s.body.z)])
    with pytest.raises(AnalysisError, match='HamiltonianInteraction'):
        analysis.prepare(sitnikov()).run()


def test_run_confirms_the_rotation_invariance_of_kepler_with_its_charge():
    """Test that the rotation of the plane keeps the Kepler Hamiltonian, with the angular momentum as charge."""
    rotation = SymmetrySpec('rotation', lambda s: {'planet.x': -s.planet.y, 'planet.y': s.planet.x})
    analysis = InvariantAnalysis('symmetry', symmetry=rotation).prepare(two_body_orbit()).run()
    namespace = analysis.namespace_.planet
    assert analysis.invariant_
    assert analysis.remainder_ == 0
    assert bool((analysis.charge_ - (namespace.x * namespace.py - namespace.y * namespace.px)).simplify_full() == 0)


def test_run_reports_the_rotation_charge_of_a_lagrangian():
    """Test that the rotating-frame Lagrangian is not invariant under rotation, since the primaries break it."""
    rotation = SymmetrySpec('rotation', lambda s: {'body.x': -s.body.y, 'body.y': s.body.x})
    analysis = InvariantAnalysis('symmetry', symmetry=rotation).prepare(_lagrangian_three_body()).run()
    namespace = analysis.namespace_.body
    assert not analysis.invariant_
    expected = namespace.x * namespace.y_d1 - namespace.y * namespace.x_d1 + namespace.x**2 + namespace.y**2
    assert bool((analysis.charge_ - expected).simplify_full() == 0)


def test_run_reports_a_broken_translation_of_henon_heiles():
    """Test that a translation of the first coordinate changes the Hénon-Heiles Hamiltonian by `x + 2 x y`."""
    translation = SymmetrySpec('translation', lambda s: {'star.x': 1})
    analysis = InvariantAnalysis('symmetry', symmetry=translation).prepare(henon_heiles()).run()
    namespace = analysis.namespace_.star
    assert not analysis.invariant_
    assert bool((analysis.remainder_ - (namespace.x + 2 * namespace.x * namespace.y)).simplify_full() == 0)
    assert bool(analysis.charge_ == namespace.px)


def test_check_rejects_a_symmetry_kind_without_a_symmetry():
    """Test that the symmetry kind needs a `SymmetrySpec`."""
    with pytest.raises(AnalysisError, match='SymmetrySpec'):
        InvariantAnalysis('symmetry').prepare()
