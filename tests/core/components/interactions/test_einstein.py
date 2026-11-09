"""Test the `_einstein` module of `skcomplex.core.components.interactions`."""

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, diagonal_matrix, function, sin

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import EinsteinInteraction
from skcomplex.core.components.laws.utils import partial
from skcomplex.core.components.spaces import LatticeSpace, PseudoRiemannianSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import SpaceError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec

ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)

t, r, theta, phi, x, y, z = SR.var('t r theta phi x y z')


class UniverseElement(Element):
    """Universe with the scale factor `a` of order 2 and the matter density today `rho`."""

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[float, StateSpec(order=2)] = 1.0,
        a_d1: float = 0.0,
        rho: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.a = a
        self.a_d1 = a_d1
        self.rho = rho


class ProfileElement(Element):
    """Site of a field with the warp `u` of order 2."""

    def __init__(self, label: str, *, u: Annotated[float, StateSpec(order=2)] = 1.0, u_d1: float = 0.0) -> None:
        super().__init__(label)
        self.u = u
        self.u_d1 = u_d1


def _flat_universe() -> PseudoRiemannianSpace:
    """Return the flat Robertson-Walker ansatz of the scale factor `a(t)`."""
    a = function('a')(t)
    return PseudoRiemannianSpace(
        diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2]),
        (t, r, theta, phi),
    )


def _warped_plane() -> PseudoRiemannianSpace:
    """Return the ansatz `ds² = -dt² + dx² + u(t, x)² (dy² + dz²)`, whose `G^y_y = 0` is the wave equation of `u`."""
    u = function('u')(t, x)
    return PseudoRiemannianSpace(diagonal_matrix(SR, [-1, 1, u**2, u**2]), (t, x, y, z))


def test_dust_universe_follows_the_einstein_de_sitter_solution():
    """Test that a flat universe of dust grows as `(3 H t / 2 + 1)^(2/3)` with `H² = 8 π rho / 3`."""
    rho = 3 / (8 * math.pi)
    universe = UniverseElement('universe', a=1.0, a_d1=1.0, rho=rho)
    einstein = EinsteinInteraction(
        'einstein',
        target='universe',
        states='a',
        space=_flat_universe(),
        components=[(1, 1)],
        source=lambda s: diagonal_matrix(SR, [-s.target.rho / s.target.a**3, 0, 0, 0]),
    )
    system = System(elements=[universe], interactions=[einstein])
    trajectory = simulate(system, until=2.0, dt=0.1, solver=ACCURATE)
    expected = (1.5 * np.asarray(trajectory.times) + 1) ** (2 / 3)
    assert np.ravel(trajectory.values('universe.a')) == pytest.approx(expected, rel=1e-8)


def test_warped_plane_reduces_to_the_wave_equation_of_a_field():
    """Test that `G^y_y = 0` of the warped plane gives `u'' = ∂²u/∂x²`, which moves a standing wave on a lattice."""
    einstein = EinsteinInteraction(
        'einstein',
        target='plane',
        states='u',
        space=_warped_plane(),
        components=[(2, 2)],
        axes={1: 0},
    )
    n_sites = 64
    spacing = 2 * math.pi / n_sites
    positions = spacing * np.arange(n_sites)
    plane = FieldElement(
        'plane',
        element=ProfileElement,
        space=LatticeSpace((n_sites,), spacing=spacing, boundary='periodic'),
        values={'u': 2 + 0.1 * np.sin(positions)},
        discretization='spectral',
    )
    system = System(elements=[plane], interactions=[einstein])
    trajectory = simulate(system, until=1.0, dt=0.1, solver=ACCURATE)
    expected = 2 + 0.1 * np.sin(positions) * np.cos(1.0)
    assert np.asarray(trajectory.values('plane.u'))[-1] == pytest.approx(expected, abs=1e-6)


def test_warped_plane_law_reads_the_second_derivative_along_the_lattice_axis():
    """Test that the law of the warped plane holds `partial(partial(u, 0), 0)`."""
    u = SR.var('u')
    namespace = SimpleNamespace(target=SimpleNamespace(u=u), t=SR.var('t'))
    einstein = EinsteinInteraction(
        'einstein',
        target='plane',
        states='u',
        space=_warped_plane(),
        components=[(2, 2)],
        axes={1: 0},
    )
    (law,) = einstein.prepare().run(namespace).laws_
    assert bool(SR(law.value) == partial(partial(u, SR(0)), SR(0)))


@pytest.mark.parametrize(
    ('components', 'axes', 'message'),
    [
        ([(0, 1)], {1: 0}, 'mixes coordinates'),
        ([(2, 2)], {}, 'has no lattice axis'),
        ([(0, 0)], {1: 0}, 'do not fix the second derivatives'),
    ],
)
def test_einstein_interaction_rejects_equations_that_do_not_reduce(components, axes, message):
    """Test that a mixed derivative, a derivative along a coordinate without an axis, or a constraint is rejected."""
    namespace = SimpleNamespace(target=SimpleNamespace(u=SR.var('u')), t=SR.var('t'))
    einstein = EinsteinInteraction(
        'einstein',
        target='plane',
        states='u',
        space=_warped_plane(),
        components=components,
        axes=axes,
    )
    with pytest.raises(ValidationError, match=message):
        einstein.prepare().run(namespace)


def test_einstein_interaction_rejects_coordinates_the_ansatz_keeps():
    """Test that a covariant component that keeps the radius is rejected, unlike its mixed counterpart."""
    namespace = SimpleNamespace(target=SimpleNamespace(a=SR.var('a')), t=SR.var('t'))
    a = function('a')(t)
    space = PseudoRiemannianSpace(
        diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2 + r]),
        (t, r, theta, phi),
    )
    einstein = EinsteinInteraction('einstein', target='universe', states='a', space=space, components=[(1, 1)])
    with pytest.raises(ValidationError, match='keep the coordinates'):
        einstein.prepare().run(namespace)


def test_einstein_interaction_rejects_functions_that_name_nothing():
    """Test that a function of the ansatz that is no state or parameter of the target is rejected."""
    namespace = SimpleNamespace(target=SimpleNamespace(b=SR.var('b')), t=SR.var('t'))
    einstein = EinsteinInteraction(
        'einstein',
        target='universe',
        states='b',
        space=_flat_universe(),
        components=[(1, 1)],
    )
    with pytest.raises(ValidationError, match='time derivative of'):
        einstein.prepare().run(namespace)


@pytest.mark.parametrize(
    ('einstein', 'error', 'message'),
    [
        (EinsteinInteraction('einstein', target='u', states='1a', components=[(1, 1)]), ValidationError, 'names'),
        (EinsteinInteraction('einstein', target='u', states='a', components=[(1, 1), (2, 2)]), ValidationError, 'pair'),
        (
            EinsteinInteraction('einstein', target='u', states='a', components=[(1, 1)], source=1.0),
            ValidationError,
            'source',
        ),
        (EinsteinInteraction('einstein', target='u', states='a', components=[(1, 1)]), SpaceError, 'curvature=True'),
    ],
)
def test_einstein_interaction_rejects_invalid_arguments(einstein, error, message):
    """Test that invalid states, components, sources, or spaces are rejected by `prepare`."""
    with pytest.raises(error, match=message):
        einstein.prepare()


def test_einstein_interaction_rejects_a_space_without_curvature():
    """Test that a space that leaves the curvature out is rejected."""
    space = _flat_universe().set_params(curvature=False)
    einstein = EinsteinInteraction('einstein', target='u', states='a', space=space, components=[(1, 1)])
    with pytest.raises(SpaceError, match='curvature=True'):
        einstein.prepare()
