"""Test the `_metric_cosmologies` module of `skcomplex.systems.physics.cosmology`."""

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, diagonal_matrix, function, sin
from scipy.optimize import brentq

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws.utils import partial
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import SpaceError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.systems.physics.common.relativity import SpacetimeSpace
from skcomplex.systems.physics.cosmology import (
    EinsteinFieldEquationsInteraction,
    FLRWCosmologySystem,
    OppenheimerSnyderCollapseSystem,
)

from ..conftest import _run, _series

ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    ('kwargs', 'exact'),
    [
        ({'omega_matter': 1.0, 'omega_lambda': 0.0}, lambda t: (1.5 * t + 0.5**1.5) ** (2 / 3)),
        ({'omega_matter': 0.0, 'omega_radiation': 1.0, 'omega_lambda': 0.0}, lambda t: np.sqrt(2 * t + 0.25)),
        ({'omega_matter': 0.0, 'omega_lambda': 1.0}, lambda t: 0.5 * np.exp(t)),
        ({'omega_matter': 0.0, 'omega_lambda': 0.0}, lambda t: 0.5 + t),
    ],
)
def test_scale_factor_follows_the_analytic_friedmann_solutions(kwargs, exact):
    """Test the scale factor of dust `t^(2/3)`, radiation `t^(1/2)`, vacuum `e^(H t)`, and the empty Milne universe."""
    trajectory = _run(FLRWCosmologySystem(0.5, **kwargs), 2.0, 0.1, ACCURATE)
    (a,) = _series(trajectory, 'universe.a')
    assert np.ravel(a) == pytest.approx(exact(np.asarray(trajectory.times)), rel=1e-8)


def test_mixed_universe_keeps_the_friedmann_constraint_and_accelerates_late():
    """Test that matter, radiation, vacuum, and curvature keep `(a'/a)² = H0² Σ Ω a^-n` and accelerate at late times."""
    trajectory = _run(
        FLRWCosmologySystem(0.3, omega_matter=0.3, omega_radiation=0.01, omega_lambda=0.8),
        3.0,
        0.05,
        ACCURATE,
    )
    a, rate = (np.ravel(item) for item in _series(trajectory, 'universe.a', 'universe.a_d1'))
    curvature = 1 - 0.3 - 0.01 - 0.8
    constraint = rate**2 - (0.3 / a + 0.01 / a**2 + 0.8 * a**2 + curvature)
    assert np.abs(constraint).max() < 1e-8
    assert rate[-1] > rate[len(rate) // 2]


def test_flrw_cosmology_rejects_a_universe_that_cannot_expand():
    """Test that a closed universe beyond its turning point is rejected."""
    with pytest.raises(ValidationError, match='negative squared expansion rate'):
        FLRWCosmologySystem(3.0, omega_matter=3.0, omega_lambda=0.0).prepare()


@pytest.fixture(scope='module')
def collapse():
    """Return the times and radii of a collapsing ball of eight shells."""
    trajectory = _run(OppenheimerSnyderCollapseSystem(8), 34.5, 0.5, ACCURATE)
    (radii,) = _series(trajectory, 'ball.R')
    return np.asarray(trajectory.times), radii


def test_dust_ball_collapses_along_the_oppenheimer_snyder_cycloid(collapse):
    """Test that every shell keeps `R = r a(t)` on the cycloid `a = (1 + cos η) / 2`, `t = (η + sin η) / (2 √k)`."""
    times, radii = collapse
    comoving = 10.0 * np.arange(1, 9) / 8
    k = 2 * 1.0 / 10.0**3
    angles = [brentq(lambda eta, t=t: (eta + math.sin(eta)) / (2 * math.sqrt(k)) - t, 0.0, math.pi) for t in times]
    scale = (1 + np.cos(angles)) / 2
    assert radii / comoving == pytest.approx(np.repeat(scale[:, None], 8, axis=1), rel=1e-7)


def test_dust_ball_surface_crosses_its_schwarzschild_radius(collapse):
    """Test that the surface falls inside `2 M` at the time the cycloid predicts, before the ball reaches zero size."""
    times, radii = collapse
    k = 2 * 1.0 / 10.0**3
    eta = math.acos(2 * 2.0 / 10.0 - 1)
    crossing = (eta + math.sin(eta)) / (2 * math.sqrt(k))
    surface = radii[:, -1]
    index = int(np.flatnonzero(surface < 2.0)[0])
    assert times[index - 1] < crossing <= times[index]
    assert times[-1] < math.pi / (2 * math.sqrt(k))


def test_oppenheimer_snyder_collapse_rejects_a_ball_inside_its_horizon():
    """Test that a ball that starts inside its Schwarzschild radius is rejected."""
    with pytest.raises(ValidationError, match='inside the Schwarzschild radius'):
        OppenheimerSnyderCollapseSystem(radius=1.5).prepare()


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


def _flat_universe() -> SpacetimeSpace:
    """Return the flat Robertson-Walker ansatz of the scale factor `a(t)`."""
    a = function('a')(t)
    return SpacetimeSpace(
        diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2]),
        (t, r, theta, phi),
    )


def _warped_plane() -> SpacetimeSpace:
    """Return the ansatz `ds² = -dt² + dx² + u(t, x)² (dy² + dz²)`, whose `G^y_y = 0` is the wave equation of `u`."""
    u = function('u')(t, x)
    return SpacetimeSpace(diagonal_matrix(SR, [-1, 1, u**2, u**2]), (t, x, y, z))


def test_dust_universe_follows_the_einstein_de_sitter_solution():
    """Test that a flat universe of dust grows as `(3 H t / 2 + 1)^(2/3)` with `H² = 8 π rho / 3`."""
    rho = 3 / (8 * math.pi)
    universe = UniverseElement('universe', a=1.0, a_d1=1.0, rho=rho)
    einstein = EinsteinFieldEquationsInteraction(
        'einstein',
        target='universe',
        states='a',
        space=_flat_universe(),
        components=[(1, 1)],
        source=lambda s: diagonal_matrix(SR, [-s.target.rho / s.target.a**3, 0, 0, 0]),
    )
    system = CompositeSystem(elements=[universe], interactions=[einstein])
    trajectory = simulate(system, until=2.0, dt=0.1, solver=ACCURATE)
    expected = (1.5 * np.asarray(trajectory.times) + 1) ** (2 / 3)
    assert np.ravel(trajectory.values('universe.a')) == pytest.approx(expected, rel=1e-8)


def test_warped_plane_reduces_to_the_wave_equation_of_a_field():
    """Test that `G^y_y = 0` of the warped plane gives `u'' = ∂²u/∂x²`, which moves a standing wave on a lattice."""
    einstein = EinsteinFieldEquationsInteraction(
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
    system = CompositeSystem(elements=[plane], interactions=[einstein])
    trajectory = simulate(system, until=1.0, dt=0.1, solver=ACCURATE)
    expected = 2 + 0.1 * np.sin(positions) * np.cos(1.0)
    assert np.asarray(trajectory.values('plane.u'))[-1] == pytest.approx(expected, abs=1e-6)


def test_warped_plane_law_reads_the_second_derivative_along_the_lattice_axis():
    """Test that the law of the warped plane holds `partial(partial(u, 0), 0)`."""
    u = SR.var('u')
    namespace = SimpleNamespace(target=SimpleNamespace(u=u), t=SR.var('t'))
    einstein = EinsteinFieldEquationsInteraction(
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
    einstein = EinsteinFieldEquationsInteraction(
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
    space = SpacetimeSpace(
        diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2 + r]),
        (t, r, theta, phi),
    )
    einstein = EinsteinFieldEquationsInteraction(
        'einstein',
        target='universe',
        states='a',
        space=space,
        components=[(1, 1)],
    )
    with pytest.raises(ValidationError, match='keep the coordinates'):
        einstein.prepare().run(namespace)


def test_einstein_interaction_rejects_functions_that_name_nothing():
    """Test that a function of the ansatz that is no state or parameter of the target is rejected."""
    namespace = SimpleNamespace(target=SimpleNamespace(b=SR.var('b')), t=SR.var('t'))
    einstein = EinsteinFieldEquationsInteraction(
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
        (
            EinsteinFieldEquationsInteraction('einstein', target='u', states='1a', components=[(1, 1)]),
            ValidationError,
            'names',
        ),
        (
            EinsteinFieldEquationsInteraction('einstein', target='u', states='a', components=[(1, 1), (2, 2)]),
            ValidationError,
            'pair',
        ),
        (
            EinsteinFieldEquationsInteraction('einstein', target='u', states='a', components=[(1, 1)], source=1.0),
            ValidationError,
            'source',
        ),
        (
            EinsteinFieldEquationsInteraction('einstein', target='u', states='a', components=[(1, 1)]),
            SpaceError,
            'curvature=True',
        ),
    ],
)
def test_einstein_interaction_rejects_invalid_arguments(einstein, error, message):
    """Test that invalid states, components, sources, or spaces are rejected by `prepare`."""
    with pytest.raises(error, match=message):
        einstein.prepare()


def test_einstein_interaction_rejects_a_space_without_curvature():
    """Test that a space that leaves the curvature out is rejected."""
    space = _flat_universe().set_params(curvature=False)
    einstein = EinsteinFieldEquationsInteraction('einstein', target='u', states='a', space=space, components=[(1, 1)])
    with pytest.raises(SpaceError, match='curvature=True'):
        einstein.prepare()
