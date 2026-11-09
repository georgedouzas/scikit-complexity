"""Test the `_cells` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, sin, var

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import (
    cell_operations_in,
    codifferential,
    exterior_derivative,
    hodge_star,
    neighbors,
)
from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood, LatticeSpace, NetworkSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver, VelocityVerletSolver
from skcomplex.core.specs import ParameterSpec, StateSpec

BACKENDS = [NumpyBackend, NumbaBackend]
GAUSS_LIMIT = 1e-10
DISPERSION_LIMIT = 0.01


class SiteElement(Element):
    """Site with a value `u` and a value `v`."""

    def __init__(
        self,
        label,
        visual=None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v


class LinkElement(Element):
    """Link with values `a`, `b`, and `c`, and a weight."""

    def __init__(
        self,
        label,
        visual=None,
        *,
        a: Annotated[float, StateSpec()] = 0.0,
        b: Annotated[float, StateSpec()] = 0.0,
        c: Annotated[float, StateSpec()] = 0.0,
        weight: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b
        self.c = c
        self.weight = weight


class FaceElement(Element):
    """Face with values `f` and `g`."""

    def __init__(
        self,
        label,
        visual=None,
        *,
        f: Annotated[float, StateSpec()] = 0.0,
        g: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.f = f
        self.g = g


class FieldLinkElement(Element):
    """Link with a potential `a` of second order, whose velocity is minus the electric field."""

    def __init__(self, label, visual=None, *, a: Annotated[float, StateSpec(order=2)] = 0.0, a_d1=0.0):
        super().__init__(label, visual=visual)
        self.a = a
        self.a_d1 = a_d1


class PairLinkElement(Element):
    """Link with a potential `a` and an electric field `e`."""

    def __init__(
        self,
        label,
        visual=None,
        *,
        a: Annotated[float, StateSpec()] = 0.0,
        e: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.a = a
        self.e = e


def _groups(space, sites=None, links=None, faces=None):
    """Return groups on the sites, the links, and the faces of a prepared complex with given values."""
    return [
        GroupElement('sites', element=SiteElement, space=space, on='sites', values=sites or {}),
        GroupElement('links', element=LinkElement, space=space, on='edges', values=links or {}),
        GroupElement('faces', element=FaceElement, space=space, on='hyperedges', values=faces or {}),
    ]


def _rates(system, backend, name):
    """Return the rates of a dotted state of a system at its initial state."""
    problem = PassCompiler(backend=backend()).prepare(system).run().problem_
    rates = np.asarray(problem.drift(0.0, problem.initial_state, problem.initial_parameters))
    entry = next(item for item in problem.states.entries if item.name == name)
    return rates[entry.start : entry.start + entry.size]


def _cube():
    """Return the prepared complex of a periodic 3D lattice."""
    return CellComplexSpace(LatticeSpace((4, 3, 5))).prepare()


def test_operations_are_held_symbolically():
    """Test that the operations stay unevaluated and that the outermost calls are found once each."""
    flow = var('flow')
    assert str(hodge_star(flow)) == 'hodge_star(flow)'
    expression = codifferential(exterior_derivative(flow)) + sin(exterior_derivative(flow))
    assert {str(item) for item in cell_operations_in(expression)} == {
        'codifferential(exterior_derivative(flow))',
        'exterior_derivative(flow)',
    }


@pytest.mark.parametrize('backend', BACKENDS)
def test_curl_of_a_gradient_is_exactly_zero(backend):
    """Test that the exterior derivative of the exterior derivative of a random site state is exactly 0."""
    space = _cube()
    values = {'u': np.random.default_rng(0).normal(size=space.size_)}
    rate = RateInteraction(
        'curl',
        target='faces',
        state='f',
        sources=['sites'],
        rate=lambda n: exterior_derivative(exterior_derivative(n.sources[0].u)),
    )
    system = System(elements=_groups(space, sites=values), interactions=[rate])
    rates = _rates(system, backend, 'faces.f')
    assert rates.size == len(space.hyperedges_)
    assert np.all(rates == 0.0)


@pytest.mark.parametrize('backend', BACKENDS)
def test_divergence_of_a_curl_is_zero(backend):
    """Test that the codifferential of the codifferential of a random plaquette state is 0 within 1e-12."""
    space = _cube()
    values = {'f': np.random.default_rng(1).normal(size=len(space.hyperedges_))}
    rate = RateInteraction(
        'divergence',
        target='sites',
        state='u',
        sources=['faces'],
        rate=lambda n: codifferential(codifferential(n.sources[0].f)),
    )
    rates = _rates(System(elements=_groups(space, faces=values), interactions=[rate]), backend, 'sites.u')
    assert rates.size == space.size_
    assert np.abs(rates).max() < 1e-12


@pytest.mark.parametrize('backend', BACKENDS)
def test_operations_are_the_incidence_products(backend):
    """Test the values of every operation against the products of the matrices of the complex."""
    space = CellComplexSpace(LatticeSpace((4, 5))).prepare()
    rng = np.random.default_rng(2)
    links = {'a': rng.normal(size=40)}
    sites = {'u': rng.normal(size=20)}
    faces = {'f': rng.normal(size=20)}
    interactions = [
        RateInteraction(
            'd0',
            target='links',
            state='a',
            sources=['sites'],
            rate=lambda n: exterior_derivative(n.sources[0].u),
        ),
        RateInteraction(
            'd1',
            target='faces',
            state='f',
            sources=['links'],
            rate=lambda n: exterior_derivative(n.sources[0].a),
        ),
        RateInteraction(
            'c1',
            target='sites',
            state='u',
            sources=['links'],
            rate=lambda n: codifferential(n.sources[0].a),
        ),
        RateInteraction(
            'c2',
            target='links',
            state='b',
            sources=['faces'],
            rate=lambda n: codifferential(n.sources[0].f),
        ),
        RateInteraction('star', target='links', state='c', rate=lambda n: hodge_star(n.target.a)),
    ]
    system = System(elements=_groups(space, sites, links, faces), interactions=interactions)
    d0, d1 = space.incidences_
    assert np.allclose(_rates(system, backend, 'links.a'), d0 @ sites['u'])
    assert np.allclose(_rates(system, backend, 'faces.f'), d1 @ links['a'])
    assert np.allclose(_rates(system, backend, 'sites.u'), -d0.T @ links['a'])
    assert np.allclose(_rates(system, backend, 'links.b'), d1.T @ faces['f'])
    assert np.allclose(_rates(system, backend, 'links.c'), links['a'])


@pytest.mark.parametrize('backend', BACKENDS)
def test_codifferential_reads_the_spacings_of_a_lattice(backend):
    """Test the Hodge star of a lattice of spacing 2 inside the codifferential of a link state."""
    space = CellComplexSpace(LatticeSpace((4, 5), spacing=2.0)).prepare()
    links = {'a': np.random.default_rng(3).normal(size=40)}
    rate = RateInteraction(
        'c1',
        target='sites',
        state='u',
        sources=['links'],
        rate=lambda n: codifferential(n.sources[0].a),
    )
    rates = _rates(System(elements=_groups(space, links=links), interactions=[rate]), backend, 'sites.u')
    assert np.allclose(rates, -(space.incidences_[0].T @ links['a']) / 4.0)


@pytest.mark.parametrize('backend', BACKENDS)
def test_codifferential_reads_the_weights_of_the_groups_of_a_network(backend):
    """Test the Hodge star of a network complex from the parameter `weight` of its edge group."""
    graph = nx.cycle_graph(4)
    space = CellComplexSpace(NetworkSpace(graph), faces=[(0, 1, 2, 3)]).prepare()
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    links = {'a': np.array([1.0, -1.0, 2.0, 0.5]), 'weight': weights}
    rate = RateInteraction(
        'c1',
        target='sites',
        state='u',
        sources=['links'],
        rate=lambda n: codifferential(n.sources[0].a),
    )
    rates = _rates(System(elements=_groups(space, links=links), interactions=[rate]), backend, 'sites.u')
    assert np.allclose(rates, -space.incidences_[0].T @ (weights * links['a']))


@pytest.mark.parametrize('backend', BACKENDS)
def test_operations_compose_with_functions_of_their_values(backend):
    """Test the codifferential of the sine of an exterior derivative against the matrices of the complex."""
    space = CellComplexSpace(LatticeSpace((4, 5))).prepare()
    links = {'a': np.random.default_rng(4).normal(size=40)}
    rate = RateInteraction(
        'c',
        target='links',
        state='b',
        rate=lambda n: codifferential(sin(exterior_derivative(n.target.a))),
    )
    rates = _rates(System(elements=_groups(space, links=links), interactions=[rate]), backend, 'links.b')
    d1 = space.incidences_[1]
    assert np.allclose(rates, d1.T @ np.sin(d1 @ links['a']))


@pytest.mark.parametrize('backend', BACKENDS)
def test_coface_reduction_reads_the_oriented_sum_of_the_one_face(backend):
    """Test the sum of the cosines of the plaquette sums of the faces of every link."""
    space = CellComplexSpace(LatticeSpace((4, 5))).prepare()
    links = {'a': np.random.default_rng(5).uniform(-3, 3, size=40)}
    rate = RateInteraction(
        'c',
        target='links',
        state='c',
        rate=lambda n: neighbors(cos(exterior_derivative(n.target.a)), over=CofaceNeighborhood()),
    )
    rates = _rates(System(elements=_groups(space, links=links), interactions=[rate]), backend, 'links.c')
    d1 = space.incidences_[1]
    assert np.allclose(rates, abs(d1).T @ np.cos(d1 @ links['a']))


@pytest.mark.parametrize('backend', BACKENDS)
def test_abelian_plaquette_is_the_exterior_derivative_at_the_faces(backend):
    """Test that a coface reduction of the plaquette sums equals the exterior derivative at the same plaquettes."""
    space = CellComplexSpace(LatticeSpace((4, 5), boundary='fixed')).prepare()
    size = int(space.edges_[None][0].size)
    links = {'a': np.random.default_rng(6).uniform(-3, 3, size=size)}
    interactions = [
        RateInteraction(
            'sum',
            target='links',
            state='c',
            rate=lambda n: neighbors(exterior_derivative(n.target.a), over=CofaceNeighborhood()),
        ),
        RateInteraction(
            'plaquette',
            target='faces',
            state='f',
            sources=['links'],
            rate=lambda n: exterior_derivative(n.sources[0].a),
        ),
    ]
    system = System(elements=_groups(space, links=links), interactions=interactions)
    around = _rates(system, backend, 'links.c')
    plaquettes = _rates(system, backend, 'faces.f')
    assert np.allclose(around, abs(space.incidences_[1]).T @ plaquettes)
    single = np.flatnonzero(np.diff(abs(space.incidences_[1]).T.tocsr().indptr) == 1)
    faces = abs(space.incidences_[1]).T.tocsr()[single].indices
    assert np.array_equal(around[single], plaquettes[faces])


def test_coface_reduction_of_a_site_runs_over_its_edges():
    """Test that a reduction over the cofaces of a site reads the difference along each of its edges."""
    space = CellComplexSpace(NetworkSpace(nx.path_graph(3))).prepare()
    sites = {'u': np.array([1.0, 3.0, 7.0])}
    rate = RateInteraction(
        'sum',
        target='sites',
        state='v',
        rate=lambda n: neighbors(exterior_derivative(n.target.u), over=CofaceNeighborhood()),
    )
    groups = [GroupElement('sites', element=SiteElement, space=space, on='sites', values=sites)]
    rates = _rates(System(elements=groups, interactions=[rate]), NumpyBackend, 'sites.v')
    assert rates.tolist() == [2.0, 6.0, 4.0]


def _maxwell(size, charges):
    """Return the links of a lattice with a potential and an electric field from static charges, and its charges."""
    space = CellComplexSpace(LatticeSpace((size, size))).prepare()
    density = np.zeros(size * size)
    for site, charge in charges.items():
        density[site] = charge
    field = np.zeros(2 * size * size)
    if charges:
        solution, *_ = np.linalg.lstsq(-space.incidences_[0].T.toarray(), density, rcond=None)
        field = solution
    return space, field, density


def test_gauss_law_holds_for_rate_laws_of_a_first_order_pair():
    """Test the Gauss law with RK4 on the first-order form `A_t = -E`, `E_t = codifferential(dA)`."""
    space, field, density = _maxwell(8, {0: 1.0, 27: -1.0})
    potential = np.random.default_rng(7).normal(size=field.size)
    values = {'a': potential, 'e': field}
    links = GroupElement('links', element=PairLinkElement, space=space, on='edges', values=values)
    interactions = [
        RateInteraction('potential', target='links', state='a', rate=lambda n: -n.target.e),
        RateInteraction(
            'field',
            target='links',
            state='e',
            rate=lambda n: codifferential(exterior_derivative(n.target.a)),
        ),
    ]
    system = System(elements=[links], interactions=interactions)
    trajectory = simulate(system, until=2.0, solver=RK4Solver(dt=0.02))
    residual = -space.incidences_[0].T @ trajectory.values('links.e').T - density[:, None]
    assert np.abs(residual).max() < GAUSS_LIMIT


def test_gauss_law_holds_for_velocity_verlet():
    """Test the Gauss law of 1000 steps of velocity Verlet on a periodic 32 by 32 lattice with a static charge pair."""
    space, field, density = _maxwell(32, {0: 1.0, 500: -1.0})
    values = {'a_d1': -field}
    links = GroupElement('links', element=FieldLinkElement, space=space, on='edges', values=values)
    rate = RateInteraction(
        'field',
        target='links',
        state='a',
        rate=lambda n: -codifferential(exterior_derivative(n.target.a)),
    )
    system = System(elements=[links], interactions=[rate])
    trajectory = simulate(system, until=50.0, solver=VelocityVerletSolver(dt=0.05))
    electric = -trajectory.values('links.a_d1')
    residual = -space.incidences_[0].T @ electric.T - density[:, None]
    assert np.abs(residual).max() < GAUSS_LIMIT


@pytest.mark.parametrize('spacing', [1.0, 0.5])
def test_lattice_dispersion_follows_the_lattice_laplacian(spacing):
    """Test the angular frequency of a plane wave of wave number k against `(2 / a) sin(k a / 2)`."""
    size, mode = 32, 3
    space = CellComplexSpace(LatticeSpace((size, size), spacing=spacing)).prepare()
    wave_number = 2 * np.pi * mode / (size * spacing)
    sources, _ = space.edges_[None]
    along_second = np.arange(sources.size) >= size * size
    positions = np.unravel_index(sources, (size, size))[0] * spacing
    amplitude = np.where(along_second, np.cos(wave_number * positions), 0.0)
    links = GroupElement('links', element=FieldLinkElement, space=space, on='edges', values={'a': amplitude})
    rate = RateInteraction(
        'field',
        target='links',
        state='a',
        rate=lambda n: -codifferential(exterior_derivative(n.target.a)),
    )
    system = System(elements=[links], interactions=[rate])
    expected = (2 / spacing) * np.sin(wave_number * spacing / 2)
    period = 2 * np.pi / expected
    until = 4 * period
    step = period / 200
    trajectory = simulate(system, until=until, solver=VelocityVerletSolver(dt=step / 4), dt=step)
    series = trajectory.values('links.a')[:, size * size]
    crossings = np.flatnonzero(np.diff(np.sign(series)) != 0)
    measured = np.pi * (crossings.size - 1) / (trajectory.times[crossings[-1]] - trajectory.times[crossings[0]])
    assert abs(measured - expected) / expected < DISPERSION_LIMIT
