"""Test the `_cells` module of `skcomplex.core.components.laws.utils`."""

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, sin, var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import codifferential, exterior_derivative, hodge_star, neighbors
from skcomplex.core.components.laws.utils._cells import cell_operations_in
from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood, LatticeSpace, NetworkSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.solvers import RK4Solver, VelocityVerletSolver
from skcomplex.core.systems import System

BACKENDS = [NumpyBackend, NumbaBackend]
GAUSS_LIMIT = 1e-10
DISPERSION_LIMIT = 0.01
GAP = (
    'Engine gap: only a variadic sources role reads a group on a cell complex member by member through the exterior '
    'calculus, so a cell operation on a second role of another degree fails the check of groups of one size'
)


class SiteElement(Element):
    """Site with a value `u` and a value `v`."""

    def __init__(self, label, *, u=0.0, v=0.0, space=None, on=None):
        super().__init__(label, space=space, on=on)
        self.u = u
        self.v = v


class LinkElement(Element):
    """Link with values `a`, `b`, and `c`, and a weight."""

    def __init__(self, label, *, a=0.0, b=0.0, c=0.0, weight=1.0, space=None, on=None):
        super().__init__(label, space=space, on=on)
        self.a = a
        self.b = b
        self.c = c
        self.weight = weight


class FaceElement(Element):
    """Face with values `f` and `g`."""

    def __init__(self, label, *, f=0.0, g=0.0, space=None, on=None):
        super().__init__(label, space=space, on=on)
        self.f = f
        self.g = g


class PairLinkElement(Element):
    """Link with a potential `a` and an electric field `e`."""

    def __init__(self, label, *, a=0.0, e=0.0, space=None, on=None):
        super().__init__(label, space=space, on=on)
        self.a = a
        self.e = e


def _groups(space, sites=None, links=None, faces=None):
    """Return groups on the sites, the links, and the faces of a prepared complex with given values."""
    return [
        SiteElement('sites', space=space, **(sites or {})),
        LinkElement('links', space=space, on='edges', **(links or {})),
        FaceElement('faces', space=space, on='hyperedges', **(faces or {})),
    ]


def _rate(label, target, state, rate, source=None):
    """Return a one-off interaction that adds a rate, a function of the namespace, to a state of its target."""
    roles = {'target': target} if source is None else {'target': target, 'source': source}
    return Interaction(label, roles=roles, laws=lambda n: [RateLaw(getattr(n.target, state), rate(n))])


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


@pytest.mark.skip(reason=GAP)
@pytest.mark.parametrize('backend', BACKENDS)
def test_curl_of_a_gradient_is_exactly_zero(backend):
    """Test that the exterior derivative of the exterior derivative of a random site state is exactly 0."""
    space = _cube()
    values = {'u': np.random.default_rng(0).normal(size=space.size_)}
    rate = _rate(
        'curl',
        'faces',
        'f',
        lambda n: exterior_derivative(exterior_derivative(n.source.u), source='sites'),
    )
    system = System(elements=_groups(space, sites=values), interactions=[rate])
    rates = _rates(system, backend, 'faces.f')
    assert rates.size == len(space.hyperedges_)
    assert np.all(rates == 0.0)


@pytest.mark.skip(reason=GAP)
@pytest.mark.parametrize('backend', BACKENDS)
def test_divergence_of_a_curl_is_zero(backend):
    """Test that the codifferential of the codifferential of a random plaquette state is 0 within 1e-12."""
    space = _cube()
    values = {'f': np.random.default_rng(1).normal(size=len(space.hyperedges_))}
    rate = _rate(
        'divergence',
        'sites',
        'u',
        lambda n: codifferential(codifferential(n.source.f), source='faces'),
    )
    rates = _rates(System(elements=_groups(space, faces=values), interactions=[rate]), backend, 'sites.u')
    assert rates.size == space.size_
    assert np.abs(rates).max() < 1e-12


@pytest.mark.skip(reason=GAP)
@pytest.mark.parametrize('backend', BACKENDS)
def test_operations_are_the_incidence_products(backend):
    """Test the values of every operation against the products of the matrices of the complex."""
    space = CellComplexSpace(LatticeSpace((4, 5))).prepare()
    rng = np.random.default_rng(2)
    links = {'a': rng.normal(size=40)}
    sites = {'u': rng.normal(size=20)}
    faces = {'f': rng.normal(size=20)}
    interactions = [
        _rate(
            'd0',
            'links',
            'a',
            lambda n: exterior_derivative(n.source.u, source='sites'),
        ),
        _rate(
            'd1',
            'faces',
            'f',
            lambda n: exterior_derivative(n.source.a, source='links'),
        ),
        _rate(
            'c1',
            'sites',
            'u',
            lambda n: codifferential(n.source.a, source='links'),
        ),
        _rate(
            'c2',
            'links',
            'b',
            lambda n: codifferential(n.source.f, source='faces'),
        ),
        _rate('star', 'links', 'c', lambda n: hodge_star(n.target.a)),
    ]
    system = System(elements=_groups(space, sites, links, faces), interactions=interactions)
    d0, d1 = space.incidences_
    assert np.allclose(_rates(system, backend, 'links.a'), d0 @ sites['u'])
    assert np.allclose(_rates(system, backend, 'faces.f'), d1 @ links['a'])
    assert np.allclose(_rates(system, backend, 'sites.u'), -d0.T @ links['a'])
    assert np.allclose(_rates(system, backend, 'links.b'), d1.T @ faces['f'])
    assert np.allclose(_rates(system, backend, 'links.c'), links['a'])


@pytest.mark.skip(reason=GAP)
@pytest.mark.parametrize('backend', BACKENDS)
def test_codifferential_reads_the_spacings_of_a_lattice(backend):
    """Test the Hodge star of a lattice of spacing 2 inside the codifferential of a link state."""
    space = CellComplexSpace(LatticeSpace((4, 5), spacing=2.0)).prepare()
    links = {'a': np.random.default_rng(3).normal(size=40)}
    rate = _rate(
        'c1',
        'sites',
        'u',
        lambda n: codifferential(n.source.a, source='links'),
    )
    rates = _rates(System(elements=_groups(space, links=links), interactions=[rate]), backend, 'sites.u')
    assert np.allclose(rates, -(space.incidences_[0].T @ links['a']) / 4.0)


@pytest.mark.skip(reason=GAP)
@pytest.mark.parametrize('backend', BACKENDS)
def test_codifferential_reads_the_weights_of_the_groups_of_a_network(backend):
    """Test the Hodge star of a network complex from the parameter `weight` of its edge group."""
    graph = nx.cycle_graph(4)
    space = CellComplexSpace(NetworkSpace(graph), faces=[(0, 1, 2, 3)]).prepare()
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    links = {'a': np.array([1.0, -1.0, 2.0, 0.5]), 'weight': weights}
    rate = _rate(
        'c1',
        'sites',
        'u',
        lambda n: codifferential(n.source.a, source='links'),
    )
    rates = _rates(System(elements=_groups(space, links=links), interactions=[rate]), backend, 'sites.u')
    assert np.allclose(rates, -space.incidences_[0].T @ (weights * links['a']))


@pytest.mark.parametrize('backend', BACKENDS)
def test_operations_compose_with_functions_of_their_values(backend):
    """Test the codifferential of the sine of an exterior derivative against the matrices of the complex."""
    space = CellComplexSpace(LatticeSpace((4, 5))).prepare()
    links = {'a': np.random.default_rng(4).normal(size=40)}
    rate = _rate(
        'c',
        'links',
        'b',
        lambda n: codifferential(sin(exterior_derivative(n.target.a))),
    )
    rates = _rates(System(elements=_groups(space, links=links), interactions=[rate]), backend, 'links.b')
    d1 = space.incidences_[1]
    assert np.allclose(rates, d1.T @ np.sin(d1 @ links['a']))


@pytest.mark.parametrize('backend', BACKENDS)
def test_coface_reduction_reads_the_oriented_sum_of_the_one_face(backend):
    """Test the sum of the cosines of the plaquette sums of the faces of every link."""
    space = CellComplexSpace(LatticeSpace((4, 5))).prepare()
    links = {'a': np.random.default_rng(5).uniform(-3, 3, size=40)}
    rate = _rate(
        'c',
        'links',
        'c',
        lambda n: neighbors(cos(exterior_derivative(n.target.a)), over=CofaceNeighborhood()),
    )
    rates = _rates(System(elements=_groups(space, links=links), interactions=[rate]), backend, 'links.c')
    d1 = space.incidences_[1]
    assert np.allclose(rates, abs(d1).T @ np.cos(d1 @ links['a']))


@pytest.mark.skip(reason=GAP)
@pytest.mark.parametrize('backend', BACKENDS)
def test_abelian_plaquette_is_the_exterior_derivative_at_the_faces(backend):
    """Test that a coface reduction of the plaquette sums equals the exterior derivative at the same plaquettes."""
    space = CellComplexSpace(LatticeSpace((4, 5), boundary='fixed')).prepare()
    size = int(space.edges_[None][0].size)
    links = {'a': np.random.default_rng(6).uniform(-3, 3, size=size)}
    interactions = [
        _rate(
            'sum',
            'links',
            'c',
            lambda n: neighbors(exterior_derivative(n.target.a), over=CofaceNeighborhood()),
        ),
        _rate(
            'plaquette',
            'faces',
            'f',
            lambda n: exterior_derivative(n.source.a, source='links'),
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
    rate = _rate(
        'sum',
        'sites',
        'v',
        lambda n: neighbors(exterior_derivative(n.target.u), over=CofaceNeighborhood()),
    )
    groups = [SiteElement('sites', space=space, **sites)]
    rates = _rates(System(elements=groups, interactions=[rate]), NumpyBackend, 'sites.v')
    assert rates.tolist() == [2.0, 6.0, 4.0]


def _wave(n):
    """Return the second-order rate of a link potential, minus the codifferential of its exterior derivative."""
    a = n.target.a
    return [RateLaw(a.diff(n.t, 2), -codifferential(exterior_derivative(a)))]


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
    links = PairLinkElement('links', space=space, on='edges', **values)
    interactions = [
        _rate('potential', 'links', 'a', lambda n: -n.target.e),
        _rate(
            'field',
            'links',
            'e',
            lambda n: codifferential(exterior_derivative(n.target.a)),
        ),
    ]
    system = System(elements=[links], interactions=interactions, solver=RK4Solver(dt=0.02))
    trajectory = system.prepare().simulate(until=2.0)
    residual = -space.incidences_[0].T @ trajectory.get_values('links.e').T - density[:, None]
    assert np.abs(residual).max() < GAUSS_LIMIT


def test_gauss_law_holds_for_velocity_verlet():
    """Test the Gauss law of 1000 steps of velocity Verlet on a periodic 32 by 32 lattice with a static charge pair."""
    space, field, density = _maxwell(32, {0: 1.0, 500: -1.0})
    links = Element('links', values={'a': 0.0, "a'": -field}, space=space, on='edges')
    rate = Interaction('field', roles={'target': 'links'}, laws=_wave)
    system = System(elements=[links], interactions=[rate], solver=VelocityVerletSolver(dt=0.05))
    trajectory = system.prepare().simulate(until=50.0)
    electric = -trajectory.get_values("links.a'")
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
    links = Element('links', values={'a': amplitude}, space=space, on='edges')
    rate = Interaction('field', roles={'target': 'links'}, laws=_wave)
    expected = (2 / spacing) * np.sin(wave_number * spacing / 2)
    period = 2 * np.pi / expected
    until = 4 * period
    step = period / 200
    system = System(elements=[links], interactions=[rate], solver=VelocityVerletSolver(dt=step / 4))
    trajectory = system.prepare().simulate(until=until, dt=step)
    series = trajectory.get_values('links.a')[:, size * size]
    crossings = np.flatnonzero(np.diff(np.sign(series)) != 0)
    measured = np.pi * (crossings.size - 1) / (trajectory.times[crossings[-1]] - trajectory.times[crossings[0]])
    assert abs(measured - expected) / expected < DISPERSION_LIMIT
