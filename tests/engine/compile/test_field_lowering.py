"""Test fields, differential operators, boundary conditions, stability limits, and coupling with agents."""

import warnings
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import exp, pi, sin, sqrt
from skfem import MeshTri

from skcomplex.core.elements import Element, FieldElement, GroupElement
from skcomplex.core.laws import RateLaw, RuleLaw
from skcomplex.core.laws.utils import (
    advection,
    convolve,
    displacement,
    divergence,
    gradient,
    inverse_helmholtz,
    inverse_laplacian,
    laplacian,
    shifted,
)
from skcomplex.core.spaces import LatticeSpace, MeshSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec


class HeatElement(Element):
    """Heat element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
        D: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D

    def _get_laws(self, s):
        """Return the laws of the heat element."""
        return [RateLaw(s.u, s.D * laplacian(s.u))]


class DecayElement(Element):
    """Decay element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 2.0,
        k: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.k = k

    def _get_laws(self, s):
        """Return the laws of the decay element."""
        return [RateLaw(s.u, -s.k * s.u)]


class DriftElement(Element):
    """Drift element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u

    def _get_laws(self, s):
        """Return the laws of the drift element."""
        return [RateLaw(s.u, -advection(s.u, [1.0]))]


class SpreadElement(Element):
    """Spread element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u

    def _get_laws(self, s):
        """Return the laws of the spread element."""
        return [RateLaw(s.u, divergence(gradient(s.u, 1)))]


def _heat_kernel(backend):
    """Return a point of heat diffused for 0.5 on a periodic line of 400 sites and the Gaussian heat kernel."""
    n, spacing = 400, 0.05
    positions = np.arange(n) * spacing
    initial = np.zeros(n)
    initial[n // 2] = 1 / spacing
    rod = FieldElement('rod', element=HeatElement, space=LatticeSpace(n, spacing=spacing), values={'u': initial})
    values = (
        Simulation(System(elements=[rod]), dt=0.5, solver=RK4Solver(dt=0.001), backend=backend)
        .prepare()
        .run(until=0.5)
        .state_
    )
    kernel = np.exp(-((positions - positions[n // 2]) ** 2) / 2) / np.sqrt(2 * np.pi)
    return values, kernel


def test_uniform_decay():
    """Test that every site of a 50 x 50 field follows the exponential decay of the single-site equation."""
    trajectory = (
        (
            Simulation(
                System(elements=[FieldElement('grid', element=DecayElement, space=LatticeSpace((50, 50)))]),
                dt=0.5,
            ).prepare()
        )
        .run(2)
        .trajectory_
    )
    expected = 2 * np.exp(-0.5 * trajectory.times)[:, None] * np.ones(2500)
    assert trajectory.values('grid.u') == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_heat_kernel(backend):
    """Test that a point of heat on a periodic line matches the Gaussian heat kernel within 1% of its peak."""
    values, kernel = _heat_kernel(backend())
    assert np.abs(values - kernel).max() < 0.01 * kernel.max()


def test_diffusion_on_network():
    """Test that a field diffusing on a connected network keeps its sum and converges to the mean."""
    graph = nx.connected_watts_strogatz_graph(50, 4, 0.2, seed=1)
    initial = np.random.default_rng(0).random(50)
    system = System(
        elements=[FieldElement('net', element=HeatElement, space=NetworkSpace(graph), values={'u': initial})],
    )
    values = Simulation(system, dt=1).prepare().run(until=60).trajectory_.values('net.u')
    assert values.sum(axis=1) == pytest.approx(np.full(len(values), initial.sum()))
    assert np.abs(values[-1] - initial.mean()).max() < 1e-4


def test_diffusion_on_network_without_edges():
    """Test that a field on nodes without edges compiles, with no diffusive step limit, and stays put."""
    initial = np.array([0.2, 0.7])
    system = System(
        elements=[
            FieldElement('net', element=HeatElement, space=NetworkSpace(nx.empty_graph(2)), values={'u': initial}),
        ],
    )
    values = Simulation(system, dt=1).prepare().run(until=3).trajectory_.values('net.u')
    assert values[-1] == pytest.approx(initial)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_zero_flux_conserves_mass(backend):
    """Test that a field diffusing with zero-flux edges keeps its total within 1e-10."""
    initial = np.random.default_rng(0).random((30, 40))
    plate = FieldElement(
        'plate',
        element=HeatElement,
        space=LatticeSpace((30, 40)),
        boundary={'u': 'zero-flux'},
        values={'u': initial},
    )
    totals = (
        Simulation(System(elements=[plate]), dt=1, backend=backend())
        .prepare()
        .run(until=30)
        .trajectory_.values('plate.u')
        .sum(1)
    )
    assert np.abs(totals - initial.sum()).max() < 1e-10 * initial.sum()
    assert totals.std() < 1e-10 * initial.sum()


def test_fixed_boundary_holds_edge_value():
    """Test that a fixed boundary drives a line towards the linear profile between its two edge values."""
    rod = FieldElement('rod', element=HeatElement, space=LatticeSpace(20), boundary={'u': ('fixed', 1.0)})
    values = Simulation(System(elements=[rod]), dt=50).prepare().run(until=500).state_
    assert values == pytest.approx(np.ones(20), abs=1e-3)


def test_advection_and_divergence():
    """Test that upwind advection carries a profile at its velocity and that div grad diffuses conservatively."""
    positions = np.arange(200) * 0.05
    profile = np.exp(-((positions - 3) ** 2))
    space = LatticeSpace(200, spacing=0.05)
    moved = (
        (
            Simulation(
                System(elements=[FieldElement('a', element=DriftElement, space=space, values={'u': profile})]),
                dt=1,
            ).prepare()
        )
        .run(2)
        .state_
    )
    assert positions[np.argmax(moved)] == pytest.approx(5.0)
    assert moved.sum() == pytest.approx(profile.sum())
    spread = (
        Simulation(
            System(elements=[FieldElement('d', element=SpreadElement, space=space, values={'u': profile})]),
            dt=1,
        )
        .prepare()
        .run(1)
        .state_
    )
    assert spread.max() < profile.max()
    assert spread.sum() == pytest.approx(profile.sum())


def test_equations_keep_operators():
    """Test that the equations of a field show its operators, not the stencils they lower to."""
    problem = compile_system(System(elements=[FieldElement('rod', element=HeatElement, space=LatticeSpace(10))]))
    assert str(problem.equations['rod.u[i]']) == 'D_rod*laplacian(u_rod)'


def test_operator_outside_field_is_rejected():
    """Test that an operator on a group that is not a field is rejected."""
    with pytest.raises(ValidationError, match='Create the group with `FieldElement`'):
        compile_system(System(elements=[GroupElement('rod', element=HeatElement, size=5)]))


def test_unknown_boundary_is_rejected():
    """Test that an unknown boundary condition names the known ones."""
    rod = FieldElement('rod', element=HeatElement, space=LatticeSpace(10), boundary={'u': 'open'})
    with pytest.raises(ValidationError, match='zero-flux'):
        compile_system(System(elements=[rod]))


def test_too_large_step_warns():
    """Test that RK4 with four times the diffusive stability limit warns with the limit before the run."""
    problem = compile_system(
        System(elements=[FieldElement('plate', element=HeatElement, space=LatticeSpace((20, 20)))]),
    )
    assert problem.stable_step == pytest.approx(0.25)
    with pytest.warns(UserWarning, match='stability limit 0.25'):
        Simulation(problem, dt=1.0, solver=RK4Solver(dt=1.0)).prepare().run(until=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Simulation(problem, dt=1.0, solver=RK4Solver(dt=0.2)).prepare().run(until=1)


class MarkerElement(Element):
    """Agent that reads a field at its cell and deposits one unit there every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        seen: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.seen = seen

    def _get_laws(self, s):
        """Return the marking rule."""
        return [RuleLaw(self.mark)]

    def mark(self, agent, model):
        """Keep the field value at the cell and deposit one unit."""
        agent.seen = model.field('ground/level.u', agent.location)
        model.deposit('ground/level.u', agent.location, 1.0)


def test_agents_read_and_deposit_fields():
    """Test that deposits reach the field at every synchronisation point and agents read the value there."""

    class StillElement(Element):
        """Still element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            u: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.u = u

        def _get_laws(self, s):
            """Return the laws of the still element."""
            return [RateLaw(s.u, 0 * laplacian(s.u))]

    system = System(
        systems=[
            System(
                'walkers',
                elements=[GroupElement('markers', element=MarkerElement, size=1, location=(1, 2))],
                space=LatticeSpace((4, 3), boundary='fixed'),
            ),
            System('ground', elements=[FieldElement('level', element=StillElement, space=LatticeSpace((4, 3)))]),
        ],
    )
    simulation = Simulation(system, dt=1).prepare().run(until=5)
    level = simulation.trajectory_.lattice('ground/level.u')[-1]
    assert level[1, 2] == pytest.approx(5.0)
    assert level.sum() == pytest.approx(5.0)
    assert simulation.trajectory_.agents('walkers/markers', 5).values['seen'][0] == pytest.approx(4.0)


class GrowthElement(Element):
    """Growth element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u

    def _get_laws(self, s):
        """Return the laws of the growth element."""
        return [RateLaw(s.u, s.u + laplacian(s.u))]


@pytest.mark.parametrize('discretization', ['spectral', 'finite-difference'])
def test_laplacian_damps_the_checkerboard_mode(discretization):
    """Test that the Laplacian damps the Nyquist mode, so a checkerboard of `u' = u + lap(u)` decays."""
    rows, columns = np.indices((16, 16))
    checkerboard = 0.01 * np.cos(np.pi * (rows + columns))
    field = FieldElement(
        'm',
        element=GrowthElement,
        space=LatticeSpace((16, 16), spacing=1.0),
        values={'u': checkerboard},
        discretization=discretization,
    )
    final = np.asarray(
        Simulation(
            System(elements=[field]),
            dt=0.01,
        )
        .prepare()
        .run(until=0.5)
        .trajectory_.values('m.u'),
    )[-1]
    expected = (
        0.01 * np.exp((1 - 8) * 0.5)
        if discretization == 'finite-difference'
        else 0.01 * np.exp((1 - 2 * np.pi**2) * 0.5)
    )
    assert np.abs(final).max() == pytest.approx(expected, rel=1e-2)


def _operator_element(law):
    """Return an element of two states and two parameters whose first state changes at the rate `law(s)`."""

    class OperatorElement(Element):
        """Operator element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            u: Annotated[object, StateSpec()] = 0.0,
            v: Annotated[object, StateSpec()] = 0.0,
            k: Annotated[object, ParameterSpec()] = 1.0,
            w: Annotated[object, ParameterSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.u = u
            self.v = v
            self.k = k
            self.w = w

        def _get_laws(self, s):
            """Return the laws of the operator element."""
            return [RateLaw(s.u, law(s)), RateLaw(s.v, 0 * s.v)]

    return OperatorElement


def _rate(law, space, values, backend=None, **settings):
    """Return the rate of the first state of a field at its initial values, at every site."""
    field = FieldElement('m', element=_operator_element(law), space=space, values=values, **settings)
    problem = compile_system(System(elements=[field]), backend=backend or NumpyBackend())
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    return np.ravel(problem.states.read(drift, 'm.u'))


def _basin(n):
    """Return the sites of a unit square basin of n x n interior sites, the spacing, and its lattice."""
    spacing = 1.0 / (n + 1)
    x, y = np.meshgrid(np.arange(1, n + 1) * spacing, np.arange(1, n + 1) * spacing, indexing='ij')
    return x, y, LatticeSpace((n, n), spacing=spacing, boundary='fixed')


def test_inverse_helmholtz_inverts_a_single_fourier_mode():
    """Test that a spectral field inverts `sin(kx) sin(ly)` with screening kappa exactly."""
    n, spacing = 32, 2 * np.pi / 32
    x, y = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing, indexing='ij')
    mode = np.sin(2 * x) * np.sin(3 * y)
    rate = _rate(
        lambda s: inverse_helmholtz(s.u, s.k),
        LatticeSpace((n, n), spacing=spacing),
        {'u': mode, 'k': 1.5},
        discretization='spectral',
    )
    assert np.abs(rate - np.ravel(-mode / (4 + 9 + 1.5**2))).max() < 1e-12


def test_inverse_helmholtz_converges_in_a_closed_basin():
    """Test that inverting with screening 2 at fixed zero edges converges at second order to the exact solution."""
    errors = []
    for n in (32, 64):
        x, y, space = _basin(n)
        solution = np.sin(np.pi * x) * np.sin(2 * np.pi * y)
        rate = _rate(lambda s: inverse_helmholtz(s.u, 2), space, {'u': -(5 * np.pi**2 + 4) * solution})
        errors.append(np.abs(rate - np.ravel(solution)).max())
    assert errors[0] / errors[1] > 3.8


def test_inverse_laplacian_has_zero_mean_on_a_periodic_lattice():
    """Test that an unscreened periodic inversion has zero mean and its Laplacian is the field minus its mean."""
    n, spacing = 32, 2 * np.pi / 32
    x, y = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing, indexing='ij')
    field = 1 + np.sin(x) + np.cos(2 * y) * np.sin(x)
    space = LatticeSpace((n, n), spacing=spacing)
    solution = _rate(lambda s: inverse_laplacian(s.u), space, {'u': field})
    curvature = _rate(lambda s: laplacian(s.u), space, {'u': np.reshape(solution, (n, n))})
    assert abs(solution.mean()) < 1e-10
    assert np.abs(curvature - np.ravel(field - field.mean())).max() < 1e-10


def test_inverse_laplacian_has_zero_mean_at_zero_flux_edges():
    """Test that an unscreened inversion at zero-flux edges recovers a cosine with zero mean at second order."""
    errors = []
    for n in (40, 80):
        x = (np.arange(n) + 0.5) / n
        solution = np.cos(np.pi * x)
        space = LatticeSpace(n, spacing=1 / n)
        rate = _rate(
            lambda s: inverse_laplacian(s.u),
            space,
            {'u': 3 - np.pi**2 * solution},
            boundary={'u': 'zero-flux'},
        )
        assert abs(rate.mean()) < 1e-12
        errors.append(np.abs(rate - solution).max())
    assert errors[0] / errors[1] > 3.8


def test_inverse_laplacian_on_a_network_has_zero_mean():
    """Test that an unscreened inversion on a network has zero mean and inverts the graph Laplacian."""
    graph = nx.connected_watts_strogatz_graph(30, 4, 0.2, seed=1)
    field = np.random.default_rng(0).random(30)
    solution = _rate(lambda s: inverse_laplacian(s.u), NetworkSpace(graph), {'u': field})
    curvature = _rate(lambda s: laplacian(s.u), NetworkSpace(graph), {'u': solution})
    assert abs(solution.mean()) < 1e-12
    assert np.abs(curvature - (field - field.mean())).max() < 1e-12


def test_inverse_helmholtz_converges_on_an_l_shaped_mesh():
    """Test that a screened inversion on an L-shaped mesh with fixed zero edges converges at second order."""
    errors = []
    for refinements in (3, 4):
        mesh = MeshTri.init_lshaped().refined(refinements)
        solution = np.sin(np.pi * mesh.p[0]) * np.sin(np.pi * mesh.p[1])
        rate = _rate(
            lambda s: inverse_helmholtz(s.u, 1),
            MeshSpace(mesh.p, mesh.t),
            {'u': -(2 * np.pi**2 + 1) * solution},
            discretization='finite-element',
            boundary={'u': ('fixed', 0.0)},
        )
        errors.append(np.abs(rate - solution).max())
    assert errors[0] / errors[1] > 3.8


def test_gradient_of_an_inversion_converges_in_a_basin():
    """Test that the gradient of an inversion at fixed zero edges converges at second order to the exact gradient."""
    errors = []
    for n in (32, 64):
        x, y, space = _basin(n)
        solution = np.sin(np.pi * x) * np.sin(2 * np.pi * y)
        rate = _rate(
            lambda s: gradient(inverse_helmholtz(s.u, s.k), 2)[0],
            space,
            {'u': -(5 * np.pi**2 + 4) * solution, 'k': 2.0},
        )
        errors.append(np.abs(rate - np.ravel(np.pi * np.cos(np.pi * x) * np.sin(2 * np.pi * y))).max())
    assert errors[0] / errors[1] > 3.8


def test_convolve_smooths_a_gaussian():
    """Test that a Gaussian of width 1 smoothed by a normalised Gaussian of width 0.5 is the Gaussian of their sum."""
    positions = np.arange(400) * 0.1 - 20
    kernel = exp(-(displacement(0) ** 2) / (2 * 0.5**2)) / sqrt(2 * pi * 0.5**2)
    rate = _rate(lambda s: convolve(s.u, kernel), LatticeSpace(400, spacing=0.1), {'u': np.exp(-(positions**2) / 2)})
    expected = np.exp(-(positions**2) / 2.5) / np.sqrt(1.25)
    assert np.abs(rate - expected).max() < 1e-8


@pytest.mark.parametrize('radius', [2, 5])
def test_convolve_truncates_at_fixed_edges(radius):
    """Test that a convolution with fixed zero edges equals the direct sum over the sites within the radius."""
    field = np.random.default_rng(0).random((64, 48))
    space = LatticeSpace((64, 48), spacing=0.5, boundary='fixed')
    kernel = exp(-displacement(0) - displacement(1) ** 2)
    rate = _rate(lambda s: convolve(s.u, kernel, radius * 0.5), space, {'u': field})
    padded = np.pad(field, radius)
    expected = np.zeros_like(field)
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            if a**2 + b**2 <= radius**2:
                moved = padded[radius - a : radius - a + 64, radius - b : radius - b + 48]
                expected += np.exp(-a * 0.5 - (b * 0.5) ** 2) * moved * 0.25
    assert np.abs(rate - np.ravel(expected)).max() < 1e-12


def test_convolve_mirrors_at_zero_flux_edges():
    """Test that a wide convolution at zero-flux edges keeps a uniform field uniform at the kernel's sum."""
    space = LatticeSpace(50, spacing=0.2)
    kernel = exp(-abs(displacement(0)))
    rate = _rate(lambda s: convolve(s.u, kernel, 3), space, {'u': 2.0}, boundary={'u': 'zero-flux'})
    expected = 2 * 0.2 * sum(np.exp(-0.2 * abs(step)) for step in range(-15, 16))
    assert np.abs(rate - expected).max() < 1e-12


def test_convolve_follows_the_kernel_parameters():
    """Test that changing the kernel width between compilations changes the convolution to the wider kernel's."""
    positions = np.arange(200) * 0.1
    field = np.sin(positions) ** 2
    space = LatticeSpace(200, spacing=0.1)

    def law(s):
        return convolve(s.u, exp(-(displacement(0) ** 2) / (2 * s.w**2)))

    narrow = _rate(law, space, {'u': field, 'w': 1.0})
    wide = _rate(law, space, {'u': field, 'w': 2.0})
    steps = np.fft.fftfreq(200, d=1 / 200) * 0.1
    direct = np.real(np.fft.ifft(np.fft.fft(field) * np.fft.fft(np.exp(-(steps**2) / 8) * 0.1)))
    assert np.abs(wide - direct).max() < 1e-10
    assert np.abs(wide - narrow).max() > 1e-3


def test_convolve_composes_with_the_laplacian_on_a_spectral_field():
    """Test that the convolution of a Laplacian equals the Laplacian of the convolution within 1e-10."""
    positions = np.arange(64) * 0.25
    field = np.sin(2 * np.pi * positions / 16) + 0.3 * np.cos(6 * np.pi * positions / 16)
    kernel = exp(-(displacement(0) ** 2))
    space = LatticeSpace(64, spacing=0.25)
    outer = _rate(lambda s: convolve(laplacian(s.u), kernel), space, {'u': field}, discretization='spectral')
    inner = _rate(lambda s: laplacian(convolve(s.u, kernel)), space, {'u': field}, discretization='spectral')
    assert np.abs(outer - inner).max() < 1e-10


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_shifted_reads_whole_sites_exactly(backend):
    """Test that a read three spacings ahead on a periodic lattice is the field three sites ahead, on both backends."""
    field = np.random.default_rng(0).random((20, 10))
    space = LatticeSpace((20, 10), spacing=0.5)
    rate = _rate(lambda s: shifted(s.u, [1.5, 0]), space, {'u': field}, backend=backend())
    assert rate.tolist() == np.ravel(np.roll(field, -3, axis=0)).tolist()


def test_shifted_interpolates_at_second_order():
    """Test that a read at 0.37 spacings plus a state-dependent distance interpolates a sine at second order."""
    errors = []
    for n in (100, 200):
        spacing = 2 * np.pi / n
        positions = np.arange(n) * spacing
        reach = 0.3 + 0.2 * np.cos(positions)
        rate = _rate(
            lambda s, spacing=spacing: shifted(sin(s.u), [0.37 * spacing + s.v]),
            LatticeSpace(n, spacing=spacing),
            {'u': positions, 'v': reach},
        )
        errors.append(np.abs(rate - np.sin(positions + 0.37 * spacing + reach)).max())
    assert errors[0] / errors[1] > 3.8


def test_shifted_reads_the_fixed_value_beyond_the_edge():
    """Test that a read beyond the last site of a field with the fixed edge value 0.2 returns 0.2."""
    field = np.linspace(0, 1, 10)
    rate = _rate(
        lambda s: shifted(s.u, [s.v]),
        LatticeSpace(10),
        {'u': field, 'v': 2.5},
        boundary={'u': ('fixed', 0.2)},
    )
    assert rate[-2:] == pytest.approx([0.2, 0.2])
    assert rate[0] == pytest.approx(0.5 * (field[2] + field[3]))


def test_inverse_helmholtz_solves_with_the_boundary_it_is_given():
    """Test that an inversion given a fixed boundary solves its stencil with the fixed value, whatever the operand's."""
    field = np.random.default_rng(0).random((12, 10))
    space = LatticeSpace((12, 10), spacing=0.5)

    def law(s):
        solution = inverse_helmholtz(s.u, s.k, ('fixed', 0.3))
        return laplacian(solution) - s.k**2 * solution

    rate = _rate(law, space, {'u': field, 'k': 1.5}, boundary={'u': 'zero-flux'})
    assert np.abs(rate - np.ravel(field)).max() < 1e-10


def test_inverse_helmholtz_converges_with_a_given_boundary():
    """Test that an inversion given fixed zero edges converges at second order with a zero-flux operand."""
    errors = []
    for n in (32, 64):
        x, y, space = _basin(n)
        solution = np.sin(np.pi * x) * np.sin(2 * np.pi * y)
        rate = _rate(
            lambda s: inverse_helmholtz(s.u, 2, ('fixed', 0.0)),
            space,
            {'u': -(5 * np.pi**2 + 4) * solution},
            boundary={'u': 'zero-flux'},
        )
        errors.append(np.abs(rate - np.ravel(solution)).max())
    assert errors[0] / errors[1] > 3.8


def test_inverse_helmholtz_with_a_boundary_takes_states_of_several_boundaries():
    """Test that an inversion of states with different boundaries solves with the zero-flux boundary it is given."""
    first, second = np.random.default_rng(1).random((2, 16))
    space = LatticeSpace(16, spacing=0.25)

    def law(s):
        solution = inverse_helmholtz(s.u + s.v, 2, 'zero-flux')
        return laplacian(solution) - 4 * solution

    rate = _rate(law, space, {'u': first, 'v': second}, boundary={'u': ('fixed', 1.0), 'v': 'zero-flux'})
    assert np.abs(rate - (first + second)).max() < 1e-10
    with pytest.raises(ValidationError, match='different boundary conditions'):
        _rate(
            lambda s: inverse_helmholtz(s.u + s.v, 2),
            space,
            {'u': first, 'v': second},
            boundary={'u': ('fixed', 1.0), 'v': 'zero-flux'},
        )


def test_inverse_helmholtz_rejects_a_fixed_boundary_on_a_spectral_field():
    """Test that a spectral inversion given a fixed boundary raises an error that names the discretization."""
    with pytest.raises(ValidationError, match='not periodic'):
        _rate(
            lambda s: inverse_helmholtz(s.u, 1, ('fixed', 0.0)),
            LatticeSpace(16),
            {'u': np.zeros(16)},
            discretization='spectral',
        )


def test_shifted_interpolates_at_fourth_order_with_cubic_splines():
    """Test that a cubic shifted read of a sine at a state-dependent distance converges at fourth order."""
    errors = []
    for n in (50, 100):
        spacing = 2 * np.pi / n
        positions = np.arange(n) * spacing
        reach = 0.3 + 0.2 * np.cos(positions)
        rate = _rate(
            lambda s, spacing=spacing: shifted(sin(s.u), [0.37 * spacing + s.v], order=3),
            LatticeSpace(n, spacing=spacing),
            {'u': positions, 'v': reach},
        )
        errors.append(np.abs(rate - np.sin(positions + 0.37 * spacing + reach)).max())
    assert errors[0] / errors[1] > 12


def test_identical_operators_apply_once_per_evaluation(monkeypatch):
    """Test that repeated inversions transform once per evaluation, and that no value outlives its evaluation."""
    transforms = []
    forward = np.fft.rfftn
    monkeypatch.setattr(np.fft, 'rfftn', lambda *args, **kwargs: transforms.append(1) or forward(*args, **kwargs))
    n, spacing = 16, 2 * np.pi / 16
    positions = np.arange(n) * spacing

    def law(s):
        solution = inverse_laplacian(s.u)
        return inverse_laplacian(s.u) ** 2 + solution + laplacian(solution**2)

    field = FieldElement(
        'm',
        element=_operator_element(law),
        space=LatticeSpace(n, spacing=spacing),
        values={'u': np.sin(positions)},
        discretization='spectral',
    )
    problem = compile_system(System(elements=[field]), backend=NumpyBackend())
    state, parameters = problem.initial_state, problem.initial_parameters
    transforms.clear()
    first = problem.drift(0.0, state, parameters)
    assert len(transforms) == 2
    doubled = problem.drift(0.0, 2 * state, parameters)
    solution = -2 * np.sin(positions)
    expected = solution**2 + solution + 8 * np.cos(2 * positions)
    assert np.allclose(np.ravel(problem.states.read(doubled, 'm.u')), expected)
    assert np.array_equal(problem.drift(0.0, state, parameters), first)
    assert len(transforms) == 6
