"""Test fields, differential operators, boundary conditions, stability limits, and coupling with agents."""

import warnings
from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, FieldElement, GroupElement
from skcomplex.core.laws import RateLaw, RuleLaw
from skcomplex.core.laws.utils import advection, divergence, gradient, laplacian
from skcomplex.core.spaces import LatticeSpace, NetworkSpace
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
            .check()
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
            )
            .check()
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
    values = Simulation(system, dt=1).check().run(until=60).trajectory_.values('net.u')
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
    values = Simulation(system, dt=1).check().run(until=3).trajectory_.values('net.u')
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
            .check()
            .run(until=30)
            .trajectory_
            .values('plate.u')
            .sum(1)

    )
    assert np.abs(totals - initial.sum()).max() < 1e-10 * initial.sum()
    assert totals.std() < 1e-10 * initial.sum()


def test_fixed_boundary_holds_edge_value():
    """Test that a fixed boundary drives a line towards the linear profile between its two edge values."""
    rod = FieldElement('rod', element=HeatElement, space=LatticeSpace(20), boundary={'u': ('fixed', 1.0)})
    values = Simulation(System(elements=[rod]), dt=50).check().run(until=500).state_
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
            )
            .check()
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
        ).check()
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
        Simulation(problem, dt=1.0, solver=RK4Solver(dt=1.0)).check().run(until=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Simulation(problem, dt=1.0, solver=RK4Solver(dt=0.2)).check().run(until=1)


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
    simulation = Simulation(system, dt=1).check().run(until=5)
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
    final = np.asarray(Simulation(
        System(elements=[field]), dt=0.01,
    ).check().run(until=0.5).trajectory_.values('m.u'))[-1]
    expected = (
        0.01 * np.exp((1 - 8) * 0.5)
        if discretization == 'finite-difference'
        else 0.01 * np.exp((1 - 2 * np.pi**2) * 0.5)
    )
    assert np.abs(final).max() == pytest.approx(expected, rel=1e-2)
