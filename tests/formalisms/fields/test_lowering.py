"""Test fields, differential operators, boundary conditions, stability limits, and coupling with agents."""

import warnings

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.base import ValidationError, state
from skcomplex.core.components import ElementComponent, GroupComponent
from skcomplex.core.laws import RateLaw
from skcomplex.core.space import LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.formalisms.agents import RuleLaw
from skcomplex.formalisms.fields import FieldComponent, advection, divergence, gradient, laplacian

HeatComponent = ElementComponent.define(
    'HeatComponent',
    states={'u': 0.0},
    parameters={'D': 1.0},
    laws=lambda s: [RateLaw(s.u, s.D * laplacian(s.u))],
)
DecayComponent = ElementComponent.define(
    'DecayComponent',
    states={'u': 2.0},
    parameters={'k': 0.5},
    laws=lambda s: [RateLaw(s.u, -s.k * s.u)],
)
DriftComponent = ElementComponent.define(
    'DriftComponent',
    states={'u': 0.0},
    laws=lambda s: [RateLaw(s.u, -advection(s.u, [1.0]))],
)
SpreadComponent = ElementComponent.define(
    'SpreadComponent',
    states={'u': 0.0},
    laws=lambda s: [RateLaw(s.u, divergence(gradient(s.u, 1)))],
)


def _heat_kernel(backend):
    """Return a point of heat diffused for 0.5 on a periodic line of 400 sites and the Gaussian heat kernel."""
    n, spacing = 400, 0.05
    positions = np.arange(n) * spacing
    initial = np.zeros(n)
    initial[n // 2] = 1 / spacing
    rod = FieldComponent('rod', HeatComponent, LatticeSpace(n, spacing=spacing), values={'u': initial})
    values = (
        Simulation(System(components=[rod]), dt=0.5, solver=RK4Solver(dt=0.001), backend=backend).run(until=0.5).state_
    )
    kernel = np.exp(-((positions - positions[n // 2]) ** 2) / 2) / np.sqrt(2 * np.pi)
    return values, kernel


def test_uniform_decay():
    """Test that every site of a 50 x 50 field follows the exponential decay of the single-site equation."""
    trajectory = (
        Simulation(System(components=[FieldComponent('grid', DecayComponent, LatticeSpace((50, 50)))]), dt=0.5)
        .run(2)
        .trajectory_
    )
    expected = 2 * np.exp(-0.5 * trajectory.times)[:, None] * np.ones(2500)
    assert trajectory.values('grid.u') == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_heat_kernel(backend):
    """Test that a point of heat on a periodic line matches the Gaussian heat kernel within 1% of its peak."""
    values, kernel = _heat_kernel(backend)
    assert np.abs(values - kernel).max() < 0.01 * kernel.max()


def test_diffusion_on_network():
    """Test that a field diffusing on a connected network keeps its sum and converges to the mean."""
    graph = nx.connected_watts_strogatz_graph(50, 4, 0.2, seed=1)
    initial = np.random.default_rng(0).random(50)
    system = System(components=[FieldComponent('net', HeatComponent, NetworkSpace(graph), values={'u': initial})])
    values = Simulation(system, dt=1).run(until=60).trajectory_.values('net.u')
    assert values.sum(axis=1) == pytest.approx(np.full(len(values), initial.sum()))
    assert np.abs(values[-1] - initial.mean()).max() < 1e-4


def test_diffusion_on_network_without_edges():
    """Test that a field on nodes without edges compiles, with no diffusive step limit, and stays put."""
    initial = np.array([0.2, 0.7])
    system = System(
        components=[FieldComponent('net', HeatComponent, NetworkSpace(nx.empty_graph(2)), values={'u': initial})],
    )
    values = Simulation(system, dt=1).run(until=3).trajectory_.values('net.u')
    assert values[-1] == pytest.approx(initial)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_zero_flux_conserves_mass(backend):
    """Test that a field diffusing with zero-flux edges keeps its total within 1e-10."""
    initial = np.random.default_rng(0).random((30, 40))
    plate = FieldComponent(
        'plate',
        HeatComponent,
        LatticeSpace((30, 40)),
        boundary={'u': 'zero-flux'},
        values={'u': initial},
    )
    totals = (
        Simulation(System(components=[plate]), dt=1, backend=backend).run(until=30).trajectory_.values('plate.u').sum(1)
    )
    assert np.abs(totals - initial.sum()).max() < 1e-10 * initial.sum()
    assert totals.std() < 1e-10 * initial.sum()


def test_fixed_boundary_holds_edge_value():
    """Test that a fixed boundary drives a line towards the linear profile between its two edge values."""
    rod = FieldComponent('rod', HeatComponent, LatticeSpace(20), boundary={'u': ('fixed', 1.0)})
    values = Simulation(System(components=[rod]), dt=50).run(until=500).state_
    assert values == pytest.approx(np.ones(20), abs=1e-3)


def test_advection_and_divergence():
    """Test that upwind advection carries a profile at its velocity and that div grad diffuses conservatively."""
    positions = np.arange(200) * 0.05
    profile = np.exp(-((positions - 3) ** 2))
    space = LatticeSpace(200, spacing=0.05)
    moved = (
        Simulation(System(components=[FieldComponent('a', DriftComponent, space, values={'u': profile})]), dt=1)
        .run(2)
        .state_
    )
    assert positions[np.argmax(moved)] == pytest.approx(5.0)
    assert moved.sum() == pytest.approx(profile.sum())
    spread = (
        Simulation(System(components=[FieldComponent('d', SpreadComponent, space, values={'u': profile})]), dt=1)
        .run(1)
        .state_
    )
    assert spread.max() < profile.max()
    assert spread.sum() == pytest.approx(profile.sum())


def test_equations_keep_operators():
    """Test that the equations of a field show its operators, not the stencils they lower to."""
    problem = compile_system(System(components=[FieldComponent('rod', HeatComponent, LatticeSpace(10))]))
    assert str(problem.equations['rod.u[i]']) == 'D_rod*laplacian(u_rod)'


def test_operator_outside_field_is_rejected():
    """Test that an operator on a group that is not a field is rejected."""
    with pytest.raises(ValidationError, match='Create the group with `FieldComponent`'):
        compile_system(System(components=[GroupComponent('rod', element=HeatComponent, size=5)]))


def test_unknown_boundary_is_rejected():
    """Test that an unknown boundary condition names the known ones."""
    rod = FieldComponent('rod', HeatComponent, LatticeSpace(10), boundary={'u': 'open'})
    with pytest.raises(ValueError, match='zero-flux'):
        compile_system(System(components=[rod]))


def test_too_large_step_warns():
    """Test that RK4 with four times the diffusive stability limit warns with the limit before the run."""
    problem = compile_system(System(components=[FieldComponent('plate', HeatComponent, LatticeSpace((20, 20)))]))
    assert problem.stable_step == pytest.approx(0.25)
    with pytest.warns(UserWarning, match='stability limit 0.25'):
        Simulation(problem, dt=1.0, solver=RK4Solver(dt=1.0)).run(until=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Simulation(problem, dt=1.0, solver=RK4Solver(dt=0.2)).run(until=1)


class MarkerComponent(ElementComponent):
    """Agent that reads a field at its cell and deposits one unit there every step."""

    seen: float = state(default=0.0)

    def laws(self, s):
        """Return the marking rule."""
        return [RuleLaw(self.mark)]

    def mark(self, agent, model):
        """Keep the field value at the cell and deposit one unit."""
        agent.set_state('seen', model.field('ground/level.u', agent.location))
        model.deposit('ground/level.u', agent.location, 1.0)


def test_agents_read_and_deposit_fields():
    """Test that deposits reach the field at every synchronisation point and agents read the value there."""
    StillComponent = ElementComponent.define(
        'StillComponent',
        states={'u': 0.0},
        laws=lambda s: [RateLaw(s.u, 0 * laplacian(s.u))],
    )
    system = System(
        components=[
            System(
                'walkers',
                components=[MarkerComponent.group('markers', 1, location=(1, 2))],
                space=LatticeSpace((4, 3), boundary='fixed'),
            ),
            System('ground', components=[FieldComponent('level', StillComponent, LatticeSpace((4, 3)))]),
        ],
    )
    simulation = Simulation(system, dt=1).run(until=5)
    level = simulation.trajectory_.lattice('ground/level.u')[-1]
    assert level[1, 2] == pytest.approx(5.0)
    assert level.sum() == pytest.approx(5.0)
    assert simulation.trajectory_.agents('walkers/markers', 5).values['seen'][0] == pytest.approx(4.0)


GrowthComponent = ElementComponent.define(
    'GrowthComponent',
    states={'u': 0.0},
    laws=lambda s: [RateLaw(s.u, s.u + laplacian(s.u))],
)


@pytest.mark.parametrize('discretization', ['spectral', 'finite-difference'])
def test_laplacian_damps_the_checkerboard_mode(discretization):
    """Test that the Laplacian damps the Nyquist mode, so a checkerboard of `u' = u + lap(u)` decays."""
    rows, columns = np.indices((16, 16))
    checkerboard = 0.01 * np.cos(np.pi * (rows + columns))
    field = FieldComponent(
        'm',
        GrowthComponent,
        LatticeSpace((16, 16), spacing=1.0),
        values={'u': checkerboard},
        discretization=discretization,
    )
    final = np.asarray(Simulation(System(components=[field]), dt=0.01).run(until=0.5).trajectory_.values('m.u'))[-1]
    expected = (
        0.01 * np.exp((1 - 8) * 0.5)
        if discretization == 'finite-difference'
        else 0.01 * np.exp((1 - 2 * np.pi**2) * 0.5)
    )
    assert np.abs(final).max() == pytest.approx(expected, rel=1e-2)
