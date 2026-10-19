"""Test fields, differential operators, boundary conditions, stability limits, and coupling with agents."""

import warnings

import networkx as nx
import numpy as np
import pytest

from skcomplex.compile import compile_system
from skcomplex.core import ValidationError, state
from skcomplex.model import (
    AgentType,
    Element,
    Field,
    GridSpace,
    Group,
    LatticeSpace,
    NetworkSpace,
    Rate,
    Rule,
    System,
    advection,
    divergence,
    gradient,
    laplacian,
)
from skcomplex.runtime import Simulation
from skcomplex.solvers import RK4

Heat = Element.define(
    'Heat',
    states={'u': 0.0},
    parameters={'D': 1.0},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u))],
)
Decay = Element.define('Decay', states={'u': 2.0}, parameters={'k': 0.5}, laws=lambda s: [Rate(s.u, -s.k * s.u)])
Drift = Element.define('Drift', states={'u': 0.0}, laws=lambda s: [Rate(s.u, -advection(s.u, [1.0]))])
Spread = Element.define('Spread', states={'u': 0.0}, laws=lambda s: [Rate(s.u, divergence(gradient(s.u, 1)))])


def _heat_kernel(backend):
    """Return a point of heat diffused for 0.5 on a periodic line of 400 sites and the Gaussian heat kernel."""
    n, spacing = 400, 0.05
    positions = np.arange(n) * spacing
    initial = np.zeros(n)
    initial[n // 2] = 1 / spacing
    rod = Field('rod', Heat, LatticeSpace(n, spacing=spacing), values={'u': initial})
    values = Simulation(System(elements=[rod]), dt=0.5, solver=RK4(dt=0.001), backend=backend).run(until=0.5).state_
    kernel = np.exp(-((positions - positions[n // 2]) ** 2) / 2) / np.sqrt(2 * np.pi)
    return values, kernel


def test_uniform_decay():
    """Test that every site of a 50 x 50 field follows the exponential decay of the single-site equation."""
    trajectory = Simulation(System(elements=[Field('grid', Decay, LatticeSpace((50, 50)))]), dt=0.5).run(2).trajectory_
    assert trajectory['grid.u'] == pytest.approx(2 * np.exp(-0.5 * trajectory.times)[:, None] * np.ones(2500))


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_heat_kernel(backend):
    """Test that a point of heat on a periodic line matches the Gaussian heat kernel within 1% of its peak."""
    values, kernel = _heat_kernel(backend)
    assert np.abs(values - kernel).max() < 0.01 * kernel.max()


def test_diffusion_on_network():
    """Test that a field diffusing on a connected network keeps its sum and converges to the mean."""
    graph = nx.connected_watts_strogatz_graph(50, 4, 0.2, seed=1)
    initial = np.random.default_rng(0).random(50)
    system = System(elements=[Field('net', Heat, NetworkSpace(graph), values={'u': initial})])
    values = Simulation(system, dt=1).run(until=60).trajectory_['net.u']
    assert values.sum(axis=1) == pytest.approx(np.full(len(values), initial.sum()))
    assert np.abs(values[-1] - initial.mean()).max() < 1e-4


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_zero_flux_conserves_mass(backend):
    """Test that a field diffusing with zero-flux edges keeps its total within 1e-10."""
    initial = np.random.default_rng(0).random((30, 40))
    plate = Field('plate', Heat, LatticeSpace((30, 40)), boundary={'u': 'zero-flux'}, values={'u': initial})
    totals = Simulation(System(elements=[plate]), dt=1, backend=backend).run(until=30).trajectory_['plate.u'].sum(1)
    assert np.abs(totals - initial.sum()).max() < 1e-10 * initial.sum()
    assert totals.std() < 1e-10 * initial.sum()


def test_fixed_boundary_holds_edge_value():
    """Test that a fixed boundary drives a line towards the linear profile between its two edge values."""
    rod = Field('rod', Heat, LatticeSpace(20), boundary={'u': ('fixed', 1.0)})
    values = Simulation(System(elements=[rod]), dt=50).run(until=500).state_
    assert values == pytest.approx(np.ones(20), abs=1e-3)


def test_advection_and_divergence():
    """Test that upwind advection carries a profile at its velocity and that div grad diffuses conservatively."""
    positions = np.arange(200) * 0.05
    profile = np.exp(-((positions - 3) ** 2))
    space = LatticeSpace(200, spacing=0.05)
    moved = Simulation(System(elements=[Field('a', Drift, space, values={'u': profile})]), dt=1).run(2).state_
    assert positions[np.argmax(moved)] == pytest.approx(5.0)
    assert moved.sum() == pytest.approx(profile.sum())
    spread = Simulation(System(elements=[Field('d', Spread, space, values={'u': profile})]), dt=1).run(1).state_
    assert spread.max() < profile.max()
    assert spread.sum() == pytest.approx(profile.sum())


def test_equations_keep_operators():
    """Test that the equations of a field show its operators, not the stencils they lower to."""
    problem = compile_system(System(elements=[Field('rod', Heat, LatticeSpace(10))]))
    assert str(problem.equations['rod.u[i]']) == 'D_rod*laplacian(u_rod)'


def test_operator_outside_field_is_rejected():
    """Test that an operator on a group that is not a field is rejected."""
    with pytest.raises(ValidationError, match='Create the group with `Field`'):
        compile_system(System(elements=[Group('rod', element=Heat, size=5)]))


def test_unknown_boundary_is_rejected():
    """Test that an unknown boundary condition names the known ones."""
    rod = Field('rod', Heat, LatticeSpace(10), boundary={'u': 'open'})
    with pytest.raises(ValueError, match='zero-flux'):
        compile_system(System(elements=[rod]))


def test_too_large_step_warns():
    """Test that RK4 with four times the diffusive stability limit warns with the limit before the run."""
    problem = compile_system(System(elements=[Field('plate', Heat, LatticeSpace((20, 20)))]))
    assert problem.stable_step == pytest.approx(0.25)
    with pytest.warns(UserWarning, match='stability limit 0.25'):
        Simulation(problem, dt=1.0, solver=RK4(dt=1.0)).run(until=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Simulation(problem, dt=1.0, solver=RK4(dt=0.2)).run(until=1)


class Marker(AgentType):
    """Agent that reads a field at its cell and deposits one unit there every step."""

    seen: float = state(default=0.0)

    def laws(self, s):
        """Return the marking rule."""
        return [Rule(self.mark)]

    def mark(self, agent, model):
        """Keep the field value at the cell and deposit one unit."""
        agent.seen = model.field('ground/level.u', agent.location)
        model.deposit('ground/level.u', agent.location, 1.0)


def test_agents_read_and_deposit_fields():
    """Test that deposits reach the field at every synchronisation point and agents read the value there."""
    Still = Element.define('Still', states={'u': 0.0}, laws=lambda s: [Rate(s.u, 0 * laplacian(s.u))])
    system = System(
        systems=[
            System('walkers', elements=[Marker.group('markers', 1, location=(1, 2))], space=GridSpace(4, 3)),
            System('ground', elements=[Field('level', Still, LatticeSpace((4, 3)))]),
        ],
    )
    simulation = Simulation(system, dt=1).run(until=5)
    level = simulation.trajectory_.lattice('ground/level.u')[-1]
    assert level[1, 2] == pytest.approx(5.0)
    assert level.sum() == pytest.approx(5.0)
    assert simulation.trajectory_.agents('walkers/markers', 5).values['seen'][0] == pytest.approx(4.0)
