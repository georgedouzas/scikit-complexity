"""Test the base of analyses: its verbs and the readers of trajectories, arrays, and graphs."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import NN

import skcomplex.analysis
from skcomplex.analysis import BaseAnalysis
from skcomplex.core.elements import Element
from skcomplex.core.laws import JumpLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import AnalysisError, NotCheckedError, NotPreparedError
from skcomplex.models import run
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec


class BirthDeathElement(Element):
    """Birth death element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec(domain=NN)] = 10.0,
        birth: Annotated[object, ParameterSpec()] = 5.0,
        death: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.birth = birth
        self.death = death

    def _get_laws(self, s):
        """Return the laws of the birth death element."""
        return [JumpLaw(s.birth, {s.n: 1}), JumpLaw(s.death * s.n, {s.n: -1})]


class MeanAnalysis(BaseAnalysis):
    """Mean of every series of a source."""

    def __init__(self, variables=None):
        self.variables = variables

    def _get_data_kind(self):
        return 'series'

    def _prepare(self, source):
        self.series_ = self._read_series(source, self.variables)

    def _run(self, inputs=None):
        self.mean_ = self.series_.values.mean(axis=0)


class DegreeAnalysis(BaseAnalysis):
    """Mean degree of the network of every frame."""

    def _get_data_kind(self):
        return 'network'

    def _prepare(self, source):
        self.graphs_ = self._read_networks(source)

    def _run(self, inputs=None):
        self.degree_ = np.asarray([2 * graph.number_of_edges() / graph.number_of_nodes() for graph in self.graphs_])


class PathAnalysis(BaseAnalysis):
    """Number of jumps of a discrete state."""

    def _get_data_kind(self):
        return 'path'

    def _prepare(self, source):
        self.path_ = self._read_path(source)

    def _run(self, inputs=None):
        self.jumps_ = self.path_.times.size - 1


class FieldAnalysis(BaseAnalysis):
    """Mean of every frame of a field."""

    def __init__(self, name=None):
        self.name = name

    def _get_data_kind(self):
        return 'field'

    def _prepare(self, source):
        self.frames_ = self._read_fields(source, self.name)

    def _run(self, inputs=None):
        self.mean_ = self.frames_.reshape(self.frames_.shape[0], -1).mean(axis=1)


class AgentAnalysis(BaseAnalysis):
    """Number of agents of the first frame."""

    def _get_data_kind(self):
        return 'agents'

    def _prepare(self, source):
        self.snapshots_ = self._read_snapshots(source)

    def _run(self, inputs=None):
        self.count_ = self.snapshots_.ids.size


def test_analysis_reads_a_trajectory():
    """Test that the series of a trajectory are read by name."""
    trajectory = run('lotka_volterra', settings={'until': 1.0}).trajectory_
    analysis = MeanAnalysis(['prey.n']).check().prepare(trajectory).run()
    assert analysis.series_.labels == ['prey.n']
    assert analysis.mean_.tolist() == pytest.approx([trajectory.values('prey.n').mean()])


def test_analysis_reads_arrays_as_columns():
    """Test that a series is one column and a two-dimensional array is one series per column."""
    assert MeanAnalysis().check().prepare([1.0, 3.0]).run().mean_.tolist() == [2.0]
    assert MeanAnalysis().check().prepare(np.array([[1.0, 2.0], [3.0, 4.0]])).run().mean_.tolist() == [2.0, 3.0]


def test_analysis_reads_a_graph():
    """Test that a graph is the only frame of a network analysis."""
    assert DegreeAnalysis().check().prepare(nx.cycle_graph(5)).run().degree_.tolist() == [2.0]


def test_missing_data_kind_names_the_kinds():
    """Test that reading agents from an ODE trajectory names the data kinds it has."""
    trajectory = run('lotka_volterra', settings={'until': 1.0}).trajectory_
    with pytest.raises(AnalysisError, match=r'no agent group.*Its data kinds are: series'):
        AgentAnalysis().check().prepare(trajectory)


def test_series_of_an_ode():
    """Test that the series of the prey and predators equal the trajectory's columns."""
    trajectory = run('lotka_volterra', settings={'until': 5.0}).trajectory_
    series = MeanAnalysis(['prey.n', 'predators.n']).check().prepare(trajectory).series_
    assert series.values.shape == (len(trajectory.times), 2)
    np.testing.assert_array_equal(series.values[:, 0], trajectory.values('prey.n'))
    np.testing.assert_array_equal(series.values[:, 1], trajectory.values('predators.n'))
    np.testing.assert_array_equal(series.times, trajectory.times)


def test_series_labels_of_every_scalar_series():
    """Test that the series of a trajectory default to its scalar series, labelled by name."""
    trajectory = run('lotka_volterra', settings={'until': 1.0}).trajectory_
    assert MeanAnalysis().check().prepare(trajectory).series_.labels == ['prey.n', 'predators.n']


def test_jump_path_of_a_gillespie_run():
    """Test that the jump path of a birth-death run keeps only the frames where the state changed."""
    system = System(elements=[BirthDeathElement('cells')])
    trajectory = Simulation(system, dt=0.01, random_state=0).check().prepare().run(until=20).trajectory_
    path = PathAnalysis().check().prepare(trajectory).path_
    assert path.label == 'cells.n'
    assert np.all(np.diff(path.states) != 0)
    assert np.all(np.abs(np.diff(path.states)) >= 1)
    assert path.times.size < len(trajectory.times)


def test_jump_path_of_an_array_of_states():
    """Test that a series of states keeps the first state and every change, and rejects a grid."""
    analysis = PathAnalysis().check().prepare(np.array([0, 0, 1, 1, 0])).run()
    assert analysis.path_.states.tolist() == [0, 1, 0]
    assert analysis.path_.times.tolist() == [0.0, 2.0, 4.0]
    with pytest.raises(AnalysisError, match='one series of states'):
        PathAnalysis().check().prepare(np.zeros((2, 2)))


def test_fields_of_a_lattice():
    """Test that the frames of a 64 by 64 Ising run of 100 frames have the lattice shape."""
    trajectory = run('ising', size=64, settings={'until': 99}).trajectory_
    frames = FieldAnalysis('spins.spin').check().prepare(trajectory).frames_
    assert frames.shape == (100, 64, 64)
    np.testing.assert_array_equal(frames, trajectory.lattice('spins.spin'))


def test_snapshots_unwrap_agent_positions():
    """Test that walkers in a periodic square of side 10 are unwrapped so no step exceeds half the side."""
    trajectory = run('run_and_tumble', n_agents=50, speed=1.0, settings={'until': 200}).trajectory_
    snapshots = AgentAnalysis().check().prepare(trajectory).snapshots_
    steps = np.linalg.norm(np.diff(snapshots.positions, axis=0), axis=-1)
    assert snapshots.periodic
    assert steps.max() < 5
    assert np.ptp(snapshots.positions) > 10
    assert snapshots.directions.shape == (201, 50)


def test_snapshots_of_particle_groups():
    """Test that a particle group gives its positions and velocities in its periodic box."""
    trajectory = run('ideal_gas', settings={'until': 0.2}).trajectory_
    snapshots = AgentAnalysis().check().prepare(trajectory).snapshots_
    assert snapshots.positions.shape == snapshots.velocities.shape
    assert snapshots.positions.shape[0] == len(trajectory.times)


def test_snapshots_of_scalar_positions():
    """Test that a scalar position and a velocity role give one dimension per frame and member."""
    trajectory = run('optimal_velocity', settings={'until': 0.5}).trajectory_
    snapshots = AgentAnalysis().check().prepare(trajectory).snapshots_
    assert snapshots.positions.ndim == 3
    assert snapshots.positions.shape == snapshots.velocities.shape
    assert snapshots.positions.shape[-1] == 1


def test_snapshots_stack_position_components():
    """Test that scalar states of the role position stack into dimensions, with a state of the role direction."""
    trajectory = run('light_seeking_robots', settings={'until': 0.5}).trajectory_
    snapshots = AgentAnalysis().check().prepare(trajectory).snapshots_
    assert snapshots.positions.shape[-1] == 2
    assert snapshots.directions is not None
    assert 'x' not in snapshots.states


def test_snapshots_refuse_a_group_without_a_position_role():
    """Test that a group with a state named `x` and no position role gives no snapshots."""
    trajectory = run('replicator_dynamics', settings={'until': 0.5}).trajectory_
    with pytest.raises(AnalysisError, match='agent group'):
        AgentAnalysis().check().prepare(trajectory)


@pytest.mark.parametrize(
    ('model', 'state'),
    [
        ('generalized_lotka_volterra', 'species.x'),
        ('hypercycles', 'soup.x'),
        ('replicator_dynamics', 'population.x'),
        ('closed_contour_cell_exchange', 'cells.x'),
        ('doubling_map', 'points.x'),
        ('skew_tent_map', 'points.x'),
        ('on_off_intermittency', 'variables.x'),
        ('mirollo_strogatz', 'oscillators.x'),
    ],
)
def test_series_of_group_states(model, state):
    """Test that the scalar states of a group without positions are series, one column per member."""
    trajectory = run(model, settings={'until': 1.0}).trajectory_
    series = MeanAnalysis().check().prepare(trajectory).run().series_
    np.testing.assert_array_equal(series.values, trajectory.values(state))
    assert series.labels[0] == f'{state}[0]'


def test_series_leave_out_particles_and_jump_paths():
    """Test that a group of particles or of jump paths has no series, while a single jump state keeps one."""
    particles = run('random_walk_with_memory', n_walkers=5, settings={'until': 1.0}).trajectory_
    paths = run('poisson_process', n_processes=2, settings={'until': 1.0}).trajectory_
    for trajectory in (particles, paths):
        with pytest.raises(AnalysisError, match='no scalar series'):
            MeanAnalysis().check().prepare(trajectory)
    single = Simulation(System(elements=[BirthDeathElement('a')])).check().prepare().run(until=1.0).trajectory_
    assert MeanAnalysis().check().prepare(single).series_.labels == ['a.n']


@pytest.mark.parametrize('model', ['jain_krishna', 'sequentially_linear_ecosystem'])
def test_series_leave_out_variable_groups(model):
    """Test that a variable group has no default series, its rows holding no one member, but can be read by name."""
    trajectory = run(model, settings={'until': 5.0}).trajectory_
    with pytest.raises(AnalysisError, match='no scalar series'):
        MeanAnalysis().check().prepare(trajectory)
    series = MeanAnalysis(variables=['species.x']).check().prepare(trajectory).series_
    np.testing.assert_array_equal(series.values, trajectory.values('species.x'))


def test_sites_are_read_as_series_and_not_as_particles():
    """Test that oscillators on fixed sites give no particle snapshots, and their sites are no series."""
    trajectory = run('nonlocal_ring_chimera', n_oscillators=20, R=3, settings={'until': 0.5}).trajectory_
    assert trajectory.problem.elements[0].roles == {'x': 'site', 'theta': 'phase'}
    with pytest.raises(AnalysisError, match='no agent group'):
        AgentAnalysis().check().prepare(trajectory)


def test_snapshots_of_phases():
    """Test that a state with the role phase gives the phases of the snapshots, not a direction or another state."""
    trajectory = run('swarmalators', settings={'until': 0.5}).trajectory_
    snapshots = AgentAnalysis().check().prepare(trajectory).snapshots_
    np.testing.assert_array_equal(snapshots.phases, trajectory.values('units.theta'))
    assert snapshots.directions is None
    assert 'theta' not in snapshots.states


def test_networks_of_a_network_space():
    """Test that a field on a network gives its graph at every frame."""
    trajectory = run('network_diffusion', settings={'until': 0.5}).trajectory_
    graphs = DegreeAnalysis().check().prepare(trajectory).graphs_
    assert len(graphs) == len(trajectory.times)
    assert nx.is_isomorphic(graphs[0], nx.karate_club_graph())


def test_networks_of_a_changing_network_snapshot():
    """Test that a snapshot of a network whose topology changes is read as its graph."""
    trajectory = run('coevolving_voter', settings={'until': 2.0}).trajectory_
    snapshot = trajectory.network(-1)
    graph = DegreeAnalysis().check().prepare(snapshot).graphs_[0]
    assert sorted(graph.nodes()) == snapshot.nodes.tolist()
    assert graph.number_of_edges() == snapshot.graph().number_of_edges()


def test_surface_holds_only_analysis_classes():
    """Test that the public names of the analysis package are the base and classes ending in `Analysis`."""
    for name in skcomplex.analysis.__all__:
        item = getattr(skcomplex.analysis, name)
        assert isinstance(item, type)
        assert name == 'BaseAnalysis' or (name.endswith('Analysis') and issubclass(item, BaseAnalysis))


def test_verbs_run_in_order():
    """Test that prepare needs check and run needs prepare."""
    with pytest.raises(NotCheckedError):
        MeanAnalysis().prepare([1.0])
    with pytest.raises(NotPreparedError, match='must be prepared'):
        MeanAnalysis().check().run()
