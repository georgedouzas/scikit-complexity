"""Test the base of analyses: the binding of settings and the readers of trajectories, arrays, and graphs."""

import dataclasses
import inspect
import sys
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import NN

import skcomplex.core.analysis
from skcomplex.core.analysis import bind_analysis, lyapunov, motifs, network_measures, renormalization, rolling
from skcomplex.core.analysis._base import (
    _composite,
    _get_analysis_tags,
    _get_source_tags,
    _read_fields,
    _read_networks,
    _read_path,
    _read_series,
    _read_snapshots,
)
from skcomplex.core.analysis._discovery import ANALYSES
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import JumpLaw, RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError, TagError, ValidationError
from skcomplex.core.specs import AnalysisTags, ParameterSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.ecology import (
    GeneralizedLotkaVolterraSystem,
    LogisticGrowthSystem,
    LotkaVolterraSystem,
    SequentiallyLinearEcosystemSystem,
)
from skcomplex.systems.biology.evolution import HypercyclesSystem, JainKrishnaSystem, ReplicatorDynamicsSystem
from skcomplex.systems.biology.physiology import ClosedContourCellExchangeSystem
from skcomplex.systems.chemistry.pattern_formation import BrusselatorSystem
from skcomplex.systems.engineering.robotics import LightSeekingRobotsSystem
from skcomplex.systems.engineering.traffic import OptimalVelocitySystem
from skcomplex.systems.mathematics.dynamical_systems import DoublingMapSystem, SkewTentMapSystem
from skcomplex.systems.mathematics.networks import NetworkDiffusionSystem
from skcomplex.systems.mathematics.stochastic_processes import (
    OnOffIntermittencySystem,
    PoissonProcessSystem,
    RandomWalkWithMemorySystem,
    WeakeningOrnsteinUhlenbeckSystem,
)
from skcomplex.systems.physics.active_matter import RunAndTumbleSystem
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem, IsingSystem
from skcomplex.systems.physics.synchronization import (
    MirolloStrogatzSystem,
    NonlocalRingChimeraSystem,
    SwarmalatorsSystem,
)
from skcomplex.systems.social.opinion_dynamics import CoevolvingVoterSystem

from ...conftest import _queue_run


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


class PhasedElement(Element):
    """Agent with a state named `phase` that is no heading, and a state with the role of heading."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phase: Annotated[float, StateSpec()] = 3.0,
        bearing: Annotated[float, StateSpec(role='direction')] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phase = phase
        self.bearing = bearing

    def _get_laws(self, s):
        """Return no change."""
        return [RuleLaw(lambda agent, system: None)]


def test_analysis_reads_a_trajectory():
    """Test that the series of a trajectory are read by name."""
    trajectory = simulate(LotkaVolterraSystem(), until=1.0, random_state=0)
    series = _read_series(trajectory, ['prey.x'])
    assert series.labels == ['prey.x']
    assert series.values.mean(axis=0).tolist() == pytest.approx([trajectory.get_values('prey.x').mean()])


def test_analysis_reads_arrays_as_columns():
    """Test that a series is one column and a two-dimensional array is one series per column."""
    assert _read_series([1.0, 3.0]).values.mean(axis=0).tolist() == [2.0]
    assert _read_series(np.array([[1.0, 2.0], [3.0, 4.0]])).values.mean(axis=0).tolist() == [2.0, 3.0]


def test_analysis_reads_a_graph():
    """Test that a graph is the only frame of the networks of a source."""
    graphs = _read_networks(nx.cycle_graph(5))
    assert [2 * graph.number_of_edges() / graph.number_of_nodes() for graph in graphs] == [2.0]


def test_missing_data_kind_names_the_kinds():
    """Test that reading agents from an ODE trajectory names the data kinds it has."""
    trajectory = simulate(LotkaVolterraSystem(), until=1.0, random_state=0)
    with pytest.raises(AnalysisError, match=r'no agent group.*Its data kinds are: series'):
        _read_snapshots(trajectory)


def test_series_of_an_ode():
    """Test that the series of the prey and predators equal the trajectory's columns."""
    trajectory = simulate(LotkaVolterraSystem(), until=5.0, random_state=0)
    series = _read_series(trajectory, ['prey.x', 'predators.x'])
    assert series.values.shape == (len(trajectory.times), 2)
    np.testing.assert_array_equal(series.values[:, 0], trajectory.get_values('prey.x'))
    np.testing.assert_array_equal(series.values[:, 1], trajectory.get_values('predators.x'))
    np.testing.assert_array_equal(series.times, trajectory.times)


def test_series_labels_of_every_scalar_series():
    """Test that the series of a trajectory default to its scalar series, labelled by name."""
    trajectory = simulate(LotkaVolterraSystem(), until=1.0, random_state=0)
    assert _read_series(trajectory).labels == ['prey.x', 'predators.x']


def test_jump_path_of_a_gillespie_run():
    """Test that the jump path of a birth-death run keeps only the frames where the state changed."""
    system = CompositeSystem(elements=[BirthDeathElement('cells')])
    trajectory = simulate(system, until=20, dt=0.01, random_state=0)
    path = _read_path(trajectory)
    assert path.label == 'cells.n'
    assert np.all(np.diff(path.states) != 0)
    assert np.all(np.abs(np.diff(path.states)) >= 1)
    assert path.times.size < len(trajectory.times)


def test_jump_path_of_an_array_of_states():
    """Test that a series of states keeps the first state and every change, and rejects a grid."""
    path = _read_path(np.array([0, 0, 1, 1, 0]))
    assert path.states.tolist() == [0, 1, 0]
    assert path.times.tolist() == [0.0, 2.0, 4.0]
    with pytest.raises(AnalysisError, match='one series of states'):
        _read_path(np.zeros((2, 2)))


def test_fields_of_a_lattice():
    """Test that the frames of a 64 by 64 Ising run of 100 frames have the lattice shape."""
    trajectory = simulate(IsingSystem(size=64), until=99, random_state=0)
    frames = _read_fields(trajectory, 'spins.spin')
    assert frames.shape == (100, 64, 64)
    np.testing.assert_array_equal(frames, trajectory.get_lattice('spins.spin'))


def test_snapshots_unwrap_agent_positions():
    """Test that walkers in a periodic square of side 10 are unwrapped so no step exceeds half the side."""
    trajectory = simulate(RunAndTumbleSystem(n_agents=50, speed=1.0), until=200, random_state=0)
    snapshots = _read_snapshots(trajectory)
    steps = np.linalg.norm(np.diff(snapshots.positions, axis=0), axis=-1)
    assert snapshots.periodic
    assert steps.max() < 5
    assert np.ptp(snapshots.positions) > 10
    assert snapshots.directions.shape == (201, 50)


def test_snapshots_of_particle_groups():
    """Test that a particle group gives its positions and velocities in its periodic box."""
    trajectory = simulate(IdealGasSystem(), until=0.2, random_state=0)
    snapshots = _read_snapshots(trajectory)
    assert snapshots.positions.shape == snapshots.velocities.shape
    assert snapshots.positions.shape[0] == len(trajectory.times)


def test_snapshots_of_scalar_positions():
    """Test that a scalar position and a velocity role give one dimension per frame and member."""
    trajectory = simulate(OptimalVelocitySystem(), until=0.5, random_state=0)
    snapshots = _read_snapshots(trajectory)
    assert snapshots.positions.ndim == 3
    assert snapshots.positions.shape == snapshots.velocities.shape
    assert snapshots.positions.shape[-1] == 1


def test_snapshots_stack_position_components():
    """Test that scalar states of the role position stack into dimensions, with a state of the role direction."""
    trajectory = simulate(LightSeekingRobotsSystem(), until=0.5, random_state=0)
    snapshots = _read_snapshots(trajectory)
    assert snapshots.positions.shape[-1] == 2
    assert snapshots.directions is not None
    assert 'x' not in snapshots.states


def test_snapshots_refuse_a_group_without_a_position_role():
    """Test that a group with a state named `x` and no position role gives no snapshots."""
    trajectory = simulate(ReplicatorDynamicsSystem(), until=0.5, random_state=0)
    with pytest.raises(AnalysisError, match='agent group'):
        _read_snapshots(trajectory)


@pytest.mark.parametrize(
    ('system', 'state'),
    [
        (GeneralizedLotkaVolterraSystem, 'species.x'),
        (HypercyclesSystem, 'soup.x'),
        (ReplicatorDynamicsSystem, 'population.x'),
        (ClosedContourCellExchangeSystem, 'cells.x'),
        (DoublingMapSystem, 'points.x'),
        (SkewTentMapSystem, 'points.x'),
        (OnOffIntermittencySystem, 'variables.x'),
        (MirolloStrogatzSystem, 'oscillators.x'),
    ],
)
def test_series_of_group_states(system, state):
    """Test that the scalar states of a group without positions are series, one column per member."""
    trajectory = simulate(system(), until=1.0, random_state=0)
    series = _read_series(trajectory)
    np.testing.assert_array_equal(series.values, trajectory.get_values(state))
    assert series.labels[0] == f'{state}[0]'


def test_series_leave_out_particles_and_jump_paths():
    """Test that a group of particles or of jump paths has no series, while a single jump state keeps one."""
    particles = simulate(RandomWalkWithMemorySystem(n_walkers=5), until=1.0, random_state=0)
    paths = simulate(PoissonProcessSystem(n_processes=2), until=1.0, random_state=0)
    for trajectory in (particles, paths):
        with pytest.raises(AnalysisError, match='no scalar series'):
            _read_series(trajectory)
    single = simulate(CompositeSystem(elements=[BirthDeathElement('a')]), until=1.0)
    assert _read_series(single).labels == ['a.n']


@pytest.mark.parametrize('system', [JainKrishnaSystem, SequentiallyLinearEcosystemSystem])
def test_series_leave_out_variable_groups(system):
    """Test that a variable group has no default series, its rows holding no one member, but can be read by name."""
    trajectory = simulate(system(), until=5.0, random_state=0)
    with pytest.raises(AnalysisError, match='no scalar series'):
        _read_series(trajectory)
    series = _read_series(trajectory, ['species.x'])
    np.testing.assert_array_equal(series.values, trajectory.get_values('species.x'))


def test_sites_are_read_as_series_and_not_as_particles():
    """Test that oscillators on fixed sites give no particle snapshots, and their sites are no series."""
    trajectory = simulate(NonlocalRingChimeraSystem(n_oscillators=20, R=3), until=0.5, random_state=0)
    assert trajectory.problem.elements[0].roles == {'x': 'site', 'theta': 'phase'}
    with pytest.raises(AnalysisError, match='no agent group'):
        _read_snapshots(trajectory)


def test_snapshots_of_phases():
    """Test that a state with the role phase gives the phases of the snapshots, not a direction or another state."""
    trajectory = simulate(SwarmalatorsSystem(), until=0.5, random_state=0)
    snapshots = _read_snapshots(trajectory)
    np.testing.assert_array_equal(snapshots.phases, trajectory.get_values('units.theta'))
    assert snapshots.directions is None
    assert 'theta' not in snapshots.states


def test_networks_of_a_network_space():
    """Test that a field on a network gives its graph at every frame."""
    trajectory = simulate(NetworkDiffusionSystem(), until=0.5, random_state=0)
    graphs = _read_networks(trajectory)
    assert len(graphs) == len(trajectory.times)
    assert nx.is_isomorphic(graphs[0], nx.karate_club_graph())


def test_networks_of_a_changing_network_snapshot():
    """Test that a snapshot of a network whose topology changes is read as its graph."""
    trajectory = simulate(CoevolvingVoterSystem(), until=2.0, random_state=0)
    snapshot = trajectory.get_network(-1)
    graph = _read_networks(snapshot)[0]
    assert sorted(graph.nodes()) == snapshot.nodes.tolist()
    assert graph.number_of_edges() == snapshot.get_graph().number_of_edges()


def test_surface_holds_analysis_functions_and_results():
    """Test that the public names of the analysis package are functions, frozen results, and the binding helper."""
    for name in skcomplex.core.analysis.__all__:
        item = getattr(skcomplex.core.analysis, name)
        if name.endswith('Result'):
            assert dataclasses.is_dataclass(item)
            assert item.__dataclass_params__.frozen
        else:
            assert inspect.isfunction(item), name


def test_bound_analysis_applies_its_settings():
    """Test that a bound analysis keeps its function and settings and equals a call with them."""
    bound = bind_analysis(rolling, kind='variance', window=3)
    values = np.array([0.0, 1.0, 0.0, 1.0, 3.0])
    assert bound.func is rolling
    assert bound.keywords == {'kind': 'variance', 'window': 3}
    np.testing.assert_array_equal(bound(values).statistic, rolling(values, kind='variance', window=3).statistic)


def test_binding_rejects_unknown_settings_and_objects():
    """Test that binding a setting the analysis does not take, or an object that is no function, raises an error."""
    with pytest.raises(AnalysisError, match='does not take these settings'):
        bind_analysis(rolling, bins=3)
    with pytest.raises(AnalysisError, match='function of a source'):
        bind_analysis(3)


def test_a_system_class_is_read_as_the_composite_of_its_parts():
    """Test that a system class that builds its parts is read as its prepared composite, other sources as they are."""
    composite = _composite(BrusselatorSystem((8,)))
    assert isinstance(composite, CompositeSystem)
    values = np.zeros(3)
    assert _composite(values) is values


def test_every_analysis_declares_valid_tags():
    """Test that the module of every analysis declares its tags, and tags of kinds only for kinds of its family."""
    for function in ANALYSES:
        module = sys.modules[function.__module__]
        assert isinstance(module.TAGS, AnalysisTags), function.__name__
        module.TAGS._check()
        for kind, tags in getattr(module, 'KIND_TAGS', {}).items():
            assert kind in module.KINDS, (function.__name__, kind)
            tags._check()
            assert _get_analysis_tags(function, kind) is tags


def test_lyapunov_of_a_stochastic_system_names_the_randomness():
    """Test that the Lyapunov exponents of a stochastic system, or of a trajectory of one, raise an error naming it."""
    system = WeakeningOrnsteinUhlenbeckSystem(4)
    for source in (system, simulate(system, until=1.0, random_state=0)):
        with pytest.raises(TagError, match="lyapunov of kind 'spectrum' needs the randomness deterministic"):
            lyapunov(source)


def test_network_analyses_of_a_system_without_a_network_name_the_space():
    """Test that network analyses of a trajectory of a system without a network space raise an error naming it."""
    trajectory = simulate(LogisticGrowthSystem(), until=1.0)
    for analysis in (network_measures, motifs, renormalization):
        with pytest.raises(TagError, match=r'needs the space network, but the system has the space none'):
            analysis(trajectory)


def test_tags_are_not_checked_for_data_without_a_system():
    """Test that arrays and graphs carry no tags, so the analyses read them whatever they need."""
    assert _get_source_tags(np.zeros(4)) is None
    assert network_measures(nx.cycle_graph(4)).mean_degree.tolist() == [2.0]
    assert lyapunov([0.5, -1.0]).exponents.tolist() == [0.5, -1.0]


def test_series_analysis_refuses_a_state_of_objects_by_name():
    """Test that an analysis that needs numbers names the state that holds objects."""
    trajectory = _queue_run()
    with pytest.raises(ValidationError, match='The state orders of the agents of queues holds objects'):
        _read_series(trajectory, ['queues.orders'])


def test_snapshots_read_headings_from_roles_alone():
    """Test that a state named `phase` stays a state, and the state with the role is the heading."""
    group = GroupElement('walkers', element=PhasedElement, size=3)
    trajectory = simulate(CompositeSystem(elements=[group], time=DiscreteTime()), until=2, random_state=0)
    snapshots = _read_snapshots(trajectory, 'walkers')
    assert np.all(snapshots.directions == 0.5)
    assert np.all(snapshots.states['phase'] == 3.0)
