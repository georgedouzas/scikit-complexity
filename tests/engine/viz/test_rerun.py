"""Test the Rerun observer and the visuals."""

import sys
import time

import networkx as nx
import numpy as np
import pytest
import rerun as rr
from sage.all__sagemath_symbolics import I

from skcomplex.core.base import NumericDomain, VisualSpec, state
from skcomplex.core.components import ElementComponent, GroupComponent
from skcomplex.core.laws import RateLaw, UpdateLaw
from skcomplex.core.space import ContinuousSpace, DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Frame, RealTimeClock, Simulation
from skcomplex.engine.viz import (
    BaseVisual,
    ImageVisual,
    PointsVisual,
    RerunObserver,
    ScalarsVisual,
    SiteVisual,
    resolve_visual,
)
from skcomplex.formalisms.agents import RuleLaw
from skcomplex.formalisms.fields import FieldComponent
from skcomplex.formalisms.particles import ParticleComponent
from skcomplex.formalisms.topology import add_node
from skcomplex.models import create
from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationComponent
from skcomplex.models.computer_science.automata import game_of_life
from skcomplex.models.physics.realizations.particles import ConstantForceInteraction
from skcomplex.models.physics.statistical_mechanics import ideal_gas
from skcomplex.models.social.opinion_dynamics import voter_model
from skcomplex.models.social.segregation import schelling


class LoggedPaths:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self):
        """Start with no logs."""
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype, leaving out static styles."""
        if not static:
            self.logs.append((path, type(archetype).__name__))


def _run(system, observer, until=1.0, dt=0.1):
    """Run a system with a Rerun observer whose visuals log to a stand-in."""
    simulation = Simulation(system, dt=dt, observers=[observer])
    simulation.run(until=until)
    return simulation


def test_rerun_observer_saves_recording(tmp_path):
    """Test that a recording file is written."""
    path = tmp_path / 'run.rrd'
    system = System(components=[PopulationComponent('a')], interactions=[ExponentialGrowthInteraction('g', target='a')])
    _run(system, RerunObserver('test', save_path=path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_rerun_observers_release_the_threads_of_finished_recordings(tmp_path):
    """Test that many observers in one process, as in the docs build, do not accumulate recording threads."""
    psutil = pytest.importorskip('psutil')
    system = System(components=[PopulationComponent('a')], interactions=[ExponentialGrowthInteraction('g', target='a')])
    _run(system, RerunObserver('test', save_path=tmp_path / 'first.rrd'))
    before = psutil.Process().num_threads()
    for index in range(20):
        _run(system, RerunObserver('test', save_path=tmp_path / f'run{index}.rrd'))
    assert psutil.Process().num_threads() - before < 10


def test_rerun_observer_interval_thins_frames(tmp_path):
    """Test that only every n-th frame is logged."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', interval=10)
    system = System(components=[PopulationComponent('a')])
    _run(system, observer, until=0.99, dt=0.01)
    assert observer.logged_frames_ == 10


def test_scalars_visual_logs_nested_paths():
    """Test that scalar states are logged under the element path."""
    recording = LoggedPaths()
    system = System(components=[System('meadow', components=[PopulationComponent('rabbits')])])
    simulation = Simulation(system).run(until=0.0)
    element = simulation.problem_.elements[0]
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    ScalarsVisual().fit(element, frame.problem).run(frame, recording)
    assert recording.logs == [('meadow/rabbits/n', 'Scalars')]


@pytest.mark.parametrize(('n_dim', 'archetype'), [(3, 'Points3D'), (2, 'Points2D'), (1, 'Scalars')])
def test_points_visual_logs_positions(n_dim, archetype):
    """Test that positions are logged as points of their dimension."""
    system = System(
        components=[ParticleComponent('ball')],
        interactions=[ConstantForceInteraction('push', target='ball', F=1.0)],
        space=ContinuousSpace(n_dim=n_dim),
    )
    simulation = Simulation(system).run(until=0.0)
    recording = LoggedPaths()
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    PointsVisual(position='x').fit(simulation.problem_.elements[0], simulation.problem_).run(frame, recording)
    assert recording.logs[0][1] == archetype
    assert ('ball/v', 'Scalars') in recording.logs


def test_rerun_observer_logs_parameters(monkeypatch, tmp_path):
    """Test that parameters and states are logged every frame, besides static styles."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, *args, **kwargs):
        if not kwargs.get('static'):
            logged.append(path)
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    system = System(components=[PopulationComponent('a')], interactions=[ExponentialGrowthInteraction('g', target='a')])
    _run(system, RerunObserver('test', save_path=tmp_path / 'run.rrd'), until=0.2)
    assert logged.count('parameters/g.r') == 3
    assert logged.count('a/n') == 3


def test_rerun_observer_spawns_viewer_next_to_interpreter(monkeypatch, tmp_path):
    """Test that spawning uses the viewer installed with the running interpreter."""
    viewer = tmp_path / 'rerun'
    viewer.touch()
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'python'))
    calls = []
    monkeypatch.setattr(rr, 'spawn', lambda **kwargs: calls.append(kwargs))
    observer = RerunObserver('test', spawn=True)
    _run(System(components=[PopulationComponent('a')]), observer, until=0.0)
    assert calls[0]['executable_path'] == str(viewer)
    assert calls[0]['recording'] is observer.recording_


def test_rerun_observer_spawns_viewer_from_path(monkeypatch, tmp_path):
    """Test that spawning falls back to the viewer on `PATH`."""
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'python'))
    calls = []
    monkeypatch.setattr(rr, 'spawn', lambda **kwargs: calls.append(kwargs))
    _run(System(components=[PopulationComponent('a')]), RerunObserver('test', spawn=True), until=0.0)
    assert calls[0]['executable_path'] is None


def test_resolve_visual_builds_registered_visuals():
    """Test that visual specs resolve to visuals with their options."""
    visual = resolve_visual(VisualSpec('points', {'position': 'x', 'radius': 0.2}))
    assert isinstance(visual, PointsVisual)
    assert visual.radius == 0.2
    with pytest.raises(KeyError, match='unknown'):
        resolve_visual(VisualSpec('unknown'))


def test_visual_base_log_is_abstract():
    """Test that the base visual must be subclassed."""
    simulation = Simulation(create('lotka_volterra')).run(until=0.0)
    problem = simulation.problem_
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, problem)
    with pytest.raises(NotImplementedError):
        BaseVisual().fit(problem.elements[0], problem).run(frame, None)


def test_group_is_logged_as_one_batch(monkeypatch, tmp_path):
    """Test that a group of 10000 members is logged as one Points2D per frame."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, archetype, *args, **kwargs):
        logged.append((path, type(archetype).__name__, getattr(archetype, 'positions', None)))
        return original(self, path, archetype, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    observer = RerunObserver('test', save_path=tmp_path / 'gas.rrd')
    Simulation(ideal_gas(10_000, random_state=0), dt=0.01, observers=[observer]).run(until=0.02)
    batches = [entry for entry in logged if entry[0] == 'gas']
    assert len(batches) == 3
    assert all(name == 'Points2D' and len(positions) == 10_000 for _, name, positions in batches)
    assert not [entry for entry in logged if entry[0].startswith('gas/')]


def test_points_colored_by_speed_follow_the_color_scale():
    """Test that faster members get colors further along the color scale."""
    system = System(
        components=[ParticleComponent.group('gas', 3, v=[[0.1, 0.0], [1.0, 0.0], [2.0, 0.0]])],
        space=ContinuousSpace(n_dim=2),
    )
    simulation = Simulation(system).run(until=0.0)
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    recording = LoggedPaths()
    visual = PointsVisual(position='x', color_by='v', color_range=(0.0, 2.0), states=())
    visual.fit(simulation.problem_.elements[0], simulation.problem_)
    colors = []
    recording.log = lambda path, archetype: colors.append(archetype.colors.as_arrow_array().to_pylist())
    visual.run(frame, recording)
    rgb = [[(value >> shift) & 255 for shift in (24, 16, 8)] for value in colors[0]]
    assert rgb[0][2] > rgb[0][1]
    assert rgb[2][1] > rgb[0][1]
    assert rgb[2][0] > rgb[1][0]


def _agent_logs(monkeypatch, tmp_path, system, until=2):
    """Return the entity paths and archetypes logged while an agent model runs."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, archetype, *args, **kwargs):
        logged.append((path, archetype))
        return original(self, path, archetype, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    Simulation(system, observers=[RerunObserver('agents', save_path=tmp_path / 'agents.rrd')], random_state=0).run(
        until=until,
    )
    return logged


def test_schelling_grid_logs_one_batch_per_frame(monkeypatch, tmp_path):
    """Test that Schelling residents are logged as one Points2D batch per frame at their cells."""
    logged = _agent_logs(monkeypatch, tmp_path, schelling(20, 20, random_state=0))
    batches = [archetype for path, archetype in logged if path == 'residents']
    assert len(batches) == 3
    assert all(type(archetype).__name__ == 'Points2D' for archetype in batches)
    assert len(batches[0].positions.as_arrow_array()) == 320


def test_voter_network_logs_graph_with_opinion_colors(monkeypatch, tmp_path):
    """Test that a voter network is logged as graph edges once and colored graph nodes every frame."""
    logged = _agent_logs(monkeypatch, tmp_path, voter_model(nx.cycle_graph(10), random_state=0))
    kinds = [type(archetype).__name__ for path, archetype in logged if path == 'voters']
    assert kinds.count('GraphEdges') == 1
    assert kinds.count('GraphNodes') == 3
    nodes = next(archetype for path, archetype in logged if type(archetype).__name__ == 'GraphNodes')
    assert nodes.colors is not None


def test_removed_agents_disappear_from_logs(monkeypatch, tmp_path):
    """Test that an agent removed between frames is absent from the next batch."""

    class MortalComponent(ElementComponent):
        """Agent that removes itself when its id is 1."""

        def laws(self, s):
            """Return the removal rule."""
            return [RuleLaw(self.die)]

        def die(self, agent, model):
            """Remove agent 1."""
            if agent.id == 1:
                agent.remove()

    logged = _agent_logs(
        monkeypatch,
        tmp_path,
        System(
            components=[MortalComponent.group('mortals', 3, variable=True)],
            space=LatticeSpace((5, 5), boundary='fixed'),
        ),
        1,
    )
    batches = [archetype for path, archetype in logged if path == 'mortals']
    labels = [archetype.labels.as_arrow_array().to_pylist() for archetype in batches]
    assert '1' in labels[0]
    assert '1' not in labels[1]


class TurningComponent(ElementComponent):
    """Complex amplitude that turns at unit frequency."""

    z: complex = state(1.0, domain=NumericDomain.COMPLEX)

    def laws(self, s):
        """Return `dz/dt = i z`."""
        return [RateLaw(s.z, I * s.z)]


def _logged(monkeypatch, tmp_path, system):
    """Return the entity paths and archetypes logged to a saved recording while a system runs for one time unit."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, archetype, *args, **kwargs):
        logged.append((path, type(archetype).__name__))
        return original(self, path, archetype, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    Simulation(system, dt=0.5, observers=[observer]).run(until=1.0)
    return logged


def test_complex_field_logs_modulus_and_phase_images(monkeypatch, tmp_path):
    """Test that a complex field at `medium` logs a modulus and a phase image for every logged frame."""
    values = {'z': np.exp(1j * np.linspace(-np.pi, np.pi, 64)).reshape(8, 8)}
    system = System(components=[FieldComponent('medium', TurningComponent, LatticeSpace((8, 8)), values=values)])
    logged = _logged(monkeypatch, tmp_path, system)
    assert logged.count(('medium/z/modulus', 'Image')) == 3
    assert logged.count(('medium/z/phase', 'Image')) == 3
    assert ('medium/z', 'Image') not in logged


def test_complex_scalar_logs_real_and_imaginary_series(monkeypatch, tmp_path):
    """Test that a complex state of a single element logs its real and imaginary parts as series."""
    logged = _logged(monkeypatch, tmp_path, System(components=[TurningComponent('osc')]))
    assert logged.count(('osc/z/real', 'Scalars')) == 3
    assert logged.count(('osc/z/imag', 'Scalars')) == 3


def test_complex_field_has_modulus_and_phase_views():
    """Test that an image visual shows the modulus and the phase of a complex field in two views."""
    problem = compile_system(System(components=[FieldComponent('medium', TurningComponent, LatticeSpace((4, 4)))]))
    views = ImageVisual().fit(problem.elements[0], problem).views_
    assert [str(view.origin) for view in views] == ['medium/z/modulus', 'medium/z/phase']


def _logged_paths(monkeypatch, tmp_path, system, until, **settings):
    """Return the entity paths logged while a system runs with a Rerun observer, leaving out static styles."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, *args, **kwargs):
        if not kwargs.get('static'):
            logged.append(path)
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    observer = RerunObserver('paths', save_path=tmp_path / 'paths.rrd')
    Simulation(system, observers=[observer], **settings).run(until=until)
    return logged, observer


def test_rerun_observer_logs_stock_flow_auxiliaries_under_their_element(monkeypatch, tmp_path):
    """Test that an auxiliary of a stock and flow model is logged under its element, so its view is not blank."""
    logged, _ = _logged_paths(monkeypatch, tmp_path, create('hilsa_fishery'), until=10, dt=1)
    assert 'hilsa/adult_weight' in logged
    assert 'hilsa.adult_weight' not in logged


def test_rerun_observer_logs_extra_entities_in_their_own_sanitised_namespace(monkeypatch, tmp_path):
    """Test that entities a user logs go under `extra/`, with spaces replaced so Rerun needs no escapes."""
    logged, observer = _logged_paths(monkeypatch, tmp_path, create('lotka_volterra'), until=1, dt=0.5)
    observer._log_extra('chain angle/left side', rr.Scalars(1.0))
    assert logged[-1] == 'extra/chain_angle/left_side'


class _AllLogs:
    """Recording stand-in that keeps every entity path and archetype it receives."""

    def __init__(self):
        """Start with no logs."""
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype."""
        self.logs.append((path, archetype))


def test_changing_network_logs_present_nodes_and_changed_edges():
    """Test that a growing network logs the nodes present at every frame and its edges at every frame they change."""
    NodeComponent = ElementComponent.define(
        'NodeComponent',
        states={'age': 0.0},
        laws=lambda s: [UpdateLaw(s.age, s.age + 1)],
    )
    LinkComponent = ElementComponent.define('LinkComponent')
    GrowthComponent = ElementComponent.define('GrowthComponent', laws=lambda s: [UpdateLaw(add_node(edges=1), 1)])
    space = NetworkSpace.complete(2)
    system = System(
        components=[
            GroupComponent('nodes', element=NodeComponent, space=space, on='sites', variable=True, capacity=10),
            GroupComponent('links', element=LinkComponent, space=space, on='edges', variable=True, capacity=10),
            GrowthComponent('growth'),
        ],
        time=DiscreteTime(),
    )
    simulation = Simulation(system, dt=1.0, random_state=0, record=True).run(until=3)
    trajectory, problem = simulation.trajectory_, simulation.problem_
    element = next(item for item in problem.elements if item.path == 'nodes')
    visual, recording = SiteVisual().fit(element, problem), _AllLogs()
    for index in range(len(trajectory)):
        frame = Frame(index, trajectory.times[index], trajectory.states[index], trajectory.parameters[index], problem)
        visual.run(frame, recording)
    nodes = [archetype for _, archetype in recording.logs if type(archetype).__name__ == 'GraphNodes']
    edges = [archetype for _, archetype in recording.logs if type(archetype).__name__ == 'GraphEdges']
    assert [len(item.node_ids.as_arrow_array()) for item in nodes] == [2, 3, 4, 5]
    assert len(edges) == 4
    assert [type(view).__name__ for view in visual.views_] == ['GraphView']


@pytest.mark.benchmark
def test_large_game_of_life_runs_in_real_time(tmp_path):
    """Test that a 1000 x 1000 Game of Life on Numba sustains 30 steps per second while streaming images."""
    observer = RerunObserver('life', save_path=tmp_path / 'life.rrd')
    simulation = Simulation(game_of_life((1000, 1000)), dt=1, backend='numba', record=False, observers=[observer])
    simulation.run(until=2)
    simulation.observers = [RerunObserver('life', save_path=tmp_path / 'paced.rrd')]
    simulation.clock = RealTimeClock(30.0)
    start = time.perf_counter()
    simulation.run(until=90)
    assert 90 / (time.perf_counter() - start) >= 29.0
    assert simulation.problem_.lattice[0].fused
    assert simulation.observers[0].logged_frames_ == 91
