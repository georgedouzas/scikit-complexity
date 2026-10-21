"""Test the Rerun observer and the visuals."""

import sys

import networkx as nx
import numpy as np
import pytest
import rerun as rr
from sage.all__sagemath_symbolics import I

from skcomplex.core.base import Domain, VisualSpec, state
from skcomplex.core.compile import compile_system
from skcomplex.core.model import (
    AgentType,
    ConstantForce,
    Element,
    EuclideanSpace,
    Field,
    GridSpace,
    LatticeSpace,
    Particle,
    Rate,
    Rule,
    System,
)
from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.viz import ImageVisual, PointsVisual, RerunObserver, ScalarsVisual, Visual, resolve_visual
from skcomplex.models.biology.ecology import ExponentialGrowth, Population
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
    system = System(elements=[Population('a')], interactions=[ExponentialGrowth('g', target='a')])
    _run(system, RerunObserver('test', save_path=path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_rerun_observer_interval_thins_frames(tmp_path):
    """Test that only every n-th frame is logged."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', interval=10)
    system = System(elements=[Population('a')])
    _run(system, observer, until=0.99, dt=0.01)
    assert observer.logged_frames_ == 10


def test_scalars_visual_logs_nested_paths():
    """Test that scalar states are logged under the element path."""
    recording = LoggedPaths()
    system = System(systems=[System('meadow', elements=[Population('rabbits')])])
    simulation = Simulation(system).run(until=0.0)
    element = simulation.problem_.elements[0]
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    ScalarsVisual().log(recording, element, frame)
    assert recording.logs == [('meadow/rabbits/n', 'Scalars')]


@pytest.mark.parametrize(('n_dim', 'archetype'), [(3, 'Points3D'), (2, 'Points2D'), (1, 'Scalars')])
def test_points_visual_logs_positions(n_dim, archetype):
    """Test that positions are logged as points of their dimension."""
    system = System(
        elements=[Particle('ball')],
        interactions=[ConstantForce('push', target='ball', F=1.0)],
        space=EuclideanSpace(n_dim),
    )
    simulation = Simulation(system).run(until=0.0)
    recording = LoggedPaths()
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    PointsVisual(position='x').log(recording, simulation.problem_.elements[0], frame)
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
    system = System(elements=[Population('a')], interactions=[ExponentialGrowth('g', target='a')])
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
    _run(System(elements=[Population('a')]), observer, until=0.0)
    assert calls[0]['executable_path'] == str(viewer)
    assert calls[0]['recording'] is observer.recording_


def test_rerun_observer_spawns_viewer_from_path(monkeypatch, tmp_path):
    """Test that spawning falls back to the viewer on `PATH`."""
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'python'))
    calls = []
    monkeypatch.setattr(rr, 'spawn', lambda **kwargs: calls.append(kwargs))
    _run(System(elements=[Population('a')]), RerunObserver('test', spawn=True), until=0.0)
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
    with pytest.raises(NotImplementedError):
        Visual().log(None, None, None)


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
        elements=[Particle.group('gas', 3, v=[[0.1, 0.0], [1.0, 0.0], [2.0, 0.0]])],
        space=EuclideanSpace(2),
    )
    simulation = Simulation(system).run(until=0.0)
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    recording = LoggedPaths()
    visual = PointsVisual(position='x', color_by='v', color_range=(0.0, 2.0), states=())
    colors = []
    recording.log = lambda path, archetype: colors.append(archetype.colors.as_arrow_array().to_pylist())
    visual.log(recording, simulation.problem_.elements[0], frame)
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

    class Mortal(AgentType):
        """Agent that removes itself when its id is 1."""

        def laws(self, s):
            """Return the removal rule."""
            return [Rule(self.die)]

        def die(self, agent, model):
            """Remove agent 1."""
            if agent.id == 1:
                agent.remove()

    logged = _agent_logs(monkeypatch, tmp_path, System(elements=[Mortal.group('mortals', 3)], space=GridSpace(5, 5)), 1)
    batches = [archetype for path, archetype in logged if path == 'mortals']
    labels = [archetype.labels.as_arrow_array().to_pylist() for archetype in batches]
    assert '1' in labels[0]
    assert '1' not in labels[1]


class Turning(Element):
    """Complex amplitude that turns at unit frequency."""

    z: complex = state(1.0, domain=Domain.COMPLEX)

    def laws(self, s):
        """Return `dz/dt = i z`."""
        return [Rate(s.z, I * s.z)]


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
    system = System(elements=[Field('medium', Turning, LatticeSpace((8, 8)), values=values)])
    logged = _logged(monkeypatch, tmp_path, system)
    assert logged.count(('medium/z/modulus', 'Image')) == 3
    assert logged.count(('medium/z/phase', 'Image')) == 3
    assert ('medium/z', 'Image') not in logged


def test_complex_scalar_logs_real_and_imaginary_series(monkeypatch, tmp_path):
    """Test that a complex state of a single element logs its real and imaginary parts as series."""
    logged = _logged(monkeypatch, tmp_path, System(elements=[Turning('osc')]))
    assert logged.count(('osc/z/real', 'Scalars')) == 3
    assert logged.count(('osc/z/imag', 'Scalars')) == 3


def test_complex_field_has_modulus_and_phase_views():
    """Test that an image visual shows the modulus and the phase of a complex field in two views."""
    problem = compile_system(System(elements=[Field('medium', Turning, LatticeSpace((4, 4)))]))
    views = ImageVisual().views(problem.elements[0], problem)
    assert [str(view.origin) for view in views] == ['medium/z/modulus', 'medium/z/phase']
