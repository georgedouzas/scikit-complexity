"""Test the Rerun observer and the visuals."""

import sys
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
from sage.all__sagemath_symbolics import CC, I

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.laws import RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import (
    AgentsVisual,
    BaseVisual,
    ImageVisual,
    PointsVisual,
    RerunObserver,
    ScalarsVisual,
    SiteVisual,
)
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, TopologyRequestSpec, VisualSpec
from skcomplex.systems.biology.collective_behaviour import BoidsSystem
from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.systems.biology.ecology import HigherOrderRandomLotkaVolterraSystem, LotkaVolterraSystem, WolfSheepSystem
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.biology.evolution import MoranProcessSystem
from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
from skcomplex.systems.economics.resources import HilsaFisherySystem
from skcomplex.systems.physics.common.particles import ConstantForceInteraction, MassPointElement
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
from skcomplex.systems.social.opinion_dynamics import VoterSystem
from skcomplex.systems.social.segregation import SchellingSystem

from .conftest import _prepared, _views, default_blueprint


class LoggedPaths:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self):
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype, leaving out static styles."""
        if not static:
            self.logs.append((path, type(archetype).__name__))


def _run(system, observer, until=1.0, dt=0.1):
    """Return the trajectory of a system run with a Rerun observer."""
    return simulate(system, until=until, dt=dt, observers=[observer])


def primary_view(system):
    """Return the primary view of a system or a compiled problem."""
    return _prepared(system).views_[0]


def _whole_run_series(blueprint):
    """Return whether every time series view of a blueprint shows the time range of the whole run."""
    series = [view for view in _views(blueprint) if isinstance(view, rrb.TimeSeriesView)]
    return bool(series) and all('TimeAxis' in view.properties for view in series)


def test_rerun_observer_saves_recording(tmp_path):
    """Test that a recording file is written."""
    path = tmp_path / 'run.rrd'
    system = CompositeSystem(
        elements=[PopulationElement('a')],
        interactions=[ExponentialGrowthInteraction('g', target='a')],
    )
    _run(system, RerunObserver('test', save_path=path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_rerun_observers_release_the_threads_of_finished_recordings(tmp_path):
    """Test that many observers in one process, as in the docs build, do not accumulate recording threads."""
    psutil = pytest.importorskip('psutil')
    system = CompositeSystem(
        elements=[PopulationElement('a')],
        interactions=[ExponentialGrowthInteraction('g', target='a')],
    )
    _run(system, RerunObserver('test', save_path=tmp_path / 'first.rrd'))
    before = psutil.Process().num_threads()
    for index in range(20):
        _run(system, RerunObserver('test', save_path=tmp_path / f'run{index}.rrd'))
    assert psutil.Process().num_threads() - before < 10


def test_rerun_observer_interval_thins_frames(tmp_path):
    """Test that only every n-th frame is logged."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', interval=10)
    system = CompositeSystem(elements=[PopulationElement('a')])
    _run(system, observer, until=0.99, dt=0.01)
    assert observer.logged_frames_ == 10


def test_scalars_visual_logs_nested_paths():
    """Test that scalar states are logged under the element path."""
    recording = LoggedPaths()
    system = CompositeSystem(systems=[CompositeSystem('meadow', elements=[PopulationElement('rabbits')])])
    trajectory = simulate(system, until=0.0)
    element = trajectory.problem.elements[0]
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], trajectory.problem)
    ScalarsVisual(path=element.path, recording=recording).prepare(frame.problem).run(frame)
    assert recording.logs == [('meadow/rabbits/x', 'Scalars')]


@pytest.mark.parametrize(('n_dim', 'archetype'), [(3, 'Points3D'), (2, 'Points2D'), (1, 'Scalars')])
def test_points_visual_logs_positions(n_dim, archetype):
    """Test that positions are logged as points of their dimension."""
    system = CompositeSystem(
        elements=[MassPointElement('ball')],
        interactions=[ConstantForceInteraction('push', target='ball', F=1.0)],
        space=ContinuousSpace(n_dim=n_dim),
    )
    trajectory = simulate(system, until=0.0)
    recording = LoggedPaths()
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], trajectory.problem)
    (
        PointsVisual(position='x', path=trajectory.problem.elements[0].path, recording=recording)
        .prepare(trajectory.problem)
        .run(frame)
    )
    assert recording.logs[0][1] == archetype
    assert ('ball/x_d1', 'Scalars') in recording.logs


def test_rerun_observer_logs_parameters(monkeypatch, tmp_path):
    """Test that parameters and states are logged every frame, besides static styles."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, *args, **kwargs):
        if not kwargs.get('static'):
            logged.append(path)
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    system = CompositeSystem(
        elements=[PopulationElement('a')],
        interactions=[ExponentialGrowthInteraction('g', target='a')],
    )
    _run(system, RerunObserver('test', save_path=tmp_path / 'run.rrd'), until=0.2)
    assert logged.count('parameters/g.r') == 3
    assert logged.count('a/x') == 3


def test_rerun_observer_leaves_out_a_coupling_tensor(monkeypatch, tmp_path):
    """Test that a parameter of order 3 is not logged, while the parameters up to a matrix are."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, *args, **kwargs):
        if not kwargs.get('static'):
            logged.append(path)
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    _run(
        HigherOrderRandomLotkaVolterraSystem(4).prepare().system_,
        RerunObserver('test', save_path=tmp_path / 'run.rrd'),
        until=0.2,
    )
    assert 'parameters/triplets.a' not in logged
    assert 'parameters/growth.r' in logged


def test_rerun_observer_spawns_viewer_next_to_interpreter(monkeypatch, tmp_path):
    """Test that spawning uses the viewer installed with the running interpreter."""
    monkeypatch.setenv('DISPLAY', ':0')
    viewer = tmp_path / 'rerun'
    viewer.touch()
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'python'))
    calls = []
    monkeypatch.setattr(rr, 'spawn', lambda **kwargs: calls.append(kwargs))
    observer = RerunObserver('test', spawn=True)
    _run(CompositeSystem(elements=[PopulationElement('a')]), observer, until=0.0)
    assert calls[0]['executable_path'] == str(viewer)
    assert calls[0]['recording'] is observer.recording_


def test_rerun_observer_spawns_viewer_from_path(monkeypatch, tmp_path):
    """Test that spawning falls back to the viewer on `PATH`."""
    monkeypatch.setenv('DISPLAY', ':0')
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'python'))
    calls = []
    monkeypatch.setattr(rr, 'spawn', lambda **kwargs: calls.append(kwargs))
    _run(CompositeSystem(elements=[PopulationElement('a')]), RerunObserver('test', spawn=True), until=0.0)
    assert calls[0]['executable_path'] is None


def test_rerun_observer_warns_when_a_viewer_cannot_be_reached(monkeypatch):
    """Test that a run streaming to a viewer waits for a limited time, then warns and keeps its results."""
    timeouts = []

    def flush(_, *, timeout_sec):
        timeouts.append(timeout_sec)
        error_msg = 'unable to connect'
        raise RuntimeError(error_msg)

    monkeypatch.setenv('DISPLAY', ':0')
    monkeypatch.setattr(rr, 'spawn', lambda **_: None)
    monkeypatch.setattr(rr.RecordingStream, 'flush', flush)
    with pytest.warns(UserWarning, match='unable to connect'):
        trajectory = _run(
            CompositeSystem(elements=[PopulationElement('a')]),
            RerunObserver('test', spawn=True),
            until=0.0,
        )
    assert timeouts == [pytest.approx(30.0)]
    assert trajectory.get_values('a.x').size > 0


def test_rerun_observer_spawns_no_viewer_without_a_display(monkeypatch):
    """Test that a Linux machine without a display warns and runs without a viewer."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    for name in ('DISPLAY', 'WAYLAND_DISPLAY', 'WAYLAND_SOCKET'):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(rr, 'spawn', lambda **kwargs: calls.append(kwargs))
    with pytest.warns(UserWarning, match='needs a display'):
        trajectory = _run(
            CompositeSystem(elements=[PopulationElement('a')]),
            RerunObserver('test', spawn=True),
            until=0.0,
        )
    assert calls == []
    assert trajectory.get_values('a.x').size > 0


def test_rerun_observer_waits_for_a_file_without_limit(monkeypatch, tmp_path):
    """Test that a run saving to a file only waits for the whole recording."""
    calls = []
    monkeypatch.setattr(rr.RecordingStream, 'flush', lambda _, **kwargs: calls.append(kwargs))
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    _run(CompositeSystem(elements=[PopulationElement('a')]), observer, until=0.0)
    assert calls == [{}]


def _gas(visual):
    """Return a compiled gas of two particles drawn with a visual spec."""
    gas = GroupElement('gas', element=MassPointElement, size=2, visual=visual)
    return simulate(CompositeSystem(elements=[gas], space=ContinuousSpace(n_dim=2)), until=0).problem


def test_rerun_observer_builds_the_visuals_the_specs_name():
    """Test that the observer builds the visual a spec names with its options, its path, and its recording."""
    observer = RerunObserver().prepare(_gas(VisualSpec('points', {'position': 'x', 'radius': 0.2})))
    visual = observer.visuals_[0]
    assert isinstance(visual, PointsVisual)
    assert (visual.radius, visual.path, visual.recording_) == (0.2, 'gas', observer.recording_)
    with pytest.raises(KeyError, match='unknown'):
        RerunObserver().prepare(_gas(VisualSpec('unknown')))


def test_visual_base_log_is_abstract():
    """Test that the base visual must be subclassed."""
    trajectory = simulate(LotkaVolterraSystem().prepare().system_, until=0.0)
    problem = trajectory.problem
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], problem)
    recording = rr.RecordingStream('test_visual_base_log_is_abstract')
    with pytest.raises(NotImplementedError):
        BaseVisual(path=problem.elements[0].path, recording=recording).prepare(problem).run(frame)


def test_group_is_logged_as_one_batch(monkeypatch, tmp_path):
    """Test that a group of 10000 members is logged as one Points2D per frame."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, archetype, *args, **kwargs):
        logged.append((path, type(archetype).__name__, getattr(archetype, 'positions', None)))
        return original(self, path, archetype, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    observer = RerunObserver('test', save_path=tmp_path / 'gas.rrd')
    simulate(IdealGasSystem(10_000, random_state=0).prepare().system_, until=0.02, dt=0.01, observers=[observer])
    batches = [entry for entry in logged if entry[0] == 'gas']
    assert len(batches) == 3
    assert all(name == 'Points2D' and len(positions) == 10_000 for _, name, positions in batches)
    assert not [entry for entry in logged if entry[0].startswith('gas/')]


def test_points_colored_by_speed_follow_the_color_scale():
    """Test that faster members get colors further along the color scale."""
    system = CompositeSystem(
        elements=[
            GroupElement(
                'gas',
                element=MassPointElement,
                size=3,
                values={'x_d1': [[0.1, 0.0], [1.0, 0.0], [2.0, 0.0]]},
            ),
        ],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = simulate(system, until=0.0)
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], trajectory.problem)
    recording = LoggedPaths()
    visual = PointsVisual(
        position='x',
        color_by='x_d1',
        color_range=(0.0, 2.0),
        states=(),
        path='gas',
        recording=recording,
    )
    visual.prepare(trajectory.problem)
    colors = []
    recording.log = lambda path, archetype: colors.append(archetype.colors.as_arrow_array().to_pylist())
    visual.run(frame)
    rgb = [[(value >> shift) & 255 for shift in (24, 16, 8)] for value in colors[0]]
    assert rgb[0][2] > rgb[0][1]
    assert rgb[2][1] > rgb[0][1]
    assert rgb[2][0] > rgb[1][0]


def _agent_logs(monkeypatch, tmp_path, system, until=2):
    """Return the entity paths and archetypes logged while an agent system runs."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, archetype, *args, **kwargs):
        logged.append((path, archetype))
        return original(self, path, archetype, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    simulate(
        system,
        until=until,
        observers=[RerunObserver('agents', save_path=tmp_path / 'agents.rrd')],
        random_state=0,
    )
    return logged


def test_schelling_grid_logs_one_batch_per_frame(monkeypatch, tmp_path):
    """Test that Schelling residents are logged as one Points2D batch per frame at their cells."""
    logged = _agent_logs(monkeypatch, tmp_path, SchellingSystem(20, 20, random_state=0).prepare().system_)
    batches = [archetype for path, archetype in logged if path == 'residents']
    assert len(batches) == 3
    assert all(type(archetype).__name__ == 'Points2D' for archetype in batches)
    assert len(batches[0].positions.as_arrow_array()) == 320


def test_voter_network_logs_graph_with_opinion_colors(monkeypatch, tmp_path):
    """Test that a voter network is logged as graph edges once and colored graph nodes every frame."""
    logged = _agent_logs(monkeypatch, tmp_path, VoterSystem(nx.cycle_graph(10), random_state=0).prepare().system_)
    kinds = [type(archetype).__name__ for path, archetype in logged if path == 'voters']
    assert kinds.count('GraphEdges') == 1
    assert kinds.count('GraphNodes') == 3
    nodes = next(archetype for path, archetype in logged if type(archetype).__name__ == 'GraphNodes')
    assert nodes.colors is not None


def test_removed_agents_disappear_from_logs(monkeypatch, tmp_path):
    """Test that an agent removed between frames is absent from the next batch."""

    class MortalElement(Element):
        """Agent that removes itself when its id is 1."""

        def _get_laws(self, s):
            """Return the removal rule."""
            return [RuleLaw(self.die)]

        def die(self, agent, system):
            """Remove agent 1."""
            if agent.id == 1:
                agent.remove()

    logged = _agent_logs(
        monkeypatch,
        tmp_path,
        CompositeSystem(
            elements=[GroupElement('mortals', element=MortalElement, size=3, variable=True)],
            space=LatticeSpace((5, 5), boundary='fixed'),
        ),
        1,
    )
    batches = [archetype for path, archetype in logged if path == 'mortals']
    labels = [archetype.labels.as_arrow_array().to_pylist() for archetype in batches]
    assert '1' in labels[0]
    assert '1' not in labels[1]


class TurningElement(Element):
    """Complex amplitude that turns at unit frequency."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[complex, StateSpec(domain=CC)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z

    def _get_laws(self, s):
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
    simulate(system, until=1.0, dt=0.5, observers=[observer])
    return logged


def test_complex_field_logs_modulus_and_phase_images(monkeypatch, tmp_path):
    """Test that a complex field at `medium` logs a modulus and a phase image for every logged frame."""
    values = {'z': np.exp(1j * np.linspace(-np.pi, np.pi, 64)).reshape(8, 8)}
    system = CompositeSystem(
        elements=[FieldElement('medium', element=TurningElement, space=LatticeSpace((8, 8)), values=values)],
    )
    logged = _logged(monkeypatch, tmp_path, system)
    assert logged.count(('medium/z/modulus', 'Image')) == 3
    assert logged.count(('medium/z/phase', 'Image')) == 3
    assert ('medium/z', 'Image') not in logged


def test_complex_scalar_logs_real_and_imaginary_series(monkeypatch, tmp_path):
    """Test that a complex state of a single element logs its real and imaginary parts as series."""
    logged = _logged(monkeypatch, tmp_path, CompositeSystem(elements=[TurningElement('osc')]))
    assert logged.count(('osc/z/real', 'Scalars')) == 3
    assert logged.count(('osc/z/imag', 'Scalars')) == 3


def test_complex_field_has_modulus_and_phase_views():
    """Test that an image visual shows the modulus and the phase of a complex field in two views."""
    system = CompositeSystem(elements=[FieldElement('medium', element=TurningElement, space=LatticeSpace((4, 4)))])
    problem = PassCompiler().prepare(system).run().problem_
    views = ImageVisual(path=problem.elements[0].path).prepare(problem).views_
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
    simulate(system, until=until, observers=[observer], **settings)
    return logged, observer


def test_rerun_observer_logs_observables_of_elements_under_their_element(monkeypatch, tmp_path):
    """Test that an observable of an element of a stock and flow system is logged under its element, not blank."""
    system = HilsaFisherySystem().prepare().system_
    observables = HilsaFisherySystem().get_run_settings()['observables']
    logged, _ = _logged_paths(monkeypatch, tmp_path, system, until=10, dt=1, observables=observables)
    assert 'hilsa/adult_weight' in logged
    assert 'hilsa.adult_weight' not in logged


def test_rerun_observer_logs_extra_entities_in_their_own_sanitised_namespace(monkeypatch, tmp_path):
    """Test that entities a user logs go under `extra/`, with spaces replaced so Rerun needs no escapes."""
    logged, observer = _logged_paths(monkeypatch, tmp_path, LotkaVolterraSystem().prepare().system_, until=1, dt=0.5)
    observer._log_extra('chain angle/left side', rr.Scalars(1.0))
    assert logged[-1] == 'extra/chain_angle/left_side'


class _AllLogs:
    """Recording stand-in that keeps every entity path and archetype it receives."""

    def __init__(self):
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype."""
        self.logs.append((path, archetype))


@pytest.mark.parametrize('on_system', [False, True])
def test_changing_network_logs_present_nodes_and_changed_edges(on_system):
    """Test that a growing network logs its nodes and changed edges in a graph view, with its space on either owner."""

    class NodeElement(Element):
        """Node element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            age: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.age = age

        def _get_laws(self, s):
            """Return the laws of the node element."""
            return [UpdateLaw(s.age, s.age + 1)]

    class GrowthElement(Element):
        """Growth element."""

        def _get_laws(self, s):
            """Return the laws of the growth element."""
            return [UpdateLaw(TopologyRequestSpec('add-node', edges=1), 1)]

    space = NetworkSpace(nx.complete_graph(2))
    owned = {} if on_system else {'space': space}
    system = CompositeSystem(
        elements=[
            GroupElement('nodes', element=NodeElement, on='sites', variable=True, capacity=10, **owned),
            GroupElement('links', element=Element, on='edges', variable=True, capacity=10, **owned),
            GrowthElement('growth'),
        ],
        space=space if on_system else None,
        time=DiscreteTime(),
    )
    simulation = simulate(system, until=3, dt=1.0, random_state=0)
    trajectory, problem = simulation, simulation.problem
    element = next(item for item in problem.elements if item.path == 'nodes')
    recording = _AllLogs()
    visual = SiteVisual(path=element.path, recording=recording).prepare(problem)
    for index in range(len(trajectory.times)):
        frame = FrameSpec(
            index,
            trajectory.times[index],
            trajectory.states[index],
            trajectory.parameters[index],
            problem,
        )
        visual.run(frame)
    nodes = [archetype for _, archetype in recording.logs if type(archetype).__name__ == 'GraphNodes']
    edges = [archetype for _, archetype in recording.logs if type(archetype).__name__ == 'GraphEdges']
    assert [len(item.node_ids.as_arrow_array()) for item in nodes] == [2, 3, 4, 5]
    assert len(edges) == 4
    assert [type(view).__name__ for view in visual.views_] == ['GraphView']


def test_default_blueprint_shows_one_dimensional_points_over_whole_run():
    """Test that the time series view of the positions of a one-dimensional gas shows the whole run."""
    blueprint = default_blueprint(IdealGasSystem(n_molecules=10, n_dim=1).prepare().system_)
    assert 'gas/x' in [str(view.origin) for view in _views(blueprint)]
    assert _whole_run_series(blueprint)


def test_default_blueprint_shows_other_states_of_points_over_whole_run():
    """Test that the time series view of the other states of two-dimensional points shows the whole run."""

    class WalkerElement(Element):
        """Walker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = VisualSpec('points', {'position': 'x'}),
            *,
            x: Annotated[object, StateSpec(vector=True)] = 0.0,
            energy: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.energy = energy

    blueprint = default_blueprint(
        CompositeSystem(
            elements=[GroupElement('walkers', element=WalkerElement, size=3)],
            space=ContinuousSpace(n_dim=2),
        ),
    )
    assert [type(view).__name__ for view in _views(blueprint)] == ['Spatial2DView', 'TimeSeriesView']
    assert _whole_run_series(blueprint)


def test_default_blueprint_of_ode_system():
    """Test that the populations share one time series view, without parameters, collapsed, and following."""
    blueprint = default_blueprint(LotkaVolterraSystem().prepare().system_)
    views = _views(blueprint)
    assert len(views) == 1
    assert isinstance(views[0], rrb.TimeSeriesView)
    assert views[0].contents == ['/prey/**', '/predators/**']
    assert blueprint.collapse_panels
    assert blueprint.time_panel.play_state == 'following'
    assert blueprint.top_panel.state == 'hidden'


def test_default_blueprint_of_field_system():
    """Test that each field state gets a 2D view side by side."""
    blueprint = default_blueprint(GrayScottSystem(shape=(16, 16)).prepare().system_)
    assert isinstance(blueprint.root_container, rrb.Horizontal)
    assert [str(view.origin) for view in _views(blueprint)] == ['cells/u', 'cells/v']


def test_default_blueprint_puts_spatial_views_above_series():
    """Test that agents share one spatial view above the time series of their counts."""
    blueprint = default_blueprint(WolfSheepSystem().prepare().system_)
    assert isinstance(blueprint.root_container, rrb.Vertical)
    spatial, series = _views(blueprint)
    assert spatial.contents == ['/grass/**', '/sheep/**', '/wolves/**']
    assert isinstance(series, rrb.TimeSeriesView)


def test_default_blueprint_splits_series_by_six():
    """Test that a type with more than six series is split over several views."""
    system = CompositeSystem(elements=[PopulationElement(f'p{index}') for index in range(8)])
    names = [str(view.name) for view in _views(default_blueprint(system))]
    assert names == ['PopulationElement 1', 'PopulationElement 2']


def test_default_blueprint_with_parameters():
    """Test that parameters get a time series view on request."""
    views = _views(default_blueprint(LotkaVolterraSystem().prepare().system_, parameters=True))
    assert [str(view.origin) for view in views][-1] == '/'
    assert views[-1].contents == ['/parameters/**']


@pytest.mark.parametrize(
    ('system_class', 'kind', 'origin'),
    [(GrayScottSystem, 'Spatial2DView', 'cells/u'), (SIRSystem, 'TimeSeriesView', '/')],
)
def test_primary_view_is_first_view(system_class, kind, origin):
    """Test that the primary view is the first view of the default blueprint."""
    system = system_class().prepare().system_
    view = primary_view(system)
    assert (type(view).__name__, str(view.origin)) == (kind, origin)
    assert str(_views(default_blueprint(system))[0].origin) == origin


def test_views_are_empty_without_declared_views():
    """Test that a system with no declared views has no views and an empty default blueprint."""
    observer = _prepared(CompositeSystem(), blueprint='default')
    assert observer.views_ == []
    assert _views(observer.blueprint_) == []


@pytest.mark.parametrize('settings', [{'primary': 'cells/u'}, {'parameters': True}, {'blueprint': 'grid'}])
def test_rerun_observer_check_rejects_settings_without_the_default_blueprint(settings):
    """Test that the settings of the default blueprint are rejected without it, as is an unknown blueprint name."""
    with pytest.raises(ValidationError, match='default'):
        RerunObserver(**settings).prepare()


def test_rerun_observer_saves_blueprint_next_to_recording(tmp_path):
    """Test that a saved recording gets its blueprint with the same stem."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint='default')
    simulate(LotkaVolterraSystem().prepare().system_, until=1, dt=0.1, observers=[observer])
    assert (tmp_path / 'run.rrd').exists()
    assert (tmp_path / 'run.rbl').exists()
    assert isinstance(observer.blueprint_, rrb.Blueprint)


def test_rerun_observer_without_blueprint_writes_recording_only(tmp_path):
    """Test that no blueprint is sent or saved when none is given."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    simulate(LotkaVolterraSystem().prepare().system_, until=1, dt=0.1, observers=[observer])
    assert observer.blueprint_ is None
    assert not (tmp_path / 'run.rbl').exists()


def test_rerun_observer_accepts_given_blueprint(tmp_path):
    """Test that a given blueprint is sent unchanged."""
    blueprint = rrb.Blueprint(rrb.TimeSeriesView(origin='/prey'))
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint=blueprint)
    simulate(LotkaVolterraSystem().prepare().system_, until=1, dt=0.1, observers=[observer])
    assert observer.blueprint_ is blueprint


def test_rerun_observer_rejects_unknown_blueprint(tmp_path):
    """Test that an unknown blueprint name is rejected."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint='grid')
    with pytest.raises(ValidationError, match='default'):
        simulate(LotkaVolterraSystem().prepare().system_, until=1, dt=0.1, observers=[observer])


def test_default_blueprint_puts_named_primary_view_first():
    """Test that a view named as primary comes first, with its row above the others."""
    views = _views(default_blueprint(BoidsSystem().prepare().system_, primary='reporters'))
    assert [str(view.name) for view in views] == ['reporters', 'flock']


def test_default_blueprint_rejects_unknown_primary_view():
    """Test that a primary view that matches no view is rejected with the names of the views."""
    with pytest.raises(ValidationError, match='The views are: u, v'):
        default_blueprint(GrayScottSystem(shape=(16, 16)).prepare().system_, primary='cells/w')


def test_primary_view_of_one_dimensional_lattice_is_space_time():
    """Test that a one-dimensional lattice puts its space-time image first."""
    view = primary_view(ElementaryAutomatonSystem().prepare().system_)
    assert type(view).__name__ == 'Spatial2DView'
    assert str(view.origin) == 'cells/space_time/alive'
    assert str(view.name) == 'alive history'


def test_primary_view_of_agents_without_space_is_a_trait_series():
    """Test that agents without a space show the statistics of their colored trait first, not the reporters."""
    problem = PassCompiler().prepare(MoranProcessSystem().prepare().system_).run().problem_
    assert problem.agent_space is None
    assert AgentsVisual(path=problem.agents[0].path).prepare(problem).views_ == []
    view = primary_view(problem)
    assert (type(view).__name__, str(view.name)) == ('TimeSeriesView', 'population trait')
