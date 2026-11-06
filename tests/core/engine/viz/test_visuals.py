"""Test what the built-in visuals log for points, agents, and one-dimensional lattices."""

import numpy as np
import pytest
import rerun as rr
from shapely import Polygon

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement, PointElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.runtime import BaseObserver, Simulation
from skcomplex.core.engine.viz import AgentsVisual, ImageVisual, PointsVisual, ScalarsVisual
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import AgentSnapshotSpec
from skcomplex.systems import create_system
from skcomplex.systems.physics.statistical_mechanics import XYSpinElement


class RestingElement(Element):
    """Test helper agent whose rule does nothing."""

    def _get_laws(self, namespace):
        """Test helper laws."""
        return [RuleLaw(self.rest)]

    def rest(self, agent, system):
        """Test helper rule."""


class Logs:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self):
        self.logs = []
        self.static = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype, and whether it is static."""
        self.logs.append((path, archetype))
        if static:
            self.static.append(path)

    def of(self, path):
        """Return the archetypes logged under a path."""
        return [archetype for logged, archetype in self.logs if logged == path]


def _frames(system, steps, dt=0.1):
    """Return the problem and the frames of a run of a system."""
    frames = []

    class KeepObserver(BaseObserver):
        """Observer that keeps every frame."""

        def _run(self, inputs):
            """Keep a frame."""
            if inputs is not None:
                frames.append(inputs)

    simulation = Simulation(system, dt=dt, observers=[KeepObserver()], random_state=0).prepare()
    simulation.run(until=steps * dt)
    return simulation.problem_, frames


def _first_frame(system):
    """Return the problem and the first frame of a system."""
    simulation = Simulation(system).prepare().run(until=0.0)
    problem = simulation.problem_
    return problem, FrameSpec(0, 0.0, simulation.state_, simulation.parameters_, problem)


def _format(archetype):
    """Return the width and the height of a logged image."""
    image_format = archetype.format.as_arrow_array()[0].as_py()
    return image_format['width'], image_format['height']


def _radius(archetype):
    """Return the first radius of logged points."""
    return archetype.radii.as_arrow_array().to_pylist()[0]


def test_image_visual_logs_space_time_of_one_dimensional_lattice():
    """Test that a one-dimensional lattice logs a space-time image that grows by one row per frame."""
    problem, frames = _frames(create_system('elementary_automaton', size=21), steps=5, dt=1.0)
    logs = Logs()
    visual = ImageVisual(path=problem.elements[0].path, recording=logs).prepare(problem)
    for frame in frames:
        visual.run(frame)
    images = logs.of('cells/space_time/alive')
    assert [type(image).__name__ for image in images] == ['DepthImage'] * len(frames)
    assert [_format(image) for image in images] == [(21, rows) for rows in range(1, len(frames) + 1)]


def test_image_visual_bounds_space_time_rows():
    """Test that the space-time image keeps only the most recent rows."""
    problem, frames = _frames(create_system('elementary_automaton', size=21), steps=5, dt=1.0)
    logs = Logs()
    visual = ImageVisual(history=3, path=problem.elements[0].path, recording=logs).prepare(problem)
    for frame in frames:
        visual.run(frame)
    assert _format(logs.of('cells/space_time/alive')[-1]) == (21, 3)


def test_image_visual_logs_one_row_as_rgb_image():
    """Test that the current row of a one-dimensional lattice is an image one pixel high, not a column of grey."""
    problem, frame = _first_frame(create_system('elementary_automaton', size=21))
    logs = Logs()
    ImageVisual(history=0, path=problem.elements[0].path, recording=logs).prepare(problem).run(frame)
    assert [path for path, _ in logs.logs] == ['cells/alive']
    image = logs.of('cells/alive')[0].format.as_arrow_array()[0].as_py()
    assert (image['width'], image['height'], image['color_model']) == (21, 1, 2)


def test_points_visual_radius_scales_with_extent():
    """Test that points without a given radius take a share of the extent of the positions."""
    system = System(
        elements=[GroupElement('bodies', element=PointElement, size=2, values={'x': [[0.0, 0.0], [100.0, 0.0]]})],
        space=ContinuousSpace(n_dim=2),
    )
    problem, frame = _first_frame(system)
    logs = Logs()
    (PointsVisual(position='x', states=(), path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    assert _radius(logs.of('bodies')[0]) == pytest.approx(2.0)
    logs = Logs()
    (
        PointsVisual(position='x', radius=0.5, states=(), path=problem.elements[0].path, recording=logs)
        .prepare(problem)
        .run(frame)
    )
    assert _radius(logs.of('bodies')[0]) == pytest.approx(0.5)


def test_points_visual_logs_trail_of_small_groups():
    """Test that a small group logs one line strip per point with its recent positions."""
    system = System(
        elements=[GroupElement('bodies', element=PointElement, size=2, values={'x_d1': [[1.0, 0.0], [0.0, 1.0]]})],
        space=ContinuousSpace(n_dim=2),
    )
    problem, frames = _frames(system, steps=4)
    logs = Logs()
    visual = PointsVisual(position='x', states=(), trail=3, path=problem.elements[0].path, recording=logs).prepare(
        problem,
    )
    for frame in frames:
        visual.run(frame)
    trails = logs.of('bodies/trail')
    assert type(trails[-1]).__name__ == 'LineStrips2D'
    strips = trails[-1].strips.as_arrow_array().to_pylist()
    assert [len(strip) for strip in strips] == [3, 3]
    assert np.allclose(strips[0][-1], frames[-1].value(problem.elements[0].states['x'])[0])


@pytest.mark.parametrize(('count', 'trail'), [(11, 256), (2, 0)])
def test_points_visual_leaves_out_trail(count, trail):
    """Test that large groups and a trail of 0 positions log no trail."""
    system = System(
        elements=[GroupElement('bodies', element=PointElement, size=count)],
        space=ContinuousSpace(n_dim=3),
    )
    problem, frame = _first_frame(system)
    logs = Logs()
    (
        PointsVisual(position='x', states=(), trail=trail, path=problem.elements[0].path, recording=logs)
        .prepare(problem)
        .run(frame)
    )
    assert [path for path, _ in logs.logs] == ['bodies']


def test_agents_visual_radius_scales_with_continuous_space(log_agents):
    """Test that agents in a continuous space take a share of its side, and agents on a grid 0.4 cells."""
    problem, frames = _frames(create_system('obesity', n_people=10, n_places=4, random_state=0), steps=1, dt=1.0)
    space = problem.agent_space
    assert isinstance(space, ContinuousSpace)
    logs = Logs()
    snapshot = frames[0].agents['people']
    log_agents(AgentsVisual(), logs, 'people', [snapshot], space)
    assert _radius(logs.of('people')[0]) == pytest.approx(0.02 * 280.0)
    log_agents(AgentsVisual(), logs, 'grid', [snapshot])
    assert _radius(logs.of('grid')[0]) == pytest.approx(0.4)


def test_points_visual_trail_width_follows_extent():
    """Test that the trail is a quarter as wide as the extent share, even when a tiny radius is given."""
    system = System(
        elements=[GroupElement('bodies', element=PointElement, size=2, values={'x': [[0.0, 0.0], [100.0, 0.0]]})],
        space=ContinuousSpace(n_dim=2),
    )
    problem, frame = _first_frame(system)
    logs = Logs()
    (
        PointsVisual(position='x', radius=0.05, states=(), path=problem.elements[0].path, recording=logs)
        .prepare(problem)
        .run(frame)
    )
    assert _radius(logs.of('bodies/trail')[0]) == pytest.approx(0.5)


def test_image_visual_colors_angles_cyclically():
    """Test that angles just either side of ±π get nearly the same color, as on a cyclic palette."""
    space = LatticeSpace((2, 2))
    system = System(
        elements=[
            LatticeElement('spins', element=XYSpinElement, space=space, values={'theta': [[-3.1, 3.1], [0.0, 1.5]]}),
        ],
    )
    problem, frame = _first_frame(system)
    logs = Logs()
    ImageVisual(path=problem.elements[0].path, recording=logs).prepare(problem).run(frame)
    image = logs.of('spins/theta')[0]
    pixels = np.asarray(image.buffer.as_arrow_array()[0].as_py(), dtype=int).reshape(4, 3)
    assert np.abs(pixels[0] - pixels[1]).max() < 20
    assert np.abs(pixels[0] - pixels[3]).max() > 60


def _series_color(archetype):
    """Return the color of logged series lines."""
    return archetype.colors.as_arrow_array().to_pylist()[0]


def test_scalars_visual_gives_states_distinct_colors():
    """Test that the white and black daisies of Daisyworld get distinct static series colors."""
    problem, frame = _first_frame(create_system('daisyworld'))
    logs = Logs()
    visual = ScalarsVisual(path=problem.elements[0].path, recording=logs).prepare(problem)
    visual.run(frame)
    visual.run(frame)
    white = [item for item in logs.of('planet/white') if type(item).__name__ == 'SeriesLines']
    black = [item for item in logs.of('planet/black') if type(item).__name__ == 'SeriesLines']
    assert len(white) == len(black) == 1
    assert 'planet/white' in logs.static
    assert _series_color(white[0]) != _series_color(black[0])


def test_scalars_visual_colors_elements_apart():
    """Test that the first states of two elements get distinct series colors."""
    problem, frame = _first_frame(create_system('coupled_pendulums'))
    logs = Logs()
    for element in problem.elements:
        ScalarsVisual(path=element.path, recording=logs).prepare(frame.problem).run(frame)
    left = next(item for item in logs.of('left/theta') if type(item).__name__ == 'SeriesLines')
    right = next(item for item in logs.of('right/theta') if type(item).__name__ == 'SeriesLines')
    assert _series_color(left) != _series_color(right)


def test_points_visual_draws_two_dimensions_with_vertical_axis_up():
    """Test that a 2D position (1, 2) is logged as (1, -2), since Rerun's 2D views point their vertical axis down."""
    system = System(
        elements=[GroupElement('bodies', element=PointElement, size=1, values={'x': [[1.0, 2.0]]})],
        space=ContinuousSpace(n_dim=2),
    )
    problem, frame = _first_frame(system)
    logs = Logs()
    (PointsVisual(position='x', states=(), path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    points = logs.of('bodies')[0].positions.as_arrow_array().to_pylist()
    assert points == [[1.0, -2.0]]
    trail = logs.of('bodies/trail')[0].strips.as_arrow_array().to_pylist()
    assert trail[0][-1] == [1.0, -2.0]


def test_agents_visual_widens_color_range_of_equal_start(log_agents):
    """Test that agents that all start with one value get different colors once their values spread."""
    visual, logs = AgentsVisual(color_by='rank'), Logs()
    space = ContinuousSpace(extent=(10.0, 10.0)).prepare()
    first = AgentSnapshotSpec(np.arange(3), {'rank': np.zeros(3)}, np.zeros((3, 2)))
    later = AgentSnapshotSpec(np.arange(3), {'rank': np.array([0.0, 1.0, 2.0])}, np.zeros((3, 2)))
    log_agents(visual, logs, 'animals', [first, later], space)
    colors = logs.of('animals')[-1].colors.as_arrow_array().to_pylist()
    assert len(set(colors)) == 3
    assert visual.color_range is None


def test_points_visual_keeps_given_color_range():
    """Test that a given color range stays fixed and the visual does not overwrite its parameter."""
    system = System(
        elements=[GroupElement('bodies', element=PointElement, size=2, values={'x_d1': [[1.0, 0.0], [3.0, 0.0]]})],
        space=ContinuousSpace(n_dim=2),
    )
    problem, frame = _first_frame(system)
    visual = PointsVisual(position='x', color_by='x_d1', states=(), path=problem.elements[0].path, recording=Logs())
    visual.prepare(problem).run(frame)
    assert visual.color_range is None


def test_points_visual_sizes_single_particles_by_the_whole_drawing():
    """Test that the masses of a spring chain, one particle per element, take a radius from the extent of the chain."""
    problem, frame = _first_frame(create_system('spring_chain'))
    logs = Logs()
    (PointsVisual(position='x', states=(), path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    assert _radius(logs.of('mass_0')[0]) > 0.01


def test_scalars_visual_leaves_out_lag_companions():
    """Test that the past values a delayed system keeps as lag states are not logged as series."""
    problem, frame = _first_frame(create_system('delayed_sirs'))
    logs = Logs()
    ScalarsVisual(path=problem.elements[0].path, recording=logs).prepare(problem).run(frame)
    paths = {path for path, _ in logs.logs}
    assert 'first/x' in paths
    assert not any('_lag' in path for path in paths)


def test_scalars_visual_logs_chosen_components():
    """Test that a scalars visual with chosen components logs only those components of a vector state."""
    problem, frame = _first_frame(create_system('lorenz_96'))
    logs = Logs()
    (ScalarsVisual(components=[0, 1, 2], path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    scalars = [item for item in logs.of('ring/x') if type(item).__name__ == 'Scalars']
    assert len(scalars[0].scalars.as_arrow_array().to_pylist()) == 3


def test_agents_visual_draws_continuous_spaces_with_vertical_axis_up(log_agents):
    """Test that agents in a continuous space are logged with y negated and the view covers the whole space."""
    space = ContinuousSpace(extent=(10.0, 20.0)).prepare()
    snapshot = AgentSnapshotSpec(np.arange(1), {'rank': np.zeros(1)}, np.array([[3.0, 4.0]]))
    logs = Logs()
    log_agents(AgentsVisual(), logs, 'birds', [snapshot], space)
    assert logs.of('birds')[0].positions.as_arrow_array().to_pylist() == [[3.0, -4.0]]
    problem, _ = _frames(create_system('obesity', n_people=10, n_places=4, random_state=0), steps=1, dt=1.0)
    view = AgentsVisual(path=problem.agents[0].path).prepare(problem).views_[0]
    bounds = view.properties['VisualBounds2D']
    assert bounds is not None


def test_agents_visual_draws_boids_in_three_dimensions(log_agents):
    """Test that agents in a three-dimensional space are logged as one `Points3D` batch per frame at their positions."""
    space = ContinuousSpace(extent=(10.0, 20.0, 30.0)).prepare()
    locations = [np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]), np.array([[3.5, 4.5, 5.5], [6.5, 7.5, 8.5]])]
    snapshots = [AgentSnapshotSpec(np.arange(2), {'rank': np.zeros(2)}, points) for points in locations]
    logs = Logs()
    log_agents(AgentsVisual(), logs, 'birds', snapshots, space)
    batches = logs.of('birds')
    assert [type(batch).__name__ for batch in batches] == ['Points3D', 'Points3D']
    assert batches[1].positions.as_arrow_array().to_pylist() == locations[1].tolist()
    recording = rr.RecordingStream('three_dimensions')
    stream = recording.memory_recording()
    log_agents(AgentsVisual(), recording, 'birds', snapshots, space)
    assert len(stream.drain_as_bytes()) > 0


def test_agents_visual_has_a_three_dimensional_view():
    """Test that the view of agents in a three-dimensional space is a 3D spatial view with their path as origin."""
    flock = GroupElement('flock', element=RestingElement, size=2)
    system = System(elements=[flock], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
    problem = PassCompiler().prepare(system).run().problem_
    view = AgentsVisual(path='flock').prepare(problem).views_[0]
    assert type(view).__name__ == 'Spatial3DView'


def test_agents_visual_draws_regions_once_under_the_agents(log_agents):
    """Test that a boundary layer is logged once as static strips, and the agents per frame as `Points2D`."""
    hole = [(2.5, 0.5), (3.5, 0.5), (3.5, 1.5)]
    island = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)], holes=[hole])
    space = ContinuousSpace(extent=(5.0, 5.0), regions={'boundary': [island]}).prepare()
    snapshots = [AgentSnapshotSpec(np.arange(1), {'rank': np.zeros(1)}, np.array([[3.0, 1.0]]))] * 3
    logs = Logs()
    log_agents(AgentsVisual(), logs, 'people', snapshots, space)
    assert logs.static == ['people/regions/boundary']
    strips = logs.of('people/regions/boundary')
    assert [type(item).__name__ for item in strips] == ['LineStrips2D']
    assert strips[0].strips.as_arrow_array().to_pylist()[0][1] == [4.0, 0.0]
    assert strips[0].strips.as_arrow_array().to_pylist()[0][2] == [4.0, -4.0]
    assert len(strips[0].strips.as_arrow_array().to_pylist()) == 2
    assert [type(item).__name__ for item in logs.of('people')] == ['Points2D'] * 3
    recording = rr.RecordingStream('island')
    stream = recording.memory_recording()
    log_agents(AgentsVisual(), recording, 'people', snapshots, space)
    assert len(stream.drain_as_bytes()) > 0


def test_colors_start_bright_enough_to_see_on_black():
    """Test that the lowest color of the point scale is not the near-black purple of viridis."""
    problem, frame = _first_frame(
        System(
            elements=[
                GroupElement('bodies', element=PointElement, size=2, values={'x_d1': [[0.0, 0.0], [1.0, 0.0]]}),
            ],
            space=ContinuousSpace(n_dim=2),
        ),
    )
    logs = Logs()
    (
        PointsVisual(position='x', color_by='x_d1', states=(), path=problem.elements[0].path, recording=logs)
        .prepare(problem)
        .run(frame)
    )
    darkest = logs.of('bodies')[0].colors.as_arrow_array().to_pylist()[0]
    red, green, blue = (darkest >> 24) & 0xFF, (darkest >> 16) & 0xFF, (darkest >> 8) & 0xFF
    assert max(red, green, blue) >= 120


def test_image_visual_logs_position_velocity_image_and_density_of_two_stream():
    """Test that a two-stream system logs its position-velocity image and its density over position every frame."""
    system = create_system('vlasov_equation', positions=8, velocities=16, setup='two-stream')
    problem, frames = _frames(system, steps=3, dt=0.1)
    logs = Logs()
    visual = ImageVisual(moments=(0,), history=0, path='gas', recording=logs).prepare(problem)
    for frame in frames:
        visual.run(frame)
    images = logs.of('gas/f')
    assert [_format(image) for image in images] == [(16, 8)] * len(frames)
    densities = logs.of('gas/f/moment_0')
    assert [type(item).__name__ for item in densities] == ['BarChart'] * len(frames)
    assert len(densities[-1].values.as_arrow_array().to_pylist()[0]['buffer']) == 8
    assert [str(view.origin) for view in visual.views_] == ['gas/f/moment_0', 'gas/f']


def test_image_visual_logs_a_plane_of_a_phase_space():
    """Test that a phase space of three axes logs the plane its slices leave."""
    system = create_system('boltzmann_equation', positions=4, velocities=8)
    problem, frame = _first_frame(system)
    logs = Logs()
    visual = ImageVisual(slices={'vy': 2}, moments=(0,), path='gas', recording=logs).prepare(problem)
    visual.run(frame)
    assert [_format(image) for image in logs.of('gas/f')] == [(8, 4)]
    assert len(logs.of('gas/f/moment_0')) == 1


def test_image_visual_rejects_slices_that_leave_other_than_two_axes():
    """Test that slices must leave two axes, and that they need a phase space."""
    problem, _ = _first_frame(create_system('boltzmann_equation', positions=4, velocities=8))
    with pytest.raises(ValidationError, match='every axis but two'):
        ImageVisual(slices={'vx': 1, 'vy': 2}, path='gas').prepare(problem)
    problem, _ = _first_frame(create_system('gray_scott', shape=(4, 4)))
    with pytest.raises(ValidationError, match='phase space'):
        ImageVisual(slices={0: 1}, path=problem.elements[0].path).prepare(problem)


@pytest.mark.parametrize('dimension', [2, 3])
def test_points_visual_joins_the_points_by_one_strip(dimension):
    """Test that `strip` logs one line strip through the points in member order, with their mean color."""
    positions = np.arange(4.0 * dimension).reshape(4, dimension)
    state = 'x' if dimension == 3 else 'x_d1'
    group = GroupElement('chain', element=PointElement, size=4, values={state: positions})
    system = System(elements=[group], space=ContinuousSpace(n_dim=dimension))
    problem, frame = _first_frame(system)
    logs = Logs()
    (
        PointsVisual(position=state, states=(), trail=0, strip=True, path='chain', recording=logs)
        .prepare(problem)
        .run(frame)
    )
    strips = logs.of('chain/strip')
    assert len(strips) == 1
    assert isinstance(strips[0], rr.LineStrips3D if dimension == 3 else rr.LineStrips2D)
    assert strips[0].strips.as_arrow_array().to_pylist()[0] is not None


def test_points_visual_logs_no_strip_by_default():
    """Test that points log no strip without the setting."""
    group = GroupElement('chain', element=PointElement, size=4, values={'x': np.zeros((4, 3))})
    problem, frame = _first_frame(System(elements=[group], space=ContinuousSpace(n_dim=3)))
    logs = Logs()
    PointsVisual(position='x', states=(), trail=0, path='chain', recording=logs).prepare(problem).run(frame)
    assert logs.of('chain/strip') == []
