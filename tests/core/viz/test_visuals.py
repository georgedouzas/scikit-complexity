"""Test what the built-in visuals log for points, agents, and one-dimensional lattices."""

import numpy as np
import pytest

from skcomplex.core.model import ContinuousSpace, EuclideanSpace, LatticeSpace, Particle, System
from skcomplex.core.runtime import Frame, Observer, Simulation
from skcomplex.core.solvers import AgentSnapshot
from skcomplex.core.viz import AgentsVisual, ImageVisual, PointsVisual, ScalarsVisual
from skcomplex.models import create
from skcomplex.models.physics.statistical_mechanics import XYSpin


class Logs:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self):
        """Start with no logs."""
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

    class Keep(Observer):
        """Observer that keeps every frame."""

        def on_frame(self, frame):
            """Keep a frame."""
            frames.append(frame)

    simulation = Simulation(system, dt=dt, observers=[Keep()], random_state=0)
    simulation.run(until=steps * dt)
    return simulation.problem_, frames


def _first_frame(system):
    """Return the problem and the first frame of a system."""
    simulation = Simulation(system).run(until=0.0)
    problem = simulation.problem_
    return problem, Frame(0, 0.0, simulation.state_, simulation.parameters_, problem)


def _format(archetype):
    """Return the width and the height of a logged image."""
    image_format = archetype.format.as_arrow_array()[0].as_py()
    return image_format['width'], image_format['height']


def _radius(archetype):
    """Return the first radius of logged points."""
    return archetype.radii.as_arrow_array().to_pylist()[0]


def test_image_visual_logs_space_time_of_one_dimensional_lattice():
    """Test that a one-dimensional lattice logs a space-time image that grows by one row per frame."""
    problem, frames = _frames(create('elementary_automaton', size=21), steps=5, dt=1.0)
    visual, logs = ImageVisual(), Logs()
    for frame in frames:
        visual.log(logs, problem.elements[0], frame)
    images = logs.of('cells/space_time/alive')
    assert [type(image).__name__ for image in images] == ['DepthImage'] * len(frames)
    assert [_format(image) for image in images] == [(21, rows) for rows in range(1, len(frames) + 1)]


def test_image_visual_bounds_space_time_rows():
    """Test that the space-time image keeps only the most recent rows."""
    problem, frames = _frames(create('elementary_automaton', size=21), steps=5, dt=1.0)
    visual, logs = ImageVisual(history=3), Logs()
    for frame in frames:
        visual.log(logs, problem.elements[0], frame)
    assert _format(logs.of('cells/space_time/alive')[-1]) == (21, 3)


def test_image_visual_logs_one_row_as_rgb_image():
    """Test that the current row of a one-dimensional lattice is an image one pixel high, not a column of grey."""
    problem, frame = _first_frame(create('elementary_automaton', size=21))
    logs = Logs()
    ImageVisual(history=0).log(logs, problem.elements[0], frame)
    assert [path for path, _ in logs.logs] == ['cells/alive']
    image = logs.of('cells/alive')[0].format.as_arrow_array()[0].as_py()
    assert (image['width'], image['height'], image['color_model']) == (21, 1, 2)


def test_points_visual_radius_scales_with_extent():
    """Test that points without a given radius take a share of the extent of the positions."""
    system = System(elements=[Particle.group('bodies', 2, x=[[0.0, 0.0], [100.0, 0.0]])], space=EuclideanSpace(2))
    problem, frame = _first_frame(system)
    logs = Logs()
    PointsVisual(position='x', states=()).log(logs, problem.elements[0], frame)
    assert _radius(logs.of('bodies')[0]) == pytest.approx(2.0)
    logs = Logs()
    PointsVisual(position='x', radius=0.5, states=()).log(logs, problem.elements[0], frame)
    assert _radius(logs.of('bodies')[0]) == pytest.approx(0.5)


def test_points_visual_logs_trail_of_small_groups():
    """Test that a small group logs one line strip per point with its recent positions."""
    system = System(elements=[Particle.group('bodies', 2, v=[[1.0, 0.0], [0.0, 1.0]])], space=EuclideanSpace(2))
    problem, frames = _frames(system, steps=4)
    visual, logs = PointsVisual(position='x', states=(), trail=3), Logs()
    for frame in frames:
        visual.log(logs, problem.elements[0], frame)
    trails = logs.of('bodies/trail')
    assert type(trails[-1]).__name__ == 'LineStrips2D'
    strips = trails[-1].strips.as_arrow_array().to_pylist()
    assert [len(strip) for strip in strips] == [3, 3]
    assert np.allclose(strips[0][-1], frames[-1].value(problem.elements[0].states['x'])[0])


@pytest.mark.parametrize(('count', 'trail'), [(11, 256), (2, 0)])
def test_points_visual_leaves_out_trail(count, trail):
    """Test that large groups and a trail of 0 positions log no trail."""
    system = System(elements=[Particle.group('bodies', count)], space=EuclideanSpace(3))
    problem, frame = _first_frame(system)
    logs = Logs()
    PointsVisual(position='x', states=(), trail=trail).log(logs, problem.elements[0], frame)
    assert [path for path, _ in logs.logs] == ['bodies']


def test_agents_visual_radius_scales_with_continuous_space():
    """Test that agents in a continuous space take a share of its side, and agents on a grid 0.4 cells."""
    problem, frames = _frames(create('obesity', n_people=10, n_places=4, random_state=0), steps=1, dt=1.0)
    space = problem.agent_space
    assert isinstance(space, ContinuousSpace)
    logs = Logs()
    snapshot = frames[0].agents['people']
    AgentsVisual().log_agents(logs, 'people', snapshot, space)
    assert _radius(logs.of('people')[0]) == pytest.approx(0.02 * 280.0)
    AgentsVisual().log_agents(logs, 'grid', snapshot, None)
    assert _radius(logs.of('grid')[0]) == pytest.approx(0.4)


def test_points_visual_trail_width_follows_extent():
    """Test that the trail is a quarter as wide as the extent share, even when a tiny radius is given."""
    system = System(elements=[Particle.group('bodies', 2, x=[[0.0, 0.0], [100.0, 0.0]])], space=EuclideanSpace(2))
    problem, frame = _first_frame(system)
    logs = Logs()
    PointsVisual(position='x', radius=0.05, states=()).log(logs, problem.elements[0], frame)
    assert _radius(logs.of('bodies/trail')[0]) == pytest.approx(0.5)


def test_image_visual_colors_angles_cyclically():
    """Test that angles just either side of ±π get nearly the same color, as on a cyclic palette."""
    space = LatticeSpace((2, 2))
    system = System(elements=[XYSpin.lattice('spins', space, theta=[[-3.1, 3.1], [0.0, 1.5]])])
    problem, frame = _first_frame(system)
    logs = Logs()
    ImageVisual().log(logs, problem.elements[0], frame)
    image = logs.of('spins/theta')[0]
    pixels = np.asarray(image.buffer.as_arrow_array()[0].as_py(), dtype=int).reshape(4, 3)
    assert np.abs(pixels[0] - pixels[1]).max() < 20
    assert np.abs(pixels[0] - pixels[3]).max() > 60


def _series_color(archetype):
    """Return the color of logged series lines."""
    return archetype.colors.as_arrow_array().to_pylist()[0]


def test_scalars_visual_gives_states_distinct_colors():
    """Test that the white and black daisies of Daisyworld get distinct static series colors."""
    problem, frame = _first_frame(create('daisyworld'))
    logs = Logs()
    visual = ScalarsVisual()
    visual.log(logs, problem.elements[0], frame)
    visual.log(logs, problem.elements[0], frame)
    white = [item for item in logs.of('planet/white') if type(item).__name__ == 'SeriesLines']
    black = [item for item in logs.of('planet/black') if type(item).__name__ == 'SeriesLines']
    assert len(white) == len(black) == 1
    assert 'planet/white' in logs.static
    assert _series_color(white[0]) != _series_color(black[0])


def test_scalars_visual_colors_elements_apart():
    """Test that the first states of two elements get distinct series colors."""
    problem, frame = _first_frame(create('coupled_pendulums'))
    logs = Logs()
    for element in problem.elements:
        ScalarsVisual().log(logs, element, frame)
    left = next(item for item in logs.of('left/theta') if type(item).__name__ == 'SeriesLines')
    right = next(item for item in logs.of('right/theta') if type(item).__name__ == 'SeriesLines')
    assert _series_color(left) != _series_color(right)


def test_points_visual_draws_two_dimensions_with_vertical_axis_up():
    """Test that a 2D position (1, 2) is logged as (1, -2), since Rerun's 2D views point their vertical axis down."""
    system = System(elements=[Particle.group('bodies', 1, x=[[1.0, 2.0]])], space=EuclideanSpace(2))
    problem, frame = _first_frame(system)
    logs = Logs()
    PointsVisual(position='x', states=()).log(logs, problem.elements[0], frame)
    points = logs.of('bodies')[0].positions.as_arrow_array().to_pylist()
    assert points == [[1.0, -2.0]]
    trail = logs.of('bodies/trail')[0].strips.as_arrow_array().to_pylist()
    assert trail[0][-1] == [1.0, -2.0]


def test_agents_visual_widens_color_range_of_equal_start():
    """Test that agents that all start with one value get different colors once their values spread."""
    visual, logs = AgentsVisual(color_by='rank'), Logs()
    space = ContinuousSpace(10.0, 10.0)
    first = AgentSnapshot(np.arange(3), {'rank': np.zeros(3)}, np.zeros((3, 2)))
    later = AgentSnapshot(np.arange(3), {'rank': np.array([0.0, 1.0, 2.0])}, np.zeros((3, 2)))
    visual.log_agents(logs, 'animals', first, space)
    visual.log_agents(logs, 'animals', later, space)
    colors = logs.of('animals')[-1].colors.as_arrow_array().to_pylist()
    assert len(set(colors)) == 3
    assert visual.color_range is None


def test_points_visual_keeps_given_color_range():
    """Test that a given color range stays fixed and the visual does not overwrite its parameter."""
    system = System(elements=[Particle.group('bodies', 2, v=[[1.0, 0.0], [3.0, 0.0]])], space=EuclideanSpace(2))
    problem, frame = _first_frame(system)
    visual = PointsVisual(position='x', color_by='v', states=())
    visual.log(Logs(), problem.elements[0], frame)
    assert visual.color_range is None


def test_points_visual_sizes_single_particles_by_the_whole_drawing():
    """Test that the masses of a spring chain, one particle per element, take a radius from the extent of the chain."""
    problem, frame = _first_frame(create('spring_chain'))
    logs = Logs()
    PointsVisual(position='x', states=()).log(logs, problem.elements[0], frame)
    assert _radius(logs.of('mass_0')[0]) > 0.01


def test_scalars_visual_leaves_out_lag_companions():
    """Test that the past values a delayed model keeps as lag states are not logged as series."""
    problem, frame = _first_frame(create('delayed_sirs'))
    logs = Logs()
    ScalarsVisual().log(logs, problem.elements[0], frame)
    paths = {path for path, _ in logs.logs}
    assert 'first/x' in paths
    assert not any('_lag' in path for path in paths)


def test_scalars_visual_logs_chosen_components():
    """Test that a scalars visual with chosen components logs only those components of a vector state."""
    problem, frame = _first_frame(create('lorenz_96'))
    logs = Logs()
    ScalarsVisual(components=[0, 1, 2]).log(logs, problem.elements[0], frame)
    scalars = [item for item in logs.of('ring/x') if type(item).__name__ == 'Scalars']
    assert len(scalars[0].scalars.as_arrow_array().to_pylist()) == 3


def test_agents_visual_draws_continuous_spaces_with_vertical_axis_up():
    """Test that agents in a continuous space are logged with y negated and the view covers the whole space."""
    space = ContinuousSpace(10.0, 20.0)
    snapshot = AgentSnapshot(np.arange(1), {'rank': np.zeros(1)}, np.array([[3.0, 4.0]]))
    logs = Logs()
    AgentsVisual().log_agents(logs, 'birds', snapshot, space)
    assert logs.of('birds')[0].positions.as_arrow_array().to_pylist() == [[3.0, -4.0]]
    problem, _ = _frames(create('obesity', n_people=10, n_places=4, random_state=0), steps=1, dt=1.0)
    view = AgentsVisual().views(problem.agents[0], problem)[0]
    bounds = view.properties['VisualBounds2D']
    assert bounds is not None


def test_colors_start_bright_enough_to_see_on_black():
    """Test that the lowest color of the point scale is not the near-black purple of viridis."""
    problem, frame = _first_frame(
        System(elements=[Particle.group('bodies', 2, v=[[0.0, 0.0], [1.0, 0.0]])], space=EuclideanSpace(2)),
    )
    logs = Logs()
    PointsVisual(position='x', color_by='v', states=()).log(logs, problem.elements[0], frame)
    darkest = logs.of('bodies')[0].colors.as_arrow_array().to_pylist()[0]
    red, green, blue = (darkest >> 24) & 0xFF, (darkest >> 16) & 0xFF, (darkest >> 8) & 0xFF
    assert max(red, green, blue) >= 120
