"""Test the `_points` module of `skcomplex.core.engine.viz`."""

import numpy as np
import pytest
import rerun as rr

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.viz import PointsVisual
from skcomplex.core.systems import System
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
from skcomplex.systems.physics.waves import SpringChainSystem

from .conftest import Logs, _first_frame, _frames, _problem_of, _radius


def _bodies(label, size, n_dim, values=None):
    """Return a group of points of a continuous space that move at their velocities `x'`."""
    values = {'x': np.zeros((size, n_dim)), "x'": np.zeros((size, n_dim)), **(values or {})}
    inertia = Interaction(
        'inertia',
        roles={'target': label},
        laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0 * s.target.x)],
    )
    element = Element(label, values=values, size=size)
    return System(elements=[element], interactions=[inertia], space=ContinuousSpace(n_dim=n_dim))


def test_views_of_points_log_other_states_as_series():
    """Test that the states other than the position get a time series view."""
    problem = _problem_of(IdealGasSystem(n_molecules=10))
    views = PointsVisual(position='x', path=problem.elements[0].path).prepare(problem).views_
    assert [type(view).__name__ for view in views] == ['Spatial2DView', 'TimeSeriesView']


def test_points_visual_radius_scales_with_extent():
    """Test that points without a given radius take a share of the extent of the positions."""
    system = _bodies('bodies', 2, 2, {'x': [[0.0, 0.0], [100.0, 0.0]]})
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
    system = _bodies('bodies', 2, 2, {"x'": [[1.0, 0.0], [0.0, 1.0]]})
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
    assert np.allclose(strips[0][-1], frames[-1].get_value(problem.elements[0].states['x'])[0])


@pytest.mark.parametrize(('count', 'trail'), [(11, 256), (2, 0)])
def test_points_visual_leaves_out_trail(count, trail):
    """Test that large groups and a trail of 0 positions log no trail."""
    system = _bodies('bodies', count, 3)
    problem, frame = _first_frame(system)
    logs = Logs()
    (
        PointsVisual(position='x', states=(), trail=trail, path=problem.elements[0].path, recording=logs)
        .prepare(problem)
        .run(frame)
    )
    assert [path for path, _ in logs.logs] == ['bodies']


def test_points_visual_trail_width_follows_extent():
    """Test that the trail is a quarter as wide as the extent share, even when a tiny radius is given."""
    system = _bodies('bodies', 2, 2, {'x': [[0.0, 0.0], [100.0, 0.0]]})
    problem, frame = _first_frame(system)
    logs = Logs()
    (
        PointsVisual(position='x', radius=0.05, states=(), path=problem.elements[0].path, recording=logs)
        .prepare(problem)
        .run(frame)
    )
    assert _radius(logs.of('bodies/trail')[0]) == pytest.approx(0.5)


def test_points_visual_draws_two_dimensions_with_vertical_axis_up():
    """Test that a 2D position (1, 2) is logged as (1, -2), since Rerun's 2D views point their vertical axis down."""
    system = _bodies('bodies', 1, 2, {'x': [[1.0, 2.0]]})
    problem, frame = _first_frame(system)
    logs = Logs()
    (PointsVisual(position='x', states=(), path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    points = logs.of('bodies')[0].positions.as_arrow_array().to_pylist()
    assert points == [[1.0, -2.0]]
    trail = logs.of('bodies/trail')[0].strips.as_arrow_array().to_pylist()
    assert trail[0][-1] == [1.0, -2.0]


def test_points_visual_keeps_given_color_range():
    """Test that a given color range stays fixed and the visual does not overwrite its parameter."""
    system = _bodies('bodies', 2, 2, {"x'": [[1.0, 0.0], [3.0, 0.0]]})
    problem, frame = _first_frame(system)
    visual = PointsVisual(position='x', color_by="x'", states=(), path=problem.elements[0].path, recording=Logs())
    visual.prepare(problem).run(frame)
    assert visual.color_range is None


def test_points_visual_sizes_single_particles_by_the_whole_drawing():
    """Test that the masses of a spring chain, one particle per element, take a radius from the extent of the chain."""
    problem, frame = _first_frame(SpringChainSystem())
    logs = Logs()
    (PointsVisual(position='x', states=(), path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    assert _radius(logs.of('mass_0')[0]) > 0.01


def test_colors_start_bright_enough_to_see_on_black():
    """Test that the lowest color of the point scale is not the near-black purple of viridis."""
    problem, frame = _first_frame(
        _bodies('bodies', 2, 2, {"x'": [[0.0, 0.0], [1.0, 0.0]]}),
    )
    logs = Logs()
    (
        PointsVisual(position='x', color_by="x'", states=(), path=problem.elements[0].path, recording=logs)
        .prepare(problem)
        .run(frame)
    )
    darkest = logs.of('bodies')[0].colors.as_arrow_array().to_pylist()[0]
    red, green, blue = (darkest >> 24) & 0xFF, (darkest >> 16) & 0xFF, (darkest >> 8) & 0xFF
    assert max(red, green, blue) >= 120


@pytest.mark.parametrize('dimension', [2, 3])
def test_points_visual_joins_the_points_by_one_strip(dimension):
    """Test that `strip` logs one line strip through the points in member order, with their mean color."""
    positions = np.arange(4.0 * dimension).reshape(4, dimension)
    state = 'x' if dimension == 3 else "x'"
    system = _bodies('chain', 4, dimension, {state: positions})
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
    problem, frame = _first_frame(_bodies('chain', 4, 3, {'x': np.zeros((4, 3))}))
    logs = Logs()
    PointsVisual(position='x', states=(), trail=0, path='chain', recording=logs).prepare(problem).run(frame)
    assert logs.of('chain/strip') == []


def test_points_visual_draws_positions_and_colors_that_no_law_changes():
    """Test that points sit at fixed positions and take colors from a value that are both parameters."""
    sites = Element('sites', values={'x': [[0.0, 0.0], [1.0, 0.0]], 'n': [1.0, 2.0]}, size=2)
    trajectory = System(elements=[sites]).prepare().simulate(until=0.0)
    problem = trajectory.problem
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], problem)
    logs = Logs()
    visual = PointsVisual(position='x', color_by='n', states=(), path=problem.elements[0].path, recording=logs)
    visual.prepare(problem).run(frame)
    assert logs.of('sites')[0].positions.as_arrow_array().to_pylist() == [[0.0, 0.0], [1.0, 0.0]]
