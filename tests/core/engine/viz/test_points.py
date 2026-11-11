"""Test the `_points` module of `skcomplex.core.engine.viz`."""

from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
import rerun as rr

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.viz import PointsVisual
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
from skcomplex.systems.physics.waves import SpringChainSystem

from .conftest import Logs, _first_frame, _frames, _radius


class BallElement(Element):
    """Point of a continuous space, with a vector state `x` of order 2."""

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, points at the values of `x`."""
        super()._prepare(context)
        self.default_visual_ = VisualSpec('points', {'position': 'x'})


def test_views_of_points_log_other_states_as_series():
    """Test that the states other than the position get a time series view."""
    problem = PassCompiler().prepare(IdealGasSystem(n_molecules=10).prepare().system_).run().problem_
    views = PointsVisual(position='x', path=problem.elements[0].path).prepare(problem).views_
    assert [type(view).__name__ for view in views] == ['Spatial2DView', 'TimeSeriesView']


def test_points_visual_radius_scales_with_extent():
    """Test that points without a given radius take a share of the extent of the positions."""
    system = CompositeSystem(
        elements=[GroupElement('bodies', element=BallElement, size=2, values={'x': [[0.0, 0.0], [100.0, 0.0]]})],
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
    system = CompositeSystem(
        elements=[GroupElement('bodies', element=BallElement, size=2, values={'x_d1': [[1.0, 0.0], [0.0, 1.0]]})],
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
    assert np.allclose(strips[0][-1], frames[-1].get_value(problem.elements[0].states['x'])[0])


@pytest.mark.parametrize(('count', 'trail'), [(11, 256), (2, 0)])
def test_points_visual_leaves_out_trail(count, trail):
    """Test that large groups and a trail of 0 positions log no trail."""
    system = CompositeSystem(
        elements=[GroupElement('bodies', element=BallElement, size=count)],
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


def test_points_visual_trail_width_follows_extent():
    """Test that the trail is a quarter as wide as the extent share, even when a tiny radius is given."""
    system = CompositeSystem(
        elements=[GroupElement('bodies', element=BallElement, size=2, values={'x': [[0.0, 0.0], [100.0, 0.0]]})],
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


def test_points_visual_draws_two_dimensions_with_vertical_axis_up():
    """Test that a 2D position (1, 2) is logged as (1, -2), since Rerun's 2D views point their vertical axis down."""
    system = CompositeSystem(
        elements=[GroupElement('bodies', element=BallElement, size=1, values={'x': [[1.0, 2.0]]})],
        space=ContinuousSpace(n_dim=2),
    )
    problem, frame = _first_frame(system)
    logs = Logs()
    (PointsVisual(position='x', states=(), path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    points = logs.of('bodies')[0].positions.as_arrow_array().to_pylist()
    assert points == [[1.0, -2.0]]
    trail = logs.of('bodies/trail')[0].strips.as_arrow_array().to_pylist()
    assert trail[0][-1] == [1.0, -2.0]


def test_points_visual_keeps_given_color_range():
    """Test that a given color range stays fixed and the visual does not overwrite its parameter."""
    system = CompositeSystem(
        elements=[GroupElement('bodies', element=BallElement, size=2, values={'x_d1': [[1.0, 0.0], [3.0, 0.0]]})],
        space=ContinuousSpace(n_dim=2),
    )
    problem, frame = _first_frame(system)
    visual = PointsVisual(position='x', color_by='x_d1', states=(), path=problem.elements[0].path, recording=Logs())
    visual.prepare(problem).run(frame)
    assert visual.color_range is None


def test_points_visual_sizes_single_particles_by_the_whole_drawing():
    """Test that the masses of a spring chain, one particle per element, take a radius from the extent of the chain."""
    problem, frame = _first_frame(SpringChainSystem().prepare().system_)
    logs = Logs()
    (PointsVisual(position='x', states=(), path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    assert _radius(logs.of('mass_0')[0]) > 0.01


def test_colors_start_bright_enough_to_see_on_black():
    """Test that the lowest color of the point scale is not the near-black purple of viridis."""
    problem, frame = _first_frame(
        CompositeSystem(
            elements=[
                GroupElement('bodies', element=BallElement, size=2, values={'x_d1': [[0.0, 0.0], [1.0, 0.0]]}),
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


@pytest.mark.parametrize('dimension', [2, 3])
def test_points_visual_joins_the_points_by_one_strip(dimension):
    """Test that `strip` logs one line strip through the points in member order, with their mean color."""
    positions = np.arange(4.0 * dimension).reshape(4, dimension)
    state = 'x' if dimension == 3 else 'x_d1'
    group = GroupElement('chain', element=BallElement, size=4, values={state: positions})
    system = CompositeSystem(elements=[group], space=ContinuousSpace(n_dim=dimension))
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
    group = GroupElement('chain', element=BallElement, size=4, values={'x': np.zeros((4, 3))})
    problem, frame = _first_frame(CompositeSystem(elements=[group], space=ContinuousSpace(n_dim=3)))
    logs = Logs()
    PointsVisual(position='x', states=(), trail=0, path='chain', recording=logs).prepare(problem).run(frame)
    assert logs.of('chain/strip') == []
