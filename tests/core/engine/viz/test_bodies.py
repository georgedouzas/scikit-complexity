"""Test the visual of bodies, rods, trails, and outlines."""

from dataclasses import replace

import numpy as np
import pytest

from skcomplex.core.engine.runtime import BaseObserver, simulate
from skcomplex.core.engine.viz import BodiesVisual, RerunObserver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import VisualSpec
from skcomplex.systems.physics.celestial_mechanics import StormerProblemSystem
from skcomplex.systems.physics.mechanics import CircularBilliardSystem, DoublePendulumSystem


class Logs:
    """Recording stand-in that keeps the entity paths, archetypes, and static flags it receives."""

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


def _frames(system, steps, dt):
    """Return the problem and the frames of a run of a system."""
    frames = []

    class KeepObserver(BaseObserver):
        """Observer that keeps every frame."""

        def _run(self, inputs):
            """Keep a frame."""
            if inputs is not None:
                frames.append(inputs)

    trajectory = simulate(system, until=steps * dt, dt=dt, observers=[KeepObserver()])
    return trajectory.problem, frames


def bobs(theta1, theta2, l1, l2, **_):
    """Return the two bobs of a double pendulum hanging from the origin."""
    first = np.array([l1 * np.sin(theta1), -l1 * np.cos(theta1)])
    return np.stack([first, np.add(first, [l2 * np.sin(theta2), -l2 * np.cos(theta2)])])


def _points(archetype, field):
    """Return the points of a logged archetype field as a list."""
    return getattr(archetype, field).as_arrow_array().to_pylist()


def test_bodies_visual_draws_double_pendulum():
    """Test that bobs, a rod strip from the pivot, a fading trail of the lower bob, and the angles are logged."""
    problem, frames = _frames(DoublePendulumSystem().prepare().system_, steps=20, dt=0.01)
    logs = Logs()
    visual = BodiesVisual(bobs, pivot=(0.0, 0.0), rods=True, traced=[1], path=problem.elements[0].path, recording=logs)
    visual.prepare(problem)
    for frame in frames:
        visual.run(frame)
    element = problem.elements[0]
    last = bobs(
        **{name: frames[-1].get_value(dotted) for name, dotted in {**element.parameters, **element.states}.items()},
    )
    bodies = _points(logs.of('pendulum/bodies')[-1], 'positions')
    np.testing.assert_allclose(bodies, last * [1, -1])
    rods = _points(logs.of('pendulum/rods')[-1], 'strips')
    assert len(rods) == 1
    np.testing.assert_allclose(rods[0], [[0.0, 0.0], *(last * [1, -1])])
    trail = logs.of('pendulum/trail')[-1]
    alphas = [color & 0xFF for color in trail.colors.as_arrow_array().to_pylist()]
    assert len(alphas) == len(frames) - 1
    assert alphas == sorted(alphas)
    assert alphas[0] < alphas[-1]
    assert logs.of('pendulum/theta1')


def test_bodies_visual_logs_outline_once():
    """Test that the outline of a billiard table is logged once as a static line strip."""
    problem, frames = _frames(CircularBilliardSystem().prepare().system_, steps=3, dt=0.05)
    circle = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 50)), np.sin(np.linspace(0, 2 * np.pi, 50))])
    logs = Logs()
    visual = BodiesVisual(('x', 'y'), outline=[circle], path=problem.elements[0].path, recording=logs).prepare(problem)
    for frame in frames:
        visual.run(frame)
    assert len(logs.of('particle/outline')) == 1
    assert logs.static.count('particle/outline') == 1


def test_bodies_visual_bounds_cover_rods_at_full_length():
    """Test that the spatial view of a pendulum covers the pivot and the rods at full length."""
    problem = simulate(DoublePendulumSystem().prepare().system_, until=0.0).problem
    views = BodiesVisual(bobs, pivot=(0.0, 0.0), rods=True, path=problem.elements[0].path).prepare(problem).views_
    assert [type(view).__name__ for view in views] == ['Spatial2DView', 'TimeSeriesView']


def test_bodies_visual_is_registered():
    """Test that a Rerun observer draws an element whose visual spec names `bodies` with the bodies visual."""
    problem = simulate(CircularBilliardSystem().prepare().system_, until=0).problem
    element = replace(problem.elements[0], visual=VisualSpec('bodies', {'positions': ('x', 'y')}))
    problem = replace(problem, elements=(element, *problem.elements[1:]))
    assert isinstance(RerunObserver().prepare(problem).visuals_[0], BodiesVisual)


def test_bodies_visual_rejects_positions_of_wrong_shape():
    """Test that positions that are not pairs of coordinates raise a named error."""
    problem, frames = _frames(CircularBilliardSystem().prepare().system_, steps=1, dt=0.05)
    visual = BodiesVisual(lambda x, **_: np.array([x, x, x, x]), path=problem.elements[0].path, recording=Logs())
    with pytest.raises(ValidationError, match='pairs of coordinates'):
        visual.prepare(problem).run(frames[0])


def test_bodies_visual_draws_a_path_in_three_dimensions():
    """Test that a body with three coordinates is drawn as a 3D point with a 3D fading path in a 3D view."""
    problem, frames = _frames(StormerProblemSystem().prepare().system_, steps=5, dt=0.02)
    logs = Logs()
    visual = BodiesVisual(('x', 'y', 'z'), path=problem.elements[0].path, recording=logs).prepare(problem)
    for frame in frames:
        visual.run(frame)
    assert type(logs.of('charge/bodies')[-1]).__name__ == 'Points3D'
    assert type(logs.of('charge/trail')[-1]).__name__ == 'LineStrips3D'
    assert type(visual.views_[0]).__name__ == 'Spatial3DView'
