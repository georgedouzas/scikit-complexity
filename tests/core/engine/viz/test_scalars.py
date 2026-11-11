"""Test the `_scalars` module of `skcomplex.core.engine.viz`."""

from skcomplex.core.engine.viz import ScalarsVisual
from skcomplex.systems.biology.epidemiology import DelayedSIRSSystem
from skcomplex.systems.physics.climate import DaisyworldSystem, Lorenz96System
from skcomplex.systems.physics.mechanics import CoupledPendulumsSystem

from .conftest import Logs, _first_frame


def _series_color(archetype):
    """Return the color of logged series lines."""
    return archetype.colors.as_arrow_array().to_pylist()[0]


def test_scalars_visual_gives_states_distinct_colors():
    """Test that the white and black daisies of Daisyworld get distinct static series colors."""
    problem, frame = _first_frame(DaisyworldSystem().prepare().system_)
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
    problem, frame = _first_frame(CoupledPendulumsSystem().prepare().system_)
    logs = Logs()
    for element in problem.elements:
        ScalarsVisual(path=element.path, recording=logs).prepare(frame.problem).run(frame)
    left = next(item for item in logs.of('left/theta') if type(item).__name__ == 'SeriesLines')
    right = next(item for item in logs.of('right/theta') if type(item).__name__ == 'SeriesLines')
    assert _series_color(left) != _series_color(right)


def test_scalars_visual_leaves_out_lag_companions():
    """Test that the past values a delayed system keeps as lag states are not logged as series."""
    problem, frame = _first_frame(DelayedSIRSSystem().prepare().system_)
    logs = Logs()
    ScalarsVisual(path=problem.elements[0].path, recording=logs).prepare(problem).run(frame)
    paths = {path for path, _ in logs.logs}
    assert 'first/x' in paths
    assert not any('_lag' in path for path in paths)


def test_scalars_visual_logs_chosen_components():
    """Test that a scalars visual with chosen components logs only those components of a vector state."""
    problem, frame = _first_frame(Lorenz96System().prepare().system_)
    logs = Logs()
    (ScalarsVisual(components=[0, 1, 2], path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    scalars = [item for item in logs.of('ring/x') if type(item).__name__ == 'Scalars']
    assert len(scalars[0].scalars.as_arrow_array().to_pylist()) == 3
