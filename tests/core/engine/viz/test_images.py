"""Test the `_images` module of `skcomplex.core.engine.viz`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.viz import ImageVisual
from skcomplex.core.errors import ValidationError
from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
from skcomplex.systems.physics.fluids import BoltzmannEquationSystem, VlasovEquationSystem
from skcomplex.systems.physics.statistical_mechanics import XYSpinElement

from .conftest import Logs, _first_frame, _frames


def _format(archetype):
    """Return the width and the height of a logged image."""
    image_format = archetype.format.as_arrow_array()[0].as_py()
    return image_format['width'], image_format['height']


def test_views_of_image_without_history_leave_out_space_time():
    """Test that a history of 0 rows declares no space-time view."""
    problem = PassCompiler().prepare(ElementaryAutomatonSystem().prepare().system_).run().problem_
    views = ImageVisual(history=0, path=problem.elements[0].path).prepare(problem).views_
    assert [str(view.origin) for view in views] == ['cells/alive']


def test_image_visual_logs_space_time_of_one_dimensional_lattice():
    """Test that a one-dimensional lattice logs a space-time image that grows by one row per frame."""
    problem, frames = _frames(ElementaryAutomatonSystem(size=21).prepare().system_, steps=5, dt=1.0)
    logs = Logs()
    visual = ImageVisual(path=problem.elements[0].path, recording=logs).prepare(problem)
    for frame in frames:
        visual.run(frame)
    images = logs.of('cells/space_time/alive')
    assert [type(image).__name__ for image in images] == ['DepthImage'] * len(frames)
    assert [_format(image) for image in images] == [(21, rows) for rows in range(1, len(frames) + 1)]


def test_image_visual_bounds_space_time_rows():
    """Test that the space-time image keeps only the most recent rows."""
    problem, frames = _frames(ElementaryAutomatonSystem(size=21).prepare().system_, steps=5, dt=1.0)
    logs = Logs()
    visual = ImageVisual(history=3, path=problem.elements[0].path, recording=logs).prepare(problem)
    for frame in frames:
        visual.run(frame)
    assert _format(logs.of('cells/space_time/alive')[-1]) == (21, 3)


def test_image_visual_logs_one_row_as_rgb_image():
    """Test that the current row of a one-dimensional lattice is an image one pixel high, not a column of grey."""
    problem, frame = _first_frame(ElementaryAutomatonSystem(size=21).prepare().system_)
    logs = Logs()
    ImageVisual(history=0, path=problem.elements[0].path, recording=logs).prepare(problem).run(frame)
    assert [path for path, _ in logs.logs] == ['cells/alive']
    image = logs.of('cells/alive')[0].format.as_arrow_array()[0].as_py()
    assert (image['width'], image['height'], image['color_model']) == (21, 1, 2)


def test_image_visual_colors_angles_cyclically():
    """Test that angles just either side of ±π get nearly the same color, as on a cyclic palette."""
    space = LatticeSpace((2, 2))
    system = CompositeSystem(
        elements=[
            GroupElement(
                'spins',
                element=XYSpinElement,
                space=space,
                values={'theta': [[-3.1, 3.1], [0.0, 1.5]]},
                on='sites',
            ),
        ],
    )
    problem, frame = _first_frame(system)
    logs = Logs()
    ImageVisual(path=problem.elements[0].path, recording=logs).prepare(problem).run(frame)
    image = logs.of('spins/theta')[0]
    pixels = np.asarray(image.buffer.as_arrow_array()[0].as_py(), dtype=int).reshape(4, 3)
    assert np.abs(pixels[0] - pixels[1]).max() < 20
    assert np.abs(pixels[0] - pixels[3]).max() > 60


def test_image_visual_logs_position_velocity_image_and_density_of_two_stream():
    """Test that a two-stream system logs its position-velocity image and its density over position every frame."""
    system = VlasovEquationSystem(positions=8, velocities=16, setup='two-stream').prepare().system_
    problem, frames = _frames(system, steps=3, dt=0.1)
    logs = Logs()
    visual = ImageVisual(moments=(0,), history=0, path='gas', recording=logs).prepare(problem)
    for frame in frames:
        visual.run(frame)
    images = logs.of('gas/x')
    assert [_format(image) for image in images] == [(16, 8)] * len(frames)
    densities = logs.of('gas/x/moment_0')
    assert [type(item).__name__ for item in densities] == ['BarChart'] * len(frames)
    assert len(densities[-1].values.as_arrow_array().to_pylist()[0]['buffer']) == 8
    assert [str(view.origin) for view in visual.views_] == ['gas/x/moment_0', 'gas/x']


def test_image_visual_logs_a_plane_of_a_phase_space():
    """Test that a phase space of three axes logs the plane its slices leave."""
    system = BoltzmannEquationSystem(positions=4, velocities=8).prepare().system_
    problem, frame = _first_frame(system)
    logs = Logs()
    visual = ImageVisual(slices={'vy': 2}, moments=(0,), path='gas', recording=logs).prepare(problem)
    visual.run(frame)
    assert [_format(image) for image in logs.of('gas/x')] == [(8, 4)]
    assert len(logs.of('gas/x/moment_0')) == 1


def test_image_visual_rejects_slices_that_leave_other_than_two_axes():
    """Test that slices must leave two axes, and that they need a phase space."""
    problem, _ = _first_frame(BoltzmannEquationSystem(positions=4, velocities=8).prepare().system_)
    with pytest.raises(ValidationError, match='every axis but two'):
        ImageVisual(slices={'vx': 1, 'vy': 2}, path='gas').prepare(problem)
    problem, _ = _first_frame(GrayScottSystem(shape=(4, 4)).prepare().system_)
    with pytest.raises(ValidationError, match='phase space'):
        ImageVisual(slices={0: 1}, path=problem.elements[0].path).prepare(problem)
