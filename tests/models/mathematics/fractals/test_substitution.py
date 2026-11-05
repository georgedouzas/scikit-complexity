"""Test the `_substitution` module of `skcomplex.models.mathematics.fractals`."""

import numpy as np
import pytest
import rerun.blueprint as rrb

from skcomplex.analysis import ScalingAnalysis
from skcomplex.engine.compile import RewritingCompiler
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import GeometryVisual, RerunObserver
from skcomplex.errors import ValidationError
from skcomplex.models import create, describe, run


def test_koch_curve():
    """Test the symbols, the length, and the box-counting dimension of the Koch curve at generation 5."""
    trajectory = run('koch_curve', settings={'until': 5}).trajectory_
    curve = trajectory.structure('koch', 5)
    assert trajectory.values('koch.F')[-1] == 1024
    assert curve.length() == pytest.approx((4 / 3) ** 5, abs=1e-12)
    grid = curve.raster(2048)
    dimension = ScalingAnalysis('box-counting', scales=[16, 32, 64, 128, 256]).prepare(grid).run().exponent_
    assert dimension == pytest.approx(np.log(4) / np.log(3), abs=0.05)
    assert describe('koch_curve').formalism == 'rewriting system'


def test_cantor_set():
    """Test the number and the total length of the Cantor intervals at every generation."""
    trajectory = run('cantor_set', settings={'until': 8}).trajectory_
    generations = np.arange(9)
    assert np.array_equal(trajectory.values('cantor.size'), 2.0**generations)
    assert np.allclose(trajectory.values('cantor.length'), (2 / 3) ** generations, rtol=0, atol=1e-12)
    with pytest.raises(ValidationError, match='between 0 and 1/2'):
        create('cantor_set', ratio=0.6)


def test_sierpinski_carpet():
    """Test the number and the area of the squares of the carpet at generation 4."""
    trajectory = run('sierpinski_substitution', settings={'until': 4}, shape='carpet').trajectory_
    carpet = trajectory.structure('sierpinski', 4)
    assert carpet.parts == 4096
    assert carpet.area() == pytest.approx((8 / 9) ** 4, abs=1e-12)
    with pytest.raises(ValidationError, match='Unknown shape'):
        create('sierpinski_substitution', shape='hexagon')


def test_filled_gasket():
    """Test that the raster of the filled gasket covers (3/4)^6 of the pixels of the triangle."""
    trajectory = run('sierpinski_substitution', settings={'until': 6}).trajectory_
    first, last = trajectory.structure('sierpinski', 0), trajectory.structure('sierpinski', 6)
    bounds = first.bounds()
    share = last.raster(512, bounds).sum() / first.raster(512, bounds).sum()
    assert share == pytest.approx(0.75**6, rel=0.02)


def test_koch_curve_reference_values():
    """Test the word, the counts, the length, and the points of the Koch curve at generation 2 against literals."""
    trajectory = run('koch_curve', settings={'until': 2}).trajectory_
    curve = trajectory.structure('koch', 2)
    assert curve.text == 'F+F--F+F+F+F--F+F--F+F--F+F+F+F--F+F'
    assert trajectory.values('koch.F').tolist() == [1.0, 4.0, 16.0]
    assert trajectory.values('koch.size').tolist() == [1.0, 8.0, 36.0]
    assert curve.length() == pytest.approx(1.777777777777778, abs=1e-12)
    assert curve.bounds() == pytest.approx((0.0, 0.0, 1.0, 0.28867513459481287), abs=1e-12)
    segments = curve.segments()
    assert len(segments) == 16
    points = [
        [[0.111111111111, 0.0], [0.166666666667, 0.096225044865]],
        [[0.444444444444, 0.19245008973], [0.5, 0.288675134595]],
        [[0.888888888889, 0.0], [1.0, 0.0]],
    ]
    assert np.allclose(segments[[1, 7, 15]], points, rtol=0, atol=1e-11)


def test_cantor_set_reference_values():
    """Test the counts, the lengths, and the intervals of the Cantor set at generation 2 against literal values."""
    trajectory = run('cantor_set', settings={'until': 2}).trajectory_
    assert trajectory.values('cantor.size').tolist() == [1.0, 2.0, 4.0]
    assert trajectory.values('cantor.length').tolist() == pytest.approx([1.0, 0.6666666666666665, 0.4444444444444444])
    intervals = trajectory.structure('cantor', 2).segments()[:, :, 0].tolist()
    expected = [
        [0.0, 0.111111111111],
        [0.222222222222, 0.333333333333],
        [0.666666666667, 0.777777777778],
        [0.888888888889, 1.0],
    ]
    assert np.allclose(intervals, expected, rtol=0, atol=1e-11)


def test_sierpinski_substitution_reference_values():
    """Test the counts, the areas, the rasters, and the corners of the gasket and the carpet against literal values."""
    trajectory = run('sierpinski_substitution', settings={'until': 2}).trajectory_
    gasket = trajectory.structure('sierpinski', 2)
    assert trajectory.values('sierpinski.size').tolist() == [1.0, 3.0, 9.0]
    assert gasket.area() == pytest.approx(0.24356964481437335, abs=1e-12)
    assert gasket.bounds() == pytest.approx((0.0, 0.0, 1.0, 0.8660254037844386), abs=1e-12)
    assert gasket.raster(32).sum() == 264
    corners = [[[0.0, 0.0], [0.25, 0.0], [0.125, 0.216506350946]], [[0.25, 0.0], [0.5, 0.0], [0.375, 0.216506350946]]]
    assert np.allclose(gasket.polygons()[0][:2], corners, rtol=0, atol=1e-11)
    trajectory = run('sierpinski_substitution', settings={'until': 2}, shape='carpet').trajectory_
    carpet = trajectory.structure('sierpinski', 2)
    assert trajectory.values('sierpinski.size').tolist() == [1.0, 8.0, 64.0]
    assert carpet.area() == pytest.approx(0.7901234567901236, abs=1e-12)
    assert carpet.raster(27).sum() == 576


def test_koch_curve_in_rerun(tmp_path):
    """Test that the Koch curve logs one frame per generation and that its view is the primary view."""
    observer = RerunObserver('koch', save_path=tmp_path / 'koch.rrd', blueprint='default')
    Simulation(create('koch_curve'), observers=[observer], compiler=RewritingCompiler()).prepare().run(until=5)
    assert observer.logged_frames_ == 6
    view = observer.views_[0]
    assert isinstance(view, rrb.Spatial2DView)
    assert str(view.origin) == 'koch'


def test_filled_and_outlined_visuals(tmp_path):
    """Test that a filled patch and its outlines are logged."""
    for name, options in [('sierpinski_substitution', {}), ('sierpinski_substitution', {'filled': False})]:
        observer = RerunObserver(name, save_path=tmp_path / f'{name}.rrd', blueprint='default')
        Simulation(create(name, **options), observers=[observer], compiler=RewritingCompiler()).prepare().run(
            until=3,
        )
        assert observer.logged_frames_ == 4
    assert repr(GeometryVisual()) == 'GeometryVisual()'
