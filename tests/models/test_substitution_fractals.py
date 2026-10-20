"""Test the Koch curve, the Cantor set, and the Sierpiński substitutions."""

import numpy as np
import pytest
import rerun.blueprint as rrb

from skcomplex.core.analysis import box_counting_dimension
from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import GeometryVisual, RerunObserver, primary_view
from skcomplex.models import create, describe, run


def test_koch_curve():
    """Test the symbols, the length, and the box-counting dimension of the Koch curve at generation 5."""
    trajectory = run('koch_curve', settings={'until': 5}).trajectory_
    curve = trajectory.structure('koch', 5)
    assert trajectory['koch.F'][-1] == 1024
    assert curve.length() == pytest.approx((4 / 3) ** 5, abs=1e-12)
    grid = curve.raster(2048)
    assert box_counting_dimension(grid, [16, 32, 64, 128, 256]) == pytest.approx(np.log(4) / np.log(3), abs=0.05)
    assert describe('koch_curve').formalism == 'rewriting system'


def test_cantor_set():
    """Test the number and the total length of the Cantor intervals at every generation."""
    trajectory = run('cantor_set', settings={'until': 8}).trajectory_
    generations = np.arange(9)
    assert np.array_equal(trajectory['cantor.size'], 2.0**generations)
    assert np.allclose(trajectory['cantor.length'], (2 / 3) ** generations, rtol=0, atol=1e-12)
    with pytest.raises(ValidationError, match='between 0 and 1/2'):
        create('cantor_set', ratio=0.6)


def test_sierpinski_carpet():
    """Test the number and the area of the squares of the carpet at generation 4."""
    trajectory = run('sierpinski_substitution', settings={'until': 4}, shape='carpet').trajectory_
    carpet = trajectory.structure('sierpinski', 4)
    assert len(carpet) == 4096
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


def test_koch_curve_in_rerun(tmp_path):
    """Test that the Koch curve logs one frame per generation and that its view is the primary view."""
    observer = RerunObserver('koch', save_path=tmp_path / 'koch.rrd', blueprint='default')
    Simulation(create('koch_curve'), observers=[observer]).run(until=5)
    assert observer.logged_frames_ == 6
    view = primary_view(create('koch_curve'))
    assert isinstance(view, rrb.Spatial2DView)
    assert str(view.origin) == 'koch'


def test_filled_and_graph_visuals(tmp_path):
    """Test that a filled patch, outlines, and a host graph are logged."""
    for name, options in [('sierpinski_substitution', {}), ('sierpinski_substitution', {'filled': False})]:
        observer = RerunObserver(name, save_path=tmp_path / f'{name}.rrd', blueprint='default')
        Simulation(create(name, **options), observers=[observer]).run(until=3)
        assert observer.logged_frames_ == 4
    assert repr(GeometryVisual()) == 'GeometryVisual(filled=None, resolution=512)'
