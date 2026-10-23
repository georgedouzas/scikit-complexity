"""Test the `_quadratic` module of `skcomplex.models.mathematics.fractals`."""

import numpy as np

from skcomplex.models.mathematics.fractals import julia, mandelbrot

from ..conftest import _run


def test_mandelbrot_escape_times():
    """Test that `c = 1` escapes at step 3 and that `c = -1`, `c = -2`, and `c = i` never escape in 100 steps."""
    escapes = _run(mandelbrot((5, 5), window=(-2, 2, -2, 2)), 100).lattice('plane.n')[-1]
    assert escapes[2, 3] == 3
    assert escapes[2, 1] == escapes[2, 0] == escapes[1, 2] == 100


def test_julia_set_of_squaring_map_is_unit_disk():
    """Test that the filled Julia set of `z -> z^2` keeps the points inside the unit circle and loses those outside."""
    trajectory = _run(julia(0.0, (201, 201), window=(-2, 2, -2, 2)), 100)
    modulus, escapes = np.abs(trajectory.lattice('plane.z')[0]), trajectory.lattice('plane.n')[-1]
    assert np.all(escapes[modulus < 0.99] == 100)
    assert np.all(escapes[modulus > 1.01] < 100)


def test_mandelbrot_is_symmetric_about_real_axis():
    """Test that the escape times on a window symmetric about the real axis equal their mirror image."""
    escapes = _run(mandelbrot((61, 80), window=(-2.2, 0.8, -1.2, 1.2)), 60).lattice('plane.n')[-1]
    assert np.array_equal(escapes, escapes[::-1])
    assert escapes.min() < escapes.max()
