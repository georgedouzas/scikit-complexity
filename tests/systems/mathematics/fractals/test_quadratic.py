"""Test the `quadratic` module of `skcomplex.systems.mathematics.fractals`."""

import numpy as np

from skcomplex.systems.mathematics.fractals import inverse_julia, julia, mandelbrot

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


def test_inverse_julia_points_settle_on_the_unit_circle_for_zero_constant():
    """Test that the random square roots of `z` carry every point onto the Julia set `|z| = 1` of `z²`."""
    z = _run(inverse_julia(0.0, n_points=200), 30).values('points.z')[-1]
    np.testing.assert_allclose(np.abs(z), 1.0, atol=1e-6)


def test_inverse_julia_points_stay_on_the_invariant_set_of_the_rabbit():
    """Test that the points of Douady's rabbit stay bounded under the forward map and spread over the whole set."""
    omega = -0.123 + 0.745j
    z = _run(inverse_julia(omega, n_points=500), 30).values('points.z')[-1]
    forward = z.copy()
    for _ in range(20):
        forward = forward**2 + omega
    assert np.abs(forward).max() < 2
    assert np.ptp(z.real) > 1.5
