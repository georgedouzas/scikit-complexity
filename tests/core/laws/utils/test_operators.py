"""Test the differential and nonlocal operators of fields."""

from sage.all__sagemath_symbolics import exp, var

from skcomplex.core.laws.utils import convolve, displacement, inverse_helmholtz, inverse_laplacian, shifted


def test_inverse_laplacian_is_the_unscreened_inverse_helmholtz():
    """Test that the inverse Laplacian is the inverse Helmholtz operator with zero screening."""
    q = var('q')
    assert bool(inverse_laplacian(q) == inverse_helmholtz(q, 0))


def test_convolve_defaults_to_an_infinite_radius():
    """Test that a convolution without a radius holds an infinite one, and keeps a given radius."""
    u, kernel = var('u'), exp(-displacement(0) ** 2)
    assert convolve(u, kernel).operands()[2].is_infinity()
    assert convolve(u, kernel, 2).operands()[2] == 2


def test_displacement_names_its_axis():
    """Test that the displacement symbol of an axis is a real symbol named after the axis."""
    assert str(displacement(1)) == 'displacement_1'
    assert displacement(1).is_real()


def test_shifted_holds_one_displacement_per_axis():
    """Test that a shifted read holds its operand and one displacement per axis."""
    u, v = var('u v')
    assert shifted(u, [1, v]).operands() == [u, 1, v]
