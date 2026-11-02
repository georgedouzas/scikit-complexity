"""Test the differential and nonlocal operators of fields."""

import pytest
from sage.all__sagemath_symbolics import exp, var

from skcomplex.core.laws.utils import (
    INVERSION_BOUNDARIES,
    convolve,
    displacement,
    inverse_helmholtz,
    inverse_laplacian,
    shifted,
)
from skcomplex.errors import ValidationError


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


def test_inverse_helmholtz_holds_the_code_and_value_of_its_boundary():
    """Test that an inversion with a boundary holds its position in the boundaries and its fixed value."""
    q = var('q')
    assert inverse_helmholtz(q, 1, ('fixed', 0.5)).operands()[2:] == [INVERSION_BOUNDARIES.index('fixed'), 0.5]
    assert inverse_helmholtz(q, 1, 'zero-flux').operands()[2:] == [1, 0]


@pytest.mark.parametrize('boundary', ['fixed', ('zero-flux', 1.0), 'walls'])
def test_inverse_helmholtz_rejects_an_unknown_boundary(boundary):
    """Test that an inversion rejects an unknown boundary, a fixed one without value, and a value of another one."""
    with pytest.raises(ValidationError, match='boundary of the inversion'):
        inverse_helmholtz(var('q'), 1, boundary)


def test_shifted_holds_the_cubic_order():
    """Test that a cubic shifted read holds its order after the displacement, and a linear one holds none."""
    u = var('u')
    assert shifted(u, [0.5], order=3).operands() == [u, 0.5, 3]
    with pytest.raises(ValidationError, match='order of a shifted read'):
        shifted(u, [0.5], order=2)
