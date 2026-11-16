"""Test the `_bilinear` module of `skcomplex.core.components.laws.utils`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws.utils import bilinear
from skcomplex.core.components.laws.utils._bilinear import bilinears_in, unpack_bilinear
from skcomplex.core.errors import ValidationError

F, G, MU, Q = var('f g mu q')


def test_bilinear_keeps_its_densities_tables_and_conservation():
    """Test that a bilinear form gives back its densities, its tables, and whether it is conserved."""
    first, second, rate, outcomes, conserved = unpack_bilinear(bilinear(-2 * MU, None, F, G))
    assert (first, second, rate, outcomes, conserved) == (F, G, -2 * MU, None, False)
    table = np.full((2, 2, 2), 0.5)
    first, second, rate, outcomes, conserved = unpack_bilinear(
        bilinear([[1, 2], [3, Q]], table, F, G, conserved=True),
    )
    assert (first, second, conserved) == (F, G, True)
    assert rate.tolist() == [[1, 2], [3, Q]]
    assert outcomes.shape == (2, 2, 2)
    assert np.asarray(outcomes, dtype=float).tolist() == table.tolist()


def test_bilinear_without_a_second_density_is_a_redistribution():
    """Test that a conserved form without a second density keeps none."""
    assert unpack_bilinear(bilinear(1.0, np.ones((2, 1, 2)) / 2, F, conserved=True))[1] is None


def test_bilinear_is_found_in_expressions():
    """Test that the bilinear forms of an expression are found once each."""
    form = bilinear(1.0, None, F, G)
    assert bilinears_in(2 * form + form) == [form]
    assert bilinears_in(F * G) == []


@pytest.mark.parametrize(
    ('rate', 'outcomes', 'message'),
    [
        (1.0, np.ones((2, 2)), 'must be an array of 3 dimensions'),
        (np.ones(3), np.ones((2, 2, 2)), 'must be an array of 0 or 2 dimensions'),
        (np.ones((3, 2)), np.ones((2, 2, 2)), 'has the shape'),
    ],
)
def test_bilinear_rejects_tables_of_wrong_shape(rate, outcomes, message):
    """Test that a form rejects tables of the wrong number of dimensions or of disagreeing classes."""
    with pytest.raises(ValidationError, match=message):
        bilinear(rate, outcomes, F, G, conserved=True)


def test_conserved_bilinear_needs_outcomes():
    """Test that a conserved form without an outcome table asks for the outcome probabilities."""
    with pytest.raises(ValidationError, match='Give the outcome probabilities'):
        bilinear(1.0, None, F, G, conserved=True)


def test_only_a_conserved_bilinear_takes_no_second_density():
    """Test that a form without a second density must be conserved."""
    with pytest.raises(ValidationError, match='only a conserved form takes'):
        bilinear(1.0, np.ones((2, 1, 2)) / 2, F)
