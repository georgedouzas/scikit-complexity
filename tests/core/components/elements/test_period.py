"""Test the `_period` module of `skcomplex.core.components.elements`."""

import math

import numpy as np
import pytest
from sage.all__sagemath_symbolics import pi, var

from skcomplex.core.components.elements import Period
from skcomplex.core.errors import ValidationError


@pytest.mark.parametrize('length', [2 * math.pi, 2 * pi, np.float64(1.5), 3])
def test_period_accepts_a_positive_number_of_python_numpy_or_sage(length):
    """Test that a positive length of Python, NumPy, or Sage, such as `2 * pi` with the `pi` of Sage, is accepted."""
    Period(length)._check()


@pytest.mark.parametrize('length', [0, -1.0, math.inf, True, '2', var('L'), 1j])
def test_period_rejects_a_length_that_is_no_positive_finite_number(length):
    """Test that a length that is not a positive finite number is rejected with the form it suggests."""
    with pytest.raises(ValidationError, match=r'such as Period\(2 \* pi\)'):
        Period(length)._check()
