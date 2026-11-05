"""Test the `_linear_objective_spec` module of `skcomplex.specs`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.errors import ValidationError
from skcomplex.specs import LinearObjectiveSpec


def test_check_takes_a_known_sense():
    """Test that both senses pass, and that maximising is the default."""
    assert LinearObjectiveSpec(var('v')).sense == 'max'
    assert LinearObjectiveSpec(var('v'), sense='min').sense == 'min'


def test_check_rejects_an_unknown_sense():
    """Test that an unknown sense raises an error that names the senses."""
    with pytest.raises(ValidationError, match='max, min'):
        LinearObjectiveSpec(var('v'), sense='most').prepare()
