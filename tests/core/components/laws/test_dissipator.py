"""Test the `_dissipator` module of `skcomplex.core.components.laws`."""

import pytest

from skcomplex.core.components.laws import DissipatorLaw, GeneratorLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.engine._utils import symbol
from skcomplex.core.errors import ValidationError


def test_check_names_dissipators_as_symbolic_only():
    """Test that a function as the operator of a dissipator raises an error naming it as symbolic only."""
    with pytest.raises(ValidationError, match=r'need symbolic operators.*`RateLaw`'):
        DissipatorLaw(print).prepare()


def test_laws_reject_values_that_are_not_operators():
    """Test that the operator laws check their operator and rate."""
    with pytest.raises(ValidationError, match='generator takes'):
        GeneratorLaw(1.0).prepare()
    with pytest.raises(ValidationError, match='dissipator takes'):
        DissipatorLaw('q').prepare()
    with pytest.raises(ValidationError, match='rate'):
        DissipatorLaw(local_operator(symbol('q'), 'minus'), local_operator(symbol('q'), 'z')).prepare()
