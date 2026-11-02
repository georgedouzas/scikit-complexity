"""Test the `module_spec` module of `skcomplex.models.mathematics.fractals`."""

import typing

import pytest

from skcomplex.models.mathematics.fractals import Letters, ModuleSpec, RewritingError


@pytest.mark.parametrize('spec', [ModuleSpec('')])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that a module rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec.check()


def test_letters_may_be_modules():
    """Test that letters may be modules or sequences of them."""
    assert ModuleSpec in typing.get_args(Letters.__value__)
