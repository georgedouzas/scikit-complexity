"""Test the `specs` module."""

from sage.all__sagemath_symbolics import var

from skcomplex.core.engine.compile.passes._specs import SpecsPass


def test_specs_pass_get_end_symbols():
    """Test `SpecsPass._get_end_symbols` on a small case."""
    assert str(SpecsPass._get_end_symbols(var('x_cells__source'))) == '(x_cells, x_cells__other)'
