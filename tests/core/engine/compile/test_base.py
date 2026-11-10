"""Test the `base` module."""

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.compile import BaseCompiler, BasePass


def test_base_compiler_resolve_backend():
    """Test `BaseCompiler._resolve_backend` on a small case."""
    assert type(BaseCompiler._resolve_backend(None, 10)).__name__ == 'NumpyBackend'


def test_base_pass_get_replaced():
    """Test `BasePass._get_replaced` on a small case."""
    x = var('x')
    assert str(BasePass._get_replaced(RateLaw(x, -x), value=-2 * x).value) == '-2*x'
