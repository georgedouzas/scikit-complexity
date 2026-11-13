"""Test the `base` module."""

import inspect

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine import compile as package
from skcomplex.core.engine.compile import BaseCompiler, BasePass, OperatorCompiler
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile._rewriting_compiler import RewritingCompiler


def test_base_compiler_resolve_backend():
    """Test `BaseCompiler._resolve_backend` on a small case."""
    assert type(BaseCompiler._resolve_backend(None, 10)).__name__ == 'NumpyBackend'


def test_base_pass_get_replaced():
    """Test `BasePass._get_replaced` on a small case."""
    x = var('x')
    assert str(BasePass._get_replaced(RateLaw(x, -x), value=-2 * x).value) == '-2*x'


def test_public_names_are_classes():
    """Every public name is a class."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)), name


def test_bases_and_the_operator_compiler_are_public():
    """The bases and the operator compiler are public, and the other compilers and the passes are private."""
    assert set(package.__all__) == {'BaseCompiler', 'BasePass', 'BaseLowering', 'HilbertSpaceError', 'OperatorCompiler'}
    for compiler in (PassCompiler, OperatorCompiler, RewritingCompiler):
        assert issubclass(compiler, package.BaseCompiler)
    for compiler in (PassCompiler, RewritingCompiler):
        assert not hasattr(package, compiler.__name__)
    for name in ('FlattenPass', 'LayoutPass', 'ProblemPass', 'Compilation', 'ObservableEvaluator'):
        assert not hasattr(package, name)
