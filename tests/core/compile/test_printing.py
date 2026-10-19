"""Test that printed NumPy, Numba, and Lepton code evaluates every operator as Sage does."""

import math

import numpy as np
import pytest
import scipy.special
from sage.all__sagemath_symbolics import (
    abs_symbolic,
    arccos,
    arcsin,
    arctan,
    ceil,
    cos,
    cosh,
    erf,
    exp,
    floor,
    log,
    max_symbolic,
    min_symbolic,
    pi,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)

from skcomplex.core.backends import LeptonPrinter
from skcomplex.core.base import symbol
from skcomplex.core.compile import NUMPY_DIALECT, SCALAR_DIALECT, print_expressions
from skcomplex.core.model import both, either, mod, where

X, Y = symbol('x'), symbol('y')
VALUES = {X: 0.3, Y: 1.7}
OPERATORS = {
    'sum': X + Y,
    'difference': X - 2 * Y,
    'product': 3 * X * Y,
    'division': X / Y,
    'rational': 2 * X / 3,
    'negative power': X**-3,
    'square root': sqrt(X + Y),
    'inverse square root': 1 / sqrt(Y),
    'real power': Y**1.5,
    'symbolic power': Y**X,
    'negation': -((X + Y) ** 2),
    'constant': pi * X,
    'exp': exp(-X / 3),
    'log': log(Y),
    'sin': sin(X),
    'cos': cos(X),
    'tan': tan(X),
    'arcsin': arcsin(X),
    'arccos': arccos(X),
    'arctan': arctan(X),
    'sinh': sinh(X),
    'cosh': cosh(X),
    'tanh': tanh(X),
    'erf': erf(X),
    'abs': abs_symbolic(X - Y),
    'floor': floor(Y),
    'ceil': ceil(Y),
    'min': min_symbolic(X, Y),
    'max': max_symbolic(X, Y),
    'less': where(X < Y, 1, 0),
    'less or equal': where(X <= Y, 1, 0),
    'greater': where(X > Y, 1, 0),
    'greater or equal': where(X >= Y, 1, 0),
    'either': where(either(X > Y, Y > 1), 1, 0),
    'both': where(both(X > Y, Y > 1), 1, 0),
    'mod': mod(Y, X),
}
LEPTON_NAMESPACE = {
    'step': lambda value: float(value >= 0),
    'delta': lambda value: float(value == 0),
    'select': lambda condition, then, otherwise: then if condition != 0 else otherwise,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'abs': abs,
    'min': min,
    'max': max,
    **{
        name: getattr(math, name)
        for name in ('exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'erf', 'floor', 'ceil')
    },
}


def _evaluate(code, namespace):
    """Return the value of printed code in a namespace."""
    return eval(code, namespace)  # noqa: S307  # nosec B307: evaluates printed code of the test operators


def _evaluate_all(assignments, code, namespace):
    """Return the value of printed code after running the assignments of its bound subexpressions."""
    for bound, value in assignments:
        namespace[bound] = _evaluate(value, namespace)
    return float(_evaluate(code, namespace))


def _expected(expression):
    """Return the value Sage gives an expression at the test values."""
    return float(expression.subs(VALUES))


def _variables():
    """Return the test values by symbol name."""
    return {str(item): value for item, value in VALUES.items()}


@pytest.mark.parametrize('name', sorted(OPERATORS))
def test_numpy_code_evaluates_operator(name):
    """Test that NumPy code of an operator gives the Sage value."""
    assignments, (code,) = print_expressions([OPERATORS[name]], NUMPY_DIALECT)
    namespace = {'np': np, 'erf': scipy.special.erf, **_variables()}
    assert _evaluate_all(assignments, code, namespace) == pytest.approx(_expected(OPERATORS[name]), rel=1e-12)


@pytest.mark.parametrize('name', sorted(OPERATORS))
def test_numba_code_evaluates_operator(name):
    """Test that scalar code of an operator, as the Numba backend compiles it, gives the Sage value."""
    assignments, (code,) = print_expressions([OPERATORS[name]], SCALAR_DIALECT)
    namespace = {'np': np, 'math': math, **_variables()}
    assert _evaluate_all(assignments, code, namespace) == pytest.approx(_expected(OPERATORS[name]), rel=1e-12)


@pytest.mark.parametrize('name', sorted(OPERATORS))
def test_lepton_code_evaluates_operator(name):
    """Test that Lepton code of an operator, read with Lepton's functions, gives the Sage value."""
    code = LeptonPrinter().doprint(OPERATORS[name]).replace('^', '**')
    value = _evaluate(code, {**LEPTON_NAMESPACE, **_variables()})
    assert float(value) == pytest.approx(_expected(OPERATORS[name]), rel=1e-12)


def test_repeated_subexpressions_are_computed_once():
    """Test that a subexpression shared by two expressions is bound once and reused."""
    assignments, codes = print_expressions([(X + Y) ** 2, 3 * (X + Y) ** 2], NUMPY_DIALECT)
    assert assignments == [('_cse0', '(x + y)**2')]
    assert codes == ['_cse0', '3*_cse0']
