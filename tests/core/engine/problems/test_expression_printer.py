"""Test the `_expression_printer` module of `skcomplex.core.engine.problems`."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest
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

from skcomplex.core.components.laws.utils import both, either, mod, where
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.problems._expression_printer import ExpressionPrinter

X, Y = symbol('x'), symbol('y')
OPERATORS = {
    'sum': (X + Y, 'x + y', 'x + y', 'x + y'),
    'difference': (X - 2 * Y, 'x - 2*y', 'x - 2*y', 'x - 2*y'),
    'product': (3 * X * Y, '3*x*y', '3*x*y', '3*x*y'),
    'division': (X / Y, 'x/y', 'x/y', 'x/y'),
    'rational': (2 * X / 3, '2*x/3', '2*x/3', '2*x/3'),
    'negative power': (X**-3, '1/x**3', '1/x**3', '1/x^3'),
    'square root': (sqrt(X + Y), 'np.sqrt(x + y)', 'math.sqrt(x + y)', 'sqrt(x + y)'),
    'inverse square root': (1 / sqrt(Y), '1/np.sqrt(y)', '1/math.sqrt(y)', '1/sqrt(y)'),
    'real power': (Y**1.5, 'y**1.5', 'y**1.5', 'y^1.5'),
    'symbolic power': (Y**X, 'y**x', 'y**x', 'y^x'),
    'negation': (-((X + Y) ** 2), '-(x + y)**2', '-(x + y)**2', '-(x + y)^2'),
    'constant': (pi * X, '3.141592653589793*x', '3.141592653589793*x', '3.141592653589793*x'),
    'exp': (exp(-X / 3), 'np.exp(-x/3)', 'math.exp(-x/3)', 'exp(-x/3)'),
    'log': (log(Y), 'np.log(y)', 'math.log(y)', 'log(y)'),
    'sin': (sin(X), 'np.sin(x)', 'math.sin(x)', 'sin(x)'),
    'cos': (cos(X), 'np.cos(x)', 'math.cos(x)', 'cos(x)'),
    'tan': (tan(X), 'np.tan(x)', 'math.tan(x)', 'tan(x)'),
    'arcsin': (arcsin(X), 'np.arcsin(x)', 'math.asin(x)', 'asin(x)'),
    'arccos': (arccos(X), 'np.arccos(x)', 'math.acos(x)', 'acos(x)'),
    'arctan': (arctan(X), 'np.arctan(x)', 'math.atan(x)', 'atan(x)'),
    'sinh': (sinh(X), 'np.sinh(x)', 'math.sinh(x)', 'sinh(x)'),
    'cosh': (cosh(X), 'np.cosh(x)', 'math.cosh(x)', 'cosh(x)'),
    'tanh': (tanh(X), 'np.tanh(x)', 'math.tanh(x)', 'tanh(x)'),
    'erf': (erf(X), 'erf(x)', 'math.erf(x)', 'erf(x)'),
    'abs': (abs_symbolic(X - Y), 'np.abs(x - y)', 'abs(x - y)', 'abs(x - y)'),
    'floor': (floor(Y), 'np.floor(y)', 'math.floor(y)', 'floor(y)'),
    'ceil': (ceil(Y), 'np.ceil(y)', 'math.ceil(y)', 'ceil(y)'),
    'min': (min_symbolic(X, Y), 'np.minimum(x, y)', 'min(x, y)', 'min(x, y)'),
    'max': (max_symbolic(X, Y), 'np.maximum(x, y)', 'max(x, y)', 'max(x, y)'),
    'less': (
        where(X < Y, 1, 0),
        'np.where((x < y), 1, 0)',
        '(1 if (x < y) else 0)',
        'select((1 - step(x - y)), 1, 0)',
    ),
    'less or equal': (
        where(X <= Y, 1, 0),
        'np.where((x <= y), 1, 0)',
        '(1 if (x <= y) else 0)',
        'select(step(y - x), 1, 0)',
    ),
    'greater': (
        where(X > Y, 1, 0),
        'np.where((x > y), 1, 0)',
        '(1 if (x > y) else 0)',
        'select((1 - step(y - x)), 1, 0)',
    ),
    'greater or equal': (
        where(X >= Y, 1, 0),
        'np.where((x >= y), 1, 0)',
        '(1 if (x >= y) else 0)',
        'select(step(x - y), 1, 0)',
    ),
    'either': (
        where(either(X > Y, Y > 1), 1, 0),
        'np.where(np.logical_or((x > y), (y > 1)), 1, 0)',
        '(1 if ((x > y) or (y > 1)) else 0)',
        'select(max((1 - step(y - x)), (1 - step(1 - y))), 1, 0)',
    ),
    'both': (
        where(both(X > Y, Y > 1), 1, 0),
        'np.where(np.logical_and((x > y), (y > 1)), 1, 0)',
        '(1 if ((x > y) and (y > 1)) else 0)',
        'select(((1 - step(y - x))*(1 - step(1 - y))), 1, 0)',
    ),
    'mod': (mod(Y, X), 'np.mod(y, x)', '(y % x)', '(y - x*floor(y/x))'),
}


@pytest.mark.parametrize('name', sorted(OPERATORS))
def test_numpy_code_of_operator(name):
    """Test the NumPy code of an operator."""
    expression, code, _, _ = OPERATORS[name]
    assert ExpressionPrinter().prepare().run([expression]).code_ == ([], [code])


@pytest.mark.parametrize('name', sorted(OPERATORS))
def test_numba_code_of_operator(name):
    """Test the scalar code of an operator, as the Numba backend compiles it."""
    expression, _, code, _ = OPERATORS[name]
    assert ExpressionPrinter('scalar').prepare().run([expression]).code_ == ([], [code])


@pytest.mark.parametrize('name', sorted(OPERATORS))
def test_lepton_code_of_operator(name):
    """Test the Lepton code of an operator, written with Lepton's functions."""
    expression, _, _, code = OPERATORS[name]
    assert ExpressionPrinter('lepton', shared=False).prepare().run([expression]).code_ == ([], [code])


def test_repeated_subexpressions_are_computed_once():
    """Test that a subexpression shared by two expressions is bound once and reused."""
    assignments, codes = ExpressionPrinter().prepare().run([(X + Y) ** 2, 3 * (X + Y) ** 2]).code_
    assert assignments == [('_cse0', '(x + y)**2')]
    assert codes == ['_cse0', '3*_cse0']


def _printed_law(order: str) -> str:
    """Return the code of a law in every dialect, its symbols created in a forward or backward order."""
    indices = range(12) if order == 'forward' else reversed(range(12))
    reads = {index: symbol(f'a{index}') for index in indices}
    count, rate = symbol('count'), symbol('rate')
    blocked = 1
    for index in range(12):
        blocked *= 1 - where(count > index, reads[index], 0)
    law = rate * blocked * (reads[0] + reads[5] + 2 * reads[11]) + blocked**2 / (rate + reads[3])
    dialects = ('numpy', 'scalar', 'jax')
    return '\n'.join(str(ExpressionPrinter(dialect).prepare().run([law, 3 * law]).code_) for dialect in dialects)


def test_printed_code_is_the_same_in_every_process(monkeypatch):
    """Test that two processes print a law alike, whatever their hash seed and the order they create its symbols in.

    The Numba backend caches compiled kernels by their source, so a source that changes between processes would be
    compiled again in every run.
    """
    printed = []
    for seed, order in (('1', 'forward'), ('2', 'backward')):
        monkeypatch.setenv('PYTHONHASHSEED', seed)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
            printed.append(executor.submit(_printed_law, order).result())
    assert printed[0] == printed[1]
