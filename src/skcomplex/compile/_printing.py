"""Print Sage expressions as code of a target language, with common subexpressions computed once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.special
from sage.all__sagemath_symbolics import SR
from sage.rings.integer import Integer
from sage.rings.rational import Rational

from ..core import Expression, ValidationError

ATOM, POWER, NEGATION, PRODUCT, SUM = 5, 4, 3, 2, 1
CALL = re.compile(r'[\w.]+\(\{0\}(, \{\d\})*(, [\w.]+)?\)')
RELATIONS = {
    operator.lt: 'lt',
    operator.le: 'le',
    operator.gt: 'gt',
    operator.ge: 'ge',
    operator.eq: 'eq',
    operator.ne: 'ne',
}
PYTHON_RELATIONS = {
    'lt': '({0} < {1})',
    'le': '({0} <= {1})',
    'gt': '({0} > {1})',
    'ge': '({0} >= {1})',
    'eq': '({0} == {1})',
    'ne': '({0} != {1})',
}

type Printed = tuple[str, int]


class _Key:
    """Key of an expression in tables, equal to keys of structurally identical expressions.

    Sage's `==` builds a relation whose truth Maxima may be asked to prove, so tables of expressions compare keys
    structurally instead.
    """

    __slots__ = ('_hash', 'item')

    def __init__(self, item: Expression) -> None:
        self.item = item
        self._hash = hash(item)

    def __hash__(self) -> int:
        """Return the hash of the expression."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Check whether two keys hold structurally identical expressions."""
        return isinstance(other, _Key) and bool(self.item.is_trivially_equal(other.item))


def _is_sum(item: Expression) -> bool:
    """Check whether an expression is a sum."""
    return getattr(item.operator(), '__name__', '') == 'add_vararg'


def _is_product(item: Expression) -> bool:
    """Check whether an expression is a product."""
    return getattr(item.operator(), '__name__', '') == 'mul_vararg'


def _is_reciprocal(item: Expression) -> bool:
    """Check whether an expression is a power with a negative number as exponent, printed as a denominator."""
    if item.operator() is not operator.pow:
        return False
    exponent = item.operands()[1]
    return bool(exponent.is_numeric() and exponent.pyobject() < 0)


def _numeric(item: Expression) -> Any:  # noqa: ANN401
    """Return the Python or Sage number of a numeric expression."""
    return item.pyobject()


@dataclass(frozen=True)
class Dialect:
    """Syntax of a target language for printed expressions.

    Templates take their arguments as `{0}`, `{1}`, and so on, receive every argument that is not an atom in
    parentheses, and must give an atom, a name, a call, or a parenthesized expression.

    Args:
        name:
            The name of the language, used in error messages.

        functions:
            A mapping from Sage function names to templates.

        relations:
            A mapping from `lt`, `le`, `gt`, `ge`, `eq`, and `ne` to templates.

        power:
            The power operator.

        error:
            The error raised for an expression the language cannot express.

        exact_rationals:
            Whether rationals print as divisions of integers instead of floating point literals.

    Examples:
        >>> from skcomplex.compile import Dialect, print_expressions
        >>> from skcomplex.core import symbol
        >>> x = symbol('x')
        >>> dialect = Dialect('plain', {'exp': 'exp({0})'}, {})
        >>> print_expressions([2 * x**2], dialect)
        ([], ['2*x**2'])
    """

    name: str
    functions: Mapping[str, str]
    relations: Mapping[str, str]
    power: str = '**'
    error: type[Exception] = ValidationError
    exact_rationals: bool = False
    folds: frozenset[str] = field(default_factory=lambda: frozenset({'min', 'max', 'either', 'both'}))

    def unknown(self, name: str) -> Exception:
        """Return the error of an expression the language cannot express.

        Args:
            name:
                The name of the function or expression.

        Returns:
            error:
                The error.
        """
        error_msg = f'The function {name} has no {self.name} equivalent. Write the law with the functions it supports.'
        return self.error(error_msg)


NUMPY_DIALECT = Dialect(
    'NumPy',
    {
        'exp': 'np.exp({0})',
        'log': 'np.log({0})',
        'sqrt': 'np.sqrt({0})',
        'sin': 'np.sin({0})',
        'cos': 'np.cos({0})',
        'tan': 'np.tan({0})',
        'arcsin': 'np.arcsin({0})',
        'arccos': 'np.arccos({0})',
        'arctan': 'np.arctan({0})',
        'arctan2': 'np.arctan2({0}, {1})',
        'sinh': 'np.sinh({0})',
        'cosh': 'np.cosh({0})',
        'tanh': 'np.tanh({0})',
        'arcsinh': 'np.arcsinh({0})',
        'arccosh': 'np.arccosh({0})',
        'arctanh': 'np.arctanh({0})',
        'abs': 'np.abs({0})',
        'floor': 'np.floor({0})',
        'ceil': 'np.ceil({0})',
        'sgn': 'np.sign({0})',
        'heaviside': 'np.heaviside({0}, 0.5)',
        'erf': 'erf({0})',
        'erfc': 'erfc({0})',
        'gamma': 'gamma({0})',
        'min': 'np.minimum({0}, {1})',
        'max': 'np.maximum({0}, {1})',
        'where': 'np.where({0}, {1}, {2})',
        'either': 'np.logical_or({0}, {1})',
        'both': 'np.logical_and({0}, {1})',
        'mod': 'np.mod({0}, {1})',
    },
    PYTHON_RELATIONS,
)
SCALAR_DIALECT = Dialect(
    'Numba',
    {
        'exp': 'math.exp({0})',
        'log': 'math.log({0})',
        'sqrt': 'math.sqrt({0})',
        'sin': 'math.sin({0})',
        'cos': 'math.cos({0})',
        'tan': 'math.tan({0})',
        'arcsin': 'math.asin({0})',
        'arccos': 'math.acos({0})',
        'arctan': 'math.atan({0})',
        'arctan2': 'math.atan2({0}, {1})',
        'sinh': 'math.sinh({0})',
        'cosh': 'math.cosh({0})',
        'tanh': 'math.tanh({0})',
        'arcsinh': 'math.asinh({0})',
        'arccosh': 'math.acosh({0})',
        'arctanh': 'math.atanh({0})',
        'abs': 'abs({0})',
        'floor': 'math.floor({0})',
        'ceil': 'math.ceil({0})',
        'sgn': 'np.sign({0})',
        'heaviside': 'np.heaviside({0}, 0.5)',
        'erf': 'math.erf({0})',
        'erfc': 'math.erfc({0})',
        'gamma': 'math.gamma({0})',
        'min': 'min({0}, {1})',
        'max': 'max({0}, {1})',
        'where': '({1} if {0} else {2})',
        'either': '({0} or {1})',
        'both': '({0} and {1})',
        'mod': '({0} % {1})',
    },
    PYTHON_RELATIONS,
)
NAMESPACE = {'np': np, 'erf': scipy.special.erf, 'erfc': scipy.special.erfc, 'gamma': scipy.special.gamma}


@dataclass
class _Printer:
    """Printer of expressions in one dialect, with names for symbols and bound subexpressions."""

    dialect: Dialect
    names: Callable[[Expression], str]
    bound: dict[_Key, str] = field(default_factory=dict)

    def wrap(self, printed: Printed, precedence: int) -> str:
        """Return printed code, in parentheses when it binds less tightly than a precedence."""
        code, level = printed
        return code if level >= precedence else f'({code})'

    def number(self, value: Any) -> Printed:  # noqa: ANN401
        """Return the code of a number."""
        if isinstance(value, Integer | int):
            return str(int(value)), (ATOM if value >= 0 else NEGATION)
        if isinstance(value, Rational) and self.dialect.exact_rationals:
            code = f'{abs(int(value.numerator()))}/{int(value.denominator())}'
            return (f'-{code}', NEGATION) if value < 0 else (code, PRODUCT)
        try:
            number = float(value)
        except TypeError as error:
            name = f'complex number {value}'
            raise self.dialect.unknown(name) from error
        return repr(number), (ATOM if number >= 0 else NEGATION)

    def template(self, name: str, arguments: Sequence[Expression]) -> Printed:
        """Return the code of a function or relation from its template."""
        templates = {**self.dialect.functions, **self.dialect.relations}
        if name not in templates:
            raise self.dialect.unknown(name)
        call = CALL.fullmatch(templates[name]) is not None
        codes = [self.print(argument)[0] if call else self.wrap(self.print(argument), ATOM) for argument in arguments]
        if name in self.dialect.folds:
            code = codes[0]
            for other in codes[1:]:
                code = templates[name].format(code, other)
            return code, ATOM
        return templates[name].format(*codes), ATOM

    def product(self, item: Expression) -> Printed:
        """Return the code of a product, with factors of negative powers in its denominator."""
        coefficient: Any = 1
        numerator: list[str] = []
        denominator: list[str] = []
        for factor in item.operands() if _is_product(item) else [item]:
            if factor.is_numeric():
                coefficient *= _numeric(factor)
                continue
            exponent = factor.operands()[1] if factor.operator() is operator.pow else None
            if exponent is not None and exponent.is_numeric() and _numeric(exponent) < 0:
                base = factor.operands()[0]
                denominator.append(self.wrap(self.power(base, -exponent), PRODUCT + 1))
                continue
            numerator.append(self.wrap(self.print(factor), PRODUCT))
        negative = coefficient < 0
        coefficient = -coefficient if negative else coefficient
        if isinstance(coefficient, Rational) and not isinstance(coefficient, Integer):
            if coefficient.numerator() != 1:
                numerator.insert(0, str(int(coefficient.numerator())))
            denominator.insert(0, str(int(coefficient.denominator())))
        elif coefficient != 1:
            numerator.insert(0, self.number(coefficient)[0])
        code = '*'.join(numerator) or '1'
        if denominator:
            divisor = denominator[0] if len(denominator) == 1 else f'({"*".join(denominator)})'
            code = f'{code}/{divisor}'
        if negative:
            return f'-{code}', NEGATION
        return code, PRODUCT

    def power(self, base: Expression, exponent: Expression) -> Printed:
        """Return the code of a power, with square roots for exponents of a half."""
        if exponent.is_numeric() and _numeric(exponent) == Rational((1, 2)):
            return self.template('sqrt', [base])
        if exponent.is_numeric() and _numeric(exponent) == Rational((-1, 2)):
            return f'1/{self.template("sqrt", [base])[0]}', PRODUCT
        if exponent.is_numeric() and _numeric(exponent) == 1:
            return self.print(base)
        code = f'{self.wrap(self.print(base), ATOM)}{self.dialect.power}{self.wrap(self.print(exponent), ATOM)}'
        return code, POWER

    def total(self, item: Expression) -> Printed:
        """Return the code of a sum."""
        terms = [self.print(term) for term in item.operands()]
        code = self.wrap(terms[0], SUM)
        for term, level in terms[1:]:
            if level == NEGATION and term.startswith('-'):
                code += f' - {term[1:]}'
            else:
                code += f' + {self.wrap((term, level), SUM + 1)}'
        return code, SUM

    def leaf(self, item: Expression) -> Printed:
        """Return the code of a bound subexpression, a symbol, a number, or a constant."""
        if _Key(item) in self.bound:
            return self.bound[_Key(item)], ATOM
        if item.is_symbol():
            return self.names(item), ATOM
        return self.number(_numeric(item) if item.is_numeric() else float(item))

    def print(self, item: Expression) -> Printed:
        """Return the code of an expression and how tightly it binds."""
        if _Key(item) in self.bound or item.is_symbol() or item.is_numeric() or item.is_constant():
            return self.leaf(item)
        function = item.operator()
        if _is_sum(item):
            return self.total(item)
        if _is_product(item):
            return self.product(item)
        if function is operator.pow:
            return self.product(item) if _is_reciprocal(item) else self.power(*item.operands())
        if function in RELATIONS:
            return self.template(RELATIONS[function], item.operands())
        name = function.name() if hasattr(function, 'name') else str(function)
        return self.template(name, item.operands())


def _subexpressions(expressions: Sequence[Expression]) -> Counter[_Key]:
    """Return how often every compound subexpression occurs in expressions."""
    counts: Counter[_Key] = Counter()
    stack = list(expressions)
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if _is_reciprocal(item):
            stack.append(item.operands()[0])
            continue
        key = _Key(item)
        counts[key] += 1
        if counts[key] == 1:
            stack.extend(item.operands())
    return counts


def print_expressions(
    expressions: Sequence[Expression],
    dialect: Dialect,
    names: Callable[[Expression], str] = str,
    prefix: str = '_cse',
    *,
    shared: bool = True,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return the code of expressions, with every repeated compound subexpression bound to a numbered name.

    Args:
        expressions:
            The expressions.

        dialect:
            The target language.

        names:
            The function that gives the code name of a symbol.

        prefix:
            The prefix of the names of bound subexpressions.

        shared:
            Whether repeated subexpressions are bound to names, for languages without assignments.

    Returns:
        code:
            The assignments of bound subexpressions in the order they must run, and the code of every expression.

    Raises:
        ValidationError:
            An expression uses a function the dialect has no equivalent for, with the dialect's error type.

    Examples:
        >>> from skcomplex.compile import NUMPY_DIALECT, print_expressions
        >>> from skcomplex.core import symbol
        >>> x = symbol('x')
        >>> print_expressions([(x + 1) ** 2, 3 * (x + 1) ** 2], NUMPY_DIALECT)
        ([('_cse0', '(x + 1)**2')], ['_cse0', '3*_cse0'])
    """
    expressions = [SR(expression) for expression in expressions]
    printer = _Printer(dialect, names)
    counts = _subexpressions(expressions)
    assignments: list[tuple[str, str]] = []

    def bind(item: Expression) -> None:
        """Bind the repeated subexpressions of an expression, innermost first."""
        key = _Key(item)
        if item.operator() is None or key in printer.bound:
            return
        for operand in item.operands():
            bind(operand)
        if counts[key] > 1:
            name = f'{prefix}{len(assignments)}'
            assignments.append((name, printer.print(item)[0]))
            printer.bound[key] = name

    for expression in expressions if shared else ():
        bind(expression)
    return assignments, [printer.print(expression)[0] for expression in expressions]


def numpy_function(
    time: Expression,
    arguments: Sequence[Expression],
    expressions: Sequence[Expression],
) -> Callable[..., list[Any]]:
    """Return a NumPy function of time and arguments that evaluates expressions over arrays.

    Args:
        time:
            The time symbol, the first argument of the function.

        arguments:
            The symbols passed after time, in order.

        expressions:
            The expressions the function returns as a list.

    Returns:
        function:
            The function `f(t, *arguments)`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.compile import numpy_function
        >>> from skcomplex.core import symbol
        >>> t, x = symbol('t'), symbol('x')
        >>> numpy_function(t, [x], [t * x, x**2])(2.0, np.array([1.0, 3.0]))
        [array([2., 6.]), array([1., 9.])]
    """
    names = {item: f'_a{index}' for index, item in enumerate(arguments)} | {time: 't'}
    assignments, codes = print_expressions(expressions, NUMPY_DIALECT, lambda item: names[item])
    lines = [
        f'def _function(t, {", ".join(names[item] for item in arguments)}):',
        *(f'    {name} = {code}' for name, code in assignments),
        f'    return [{", ".join(codes)}]',
    ]
    namespace = dict(NAMESPACE)
    exec('\n'.join(lines), namespace)  # noqa: S102  # nosec B102: runs code printed from the model's own expressions
    return namespace['_function']
