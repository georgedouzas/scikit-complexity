"""Print Sage expressions as code of a target language, with common subexpressions computed once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from sage.rings.integer import Integer
from sage.rings.rational import Rational
from sage.structure.element import RingElement

from ...errors import ValidationError
from ...specs._dialect_spec import DIALECTS
from ..types import Number, PrintedKey
from ._base import BasePrinter
from ._printed_spec import PrintedSpec
from ._source_loader import SourceLoader

ATOM, POWER, NEGATION, PRODUCT, SUM = 5, 4, 3, 2, 1
_CALL = re.compile(r'[\w.]+\(\{0\}(, \{\d\})*(, [\w.]+)?\)')
RELATIONS = {
    operator.lt: 'lt',
    operator.le: 'le',
    operator.gt: 'gt',
    operator.ge: 'ge',
    operator.eq: 'eq',
    operator.ne: 'ne',
}

_CONDITION = 'np.where('
_QUIET_ERRORS = "with np.errstate(divide='ignore', invalid='ignore', over='ignore'):"


IMPORTS = {
    'numpy': 'import numpy as np\nfrom scipy.special import erf, erfc, gamma\n\n\n',
    'jax': 'import jax.numpy as jnp\nimport jax.scipy.special as jsp\n\n\n',
}


class ExpressionPrinter(BasePrinter[Sequence[Expression], Sequence[Expression]]):
    """Printer of Sage expressions as code of NumPy, scalar Python for Numba, or LAMMPS Lepton.

    `prepare(symbols)` takes the symbols the code reads, in order, and `run(expressions)` keeps in `code_` the
    assignments of the repeated compound subexpressions, bound to numbered names, and the code of every expression. A
    compiled printer keeps instead the NumPy or JAX function `f(t, *symbols)` of the expressions. A NumPy function runs
    in a NumPy error state that ignores division, invalid, and overflow warnings when the expressions hold conditions.

    Lepton has neither comparisons nor a remainder, so conditions become `step` and `select` and remainders `floor`.

    Args:
        dialect:
            The target language, `'numpy'`, `'scalar'` for the scalar Python of compiled Numba loops, `'lepton'`, or
            `'jax'` for JAX's NumPy, whose functions return new arrays.

        time:
            The time symbol, the first argument of a compiled function.

        names:
            The function that gives the code name of a symbol, or `None` for its printed name.

        prefix:
            The prefix of the names of bound subexpressions.

        shared:
            Whether repeated subexpressions are bound to names, false for languages without assignments.

        compiled:
            Whether `run` keeps a NumPy or JAX function of time and the symbols instead of code.

    Attributes:
        code_:
            The assignments of the repeated subexpressions and the code of every expression of the last run, or the
            NumPy function `f(t, *symbols)` of the expressions for a compiled printer.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems import ExpressionPrinter
        >>> from skcomplex.core.engine._utils import symbol
        >>> t, x = symbol('t'), symbol('x')
        >>> ExpressionPrinter().prepare().run([(x + 1) ** 2, 3 * (x + 1) ** 2]).code_
        ([('_cse0', '(x + 1)**2')], ['_cse0', '3*_cse0'])
        >>> printer = ExpressionPrinter(time=t, compiled=True).prepare([x])
        >>> printer.run([t * x, x**2]).code_(2.0, np.array([1.0, 3.0]))
        [array([2., 6.]), array([1., 9.])]
    """

    code_: tuple[list[tuple[str, str]], list[str]] | Callable[..., list[npt.ArrayLike]]

    def __init__(
        self,
        dialect: str = 'numpy',
        *,
        time: Expression | None = None,
        names: Callable[[Expression], str] | None = None,
        prefix: str = '_cse',
        shared: bool = True,
        compiled: bool = False,
    ) -> None:
        self.dialect = dialect
        self.time = time
        self.names = names
        self.prefix = prefix
        self.shared = shared
        self.compiled = compiled

    def _check(self) -> None:
        """Raise an error for an unknown dialect, or a compiled printer of a dialect without functions or of no time.

        Raises:
            ValidationError:
                The dialect is unknown, or a compiled printer is not a NumPy or JAX printer with a time symbol.
        """
        if self.dialect not in DIALECTS:
            error_msg = f'The dialect {self.dialect!r} is unknown. Use one of {", ".join(DIALECTS)}.'
            raise ValidationError(error_msg)
        if self.compiled and (self.dialect not in IMPORTS or self.time is None):
            error_msg = 'A compiled printer prints NumPy or JAX code and needs the time symbol, `time=...`.'
            raise ValidationError(error_msg)

    def _prepare(self, context: Sequence[Expression] | None) -> None:
        """Set the dialect, the symbols the code reads, and the code name of every symbol."""
        self.dialect_ = DIALECTS[self.dialect]
        self.symbols_ = list(context or ())
        if self.compiled:
            mapping = {item: f'_a{index}' for index, item in enumerate(self.symbols_)} | {self.time: 't'}
            self.names_: Callable[[Expression], str] = lambda item: mapping[item]
        else:
            self.names_ = self.names or str

    def _run(self, inputs: Sequence[Expression] | None) -> None:
        """Set the assignments and the code of the expressions, or the NumPy function of a compiled printer.

        Args:
            inputs:
                The expressions.

        Raises:
            ValidationError:
                No expressions are passed.
        """
        if inputs is None:
            error_msg = 'A printer runs on expressions. Pass them to `run`.'
            raise ValidationError(error_msg)
        expressions = [SR(expression) for expression in inputs]
        self.bound_: dict[PrintedKey, str] = {}
        counts = self._subexpressions(expressions)
        assignments: list[tuple[str, str]] = []

        def bind(item: Expression) -> None:
            """Bind the repeated subexpressions of an expression, innermost first."""
            if item.operator() is None:
                return
            key = self._key(item)
            if key in self.bound_:
                return
            for operand in item.operands():
                bind(operand)
            if counts[key] > 1:
                name = f'{self.prefix}{len(assignments)}'
                assignments.append((name, self._print(item).code))
                self.bound_[key] = name

        for expression in expressions if self.shared else ():
            bind(expression)
        codes = [self._print(expression).code for expression in expressions]
        self.code_ = self._function(assignments, codes) if self.compiled else (assignments, codes)

    @staticmethod
    def _function_of(
        printed: tuple[list[tuple[str, str]], list[str]] | Callable[..., list[npt.ArrayLike]],
    ) -> Callable[..., list[npt.ArrayLike]]:
        """Return the NumPy function a compiled printer kept in `code_`.

        Args:
            printed:
                The `code_` of a compiled printer.

        Returns:
            function:
                The function `f(t, *symbols)`.

        Raises:
            TypeError:
                The printer was not compiled, so it kept code instead of a function.
        """
        if isinstance(printed, tuple):
            error_msg = f'A compiled printer returns a function, not a {type(printed).__name__}.'
            raise TypeError(error_msg)
        return printed

    @staticmethod
    def _values_of(computed: object) -> list[npt.ArrayLike]:
        """Return the values of the expressions that the function of a compiled printer computed.

        Args:
            computed:
                What the function returned.

        Returns:
            values:
                One array or number per expression.

        Raises:
            TypeError:
                The function did not return a list.
        """
        if not isinstance(computed, list):
            error_msg = f'The function of a compiled printer returns a list, not a {type(computed).__name__}.'
            raise TypeError(error_msg)
        return computed

    def _function(self, assignments: list[tuple[str, str]], codes: list[str]) -> Callable[..., list[npt.ArrayLike]]:
        """Return the NumPy or JAX function `f(t, *symbols)` that runs printed assignments and returns the codes."""
        body = [*(f'{name} = {code}' for name, code in assignments), f'return [{", ".join(codes)}]']
        conditions = any(_CONDITION in code for _, code in assignments) or any(_CONDITION in code for code in codes)
        if self.dialect == 'numpy' and conditions:
            body = [_QUIET_ERRORS, *(f'    {line}' for line in body)]
        arguments = ', '.join(self.names_(item) for item in self.symbols_)
        lines = [f'def _function(t, {arguments}):', *(f'    {line}' for line in body)]
        loader = SourceLoader('skcomplex_functions').prepare()
        source = IMPORTS[self.dialect] + '\n'.join(lines) + '\n'
        function: Callable[..., list[npt.ArrayLike]] = loader.run(source).module_._function
        return function

    def _wrap(self, printed: PrintedSpec, precedence: int) -> str:
        """Return printed code, in parentheses when it binds less tightly than a precedence."""
        return printed.code if printed.precedence >= precedence else f'({printed.code})'

    def _number(self, value: Number) -> PrintedSpec:
        """Return the code of a number."""
        if isinstance(value, Integer | int):
            return PrintedSpec(str(int(value)), ATOM if value >= 0 else NEGATION)
        if isinstance(value, Rational) and self.dialect_.exact_rationals:
            code = f'{abs(int(value.numerator()))}/{int(value.denominator())}'
            return PrintedSpec(f'-{code}', NEGATION) if value < 0 else PrintedSpec(code, PRODUCT)
        number = None if isinstance(value, complex) else self._real(value)
        if number is None:
            if not self.dialect_.complex_numbers:
                name = f'complex number {value}'
                raise self.dialect_.unknown(name)
            return PrintedSpec(repr(complex(value)), ATOM)
        return PrintedSpec(repr(number), ATOM if number >= 0 else NEGATION)

    def _template(self, name: str, arguments: Sequence[Expression]) -> PrintedSpec:
        """Return the code of a function or relation from its template."""
        templates = {**self.dialect_.functions, **self.dialect_.relations}
        if name not in templates:
            raise self.dialect_.unknown(name)
        call = _CALL.fullmatch(templates[name]) is not None
        codes = [
            self._print(argument).code if call else self._wrap(self._print(argument), ATOM) for argument in arguments
        ]
        if name in self.dialect_.folds:
            code = codes[0]
            for other in codes[1:]:
                code = templates[name].format(code, other)
            return PrintedSpec(code, ATOM)
        return PrintedSpec(templates[name].format(*codes), ATOM)

    def _product(self, item: Expression) -> PrintedSpec:
        """Return the code of a product, with factors of negative powers in its denominator."""
        coefficient: Number = 1
        numerator: list[str] = []
        denominator: list[str] = []
        for factor in item.operands() if self._is_product(item) else [item]:
            if factor.is_numeric():
                coefficient *= self._numeric(factor)
                continue
            exponent = factor.operands()[1] if factor.operator() is operator.pow else None
            if exponent is not None and exponent.is_numeric() and self._is_negative(self._numeric(exponent)):
                base = factor.operands()[0]
                denominator.append(self._wrap(self._power(base, -exponent), PRODUCT + 1))
                continue
            numerator.append(self._wrap(self._print(factor), PRODUCT))
        negative = self._is_negative(coefficient)
        coefficient = -coefficient if negative else coefficient
        if isinstance(coefficient, Rational) and not isinstance(coefficient, Integer):
            if coefficient.numerator() != 1:
                numerator.insert(0, str(int(coefficient.numerator())))
            denominator.insert(0, str(int(coefficient.denominator())))
        elif coefficient != 1:
            numerator.insert(0, self._number(coefficient).code)
        code = '*'.join(numerator) or '1'
        if denominator:
            divisor = denominator[0] if len(denominator) == 1 else f'({"*".join(denominator)})'
            code = f'{code}/{divisor}'
        if negative:
            return PrintedSpec(f'-{code}', NEGATION)
        return PrintedSpec(code, PRODUCT)

    def _power(self, base: Expression, exponent: Expression) -> PrintedSpec:
        """Return the code of a power, with square roots for exponents of a half."""
        if exponent.is_numeric() and self._numeric(exponent) == Rational((1, 2)):
            return self._template('sqrt', [base])
        if exponent.is_numeric() and self._numeric(exponent) == Rational((-1, 2)):
            return PrintedSpec(f'1/{self._template("sqrt", [base]).code}', PRODUCT)
        if exponent.is_numeric() and self._numeric(exponent) == 1:
            return self._print(base)
        code = f'{self._wrap(self._print(base), ATOM)}{self.dialect_.power}{self._wrap(self._print(exponent), ATOM)}'
        return PrintedSpec(code, POWER)

    def _total(self, item: Expression) -> PrintedSpec:
        """Return the code of a sum."""
        terms = [self._print(term) for term in item.operands()]
        code = self._wrap(terms[0], SUM)
        for term in terms[1:]:
            if term.precedence == NEGATION and term.code.startswith('-'):
                code += f' - {term.code[1:]}'
            else:
                code += f' + {self._wrap(term, SUM + 1)}'
        return PrintedSpec(code, SUM)

    def _leaf(self, item: Expression) -> PrintedSpec:
        """Return the code of a bound subexpression, a symbol, a number, or a constant."""
        name = self.bound_.get(self._key(item)) if self.bound_ else None
        if name is not None:
            return PrintedSpec(name, ATOM)
        if item.is_symbol():
            return PrintedSpec(self.names_(item), ATOM)
        return self._number(self._numeric(item) if item.is_numeric() else float(item))

    def _print(self, item: Expression) -> PrintedSpec:
        """Return the code of an expression and how tightly it binds."""
        if (
            item.is_symbol()
            or item.is_numeric()
            or item.is_constant()
            or (self.bound_ and self._key(item) in self.bound_)
        ):
            return self._leaf(item)
        function = item.operator()
        if self._is_sum(item):
            return self._total(item)
        if self._is_product(item):
            return self._product(item)
        if function is operator.pow:
            return self._product(item) if self._is_reciprocal(item) else self._power(*item.operands())
        if function in RELATIONS:
            return self._template(RELATIONS[function], item.operands())
        name = function.name() if hasattr(function, 'name') else str(function)
        return self._template(name, item.operands())

    @staticmethod
    def _is_sum(item: Expression) -> bool:
        """Check whether an expression is a sum."""
        return getattr(item.operator(), '__name__', '') == 'add_vararg'

    @staticmethod
    def _is_product(item: Expression) -> bool:
        """Check whether an expression is a product."""
        return getattr(item.operator(), '__name__', '') == 'mul_vararg'

    @staticmethod
    def _is_negative(value: Number) -> bool:
        """Check whether a number is real and negative, so complex numbers are never negated."""
        number = None if isinstance(value, complex) else ExpressionPrinter._real(value)
        return number is not None and number < 0

    @staticmethod
    def _real(value: RingElement | float) -> float | None:
        """Return the float of a real number, or `None` for a complex Sage number."""
        try:
            return float(value)
        except TypeError:
            return None

    @staticmethod
    def _numeric(item: Expression) -> Number:
        """Return the Python or Sage number of a numeric expression."""
        return item.pyobject()

    @staticmethod
    def _is_reciprocal(item: Expression) -> bool:
        """Check whether an expression is a power with a negative number as exponent, printed as a denominator."""
        if item.operator() is not operator.pow:
            return False
        exponent = item.operands()[1]
        return bool(exponent.is_numeric() and exponent.pyobject() < 0)

    @staticmethod
    def _key(item: Expression) -> PrintedKey:
        """Return the key of an expression in tables, equal to the keys of structurally identical expressions.

        Sage's `==` builds a relation whose truth Maxima may be asked to prove, so tables of expressions are keyed by
        the structural hash and the printed form of an expression instead.
        """
        return hash(item), str(item)

    @staticmethod
    def _subexpressions(expressions: Sequence[Expression]) -> Counter[PrintedKey]:
        """Return how often every compound subexpression occurs in expressions."""
        counts: Counter[PrintedKey] = Counter()
        stack = list(expressions)
        while stack:
            item = stack.pop()
            if item.operator() is None:
                continue
            if ExpressionPrinter._is_reciprocal(item):
                stack.append(item.operands()[0])
                continue
            key = ExpressionPrinter._key(item)
            counts[key] += 1
            if counts[key] == 1:
                stack.extend(item.operands())
        return counts
