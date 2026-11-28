"""Print Sage expressions as code of a target language."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from sage.rings.integer import Integer
from sage.rings.rational import Rational
from sage.structure.element import RingElement

from ...errors import BackendError, ValidationError
from ..types import Number, PrintedKey
from ._base import BasePrinter
from ._source_loader import SourceLoader

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
CONDITION = 'np.where('
QUIET_ERRORS = "with np.errstate(divide='ignore', invalid='ignore', over='ignore'):"
IMPORTS = {
    'numpy': 'import numpy as np\nfrom scipy.special import erf, erfc, gamma\n\n\n',
    'jax': 'import jax\nimport jax.numpy as jnp\nimport jax.scipy.special as jsp\n\n\n',
}


@dataclass(frozen=True)
class _Dialect:
    """A record of the syntax of a target language for printed expressions.

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

        complex_numbers:
            Whether complex numbers print as Python complex literals instead of raising an error.
    """

    name: str
    functions: Mapping[str, str]
    relations: Mapping[str, str]
    power: str = '**'
    error: type[Exception] = ValidationError
    exact_rationals: bool = False
    complex_numbers: bool = False
    folds: frozenset[str] = field(default_factory=lambda: frozenset({'min', 'max', 'either', 'both'}))

    def get_unknown(self, name: str) -> Exception:
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


PYTHON_RELATIONS = {
    'lt': '({0} < {1})',
    'le': '({0} <= {1})',
    'gt': '({0} > {1})',
    'ge': '({0} >= {1})',
    'eq': '({0} == {1})',
    'ne': '({0} != {1})',
}

NUMPY_DIALECT = _Dialect(
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
        'stop_gradient': '({0})',
    },
    PYTHON_RELATIONS,
    complex_numbers=True,
)

SCALAR_DIALECT = _Dialect(
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
        'stop_gradient': '({0})',
    },
    PYTHON_RELATIONS,
)

LEPTON_DIALECT = _Dialect(
    'Lepton',
    {
        'exp': 'exp({0})',
        'log': 'log({0})',
        'sqrt': 'sqrt({0})',
        'sin': 'sin({0})',
        'cos': 'cos({0})',
        'tan': 'tan({0})',
        'arcsin': 'asin({0})',
        'arccos': 'acos({0})',
        'arctan': 'atan({0})',
        'sinh': 'sinh({0})',
        'cosh': 'cosh({0})',
        'tanh': 'tanh({0})',
        'erf': 'erf({0})',
        'erfc': 'erfc({0})',
        'abs': 'abs({0})',
        'floor': 'floor({0})',
        'ceil': 'ceil({0})',
        'min': 'min({0}, {1})',
        'max': 'max({0}, {1})',
        'where': 'select({0}, {1}, {2})',
        'either': 'max({0}, {1})',
        'both': '({0}*{1})',
        'mod': '({0} - {1}*floor({0}/{1}))',
        'stop_gradient': '({0})',
    },
    {
        'lt': '(1 - step({0} - {1}))',
        'le': 'step({1} - {0})',
        'gt': '(1 - step({1} - {0}))',
        'ge': 'step({0} - {1})',
        'eq': 'delta({0} - {1})',
        'ne': '(1 - delta({0} - {1}))',
    },
    power='^',
    error=BackendError,
    exact_rationals=True,
)

JAX_DIALECT = _Dialect(
    'JAX',
    {
        **{name: template.replace('np.', 'jnp.') for name, template in NUMPY_DIALECT.functions.items()},
        'erf': 'jsp.erf({0})',
        'erfc': 'jsp.erfc({0})',
        'gamma': 'jsp.gamma({0})',
        'stop_gradient': 'jax.lax.stop_gradient({0})',
    },
    PYTHON_RELATIONS,
    error=BackendError,
)

DIALECTS = {'numpy': NUMPY_DIALECT, 'scalar': SCALAR_DIALECT, 'lepton': LEPTON_DIALECT, 'jax': JAX_DIALECT}


class ExpressionPrinter(BasePrinter[Sequence[Expression], Sequence[Expression]]):
    """A printer of Sage expressions as code of NumPy, Numba, or LAMMPS.

    `prepare(symbols)` takes the symbols the code reads, in order, and `run(expressions)` keeps in `code_` the
    subexpressions that repeat, bound to numbered names, and the code of every expression, such as `'3*_cse0'` for
    `3 * (x + 1) ** 2`. It orders the terms of sums and products the same way in every process, so the Numba backend
    reuses the code it cached. A compiled printer keeps instead the NumPy or JAX function `f(t, *symbols)` of the
    expressions. Lepton has neither comparisons nor remainders, so conditions become `step` and `select` and remainders
    `floor`. It prints the laws of a reaction, of a gas, and of a pair energy for LAMMPS.

    Read more in the user guide, under Extend the library.

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

    See Also:
        BasePrinter: The base of every printer.

    Examples:
        >>> from skcomplex.core.engine._utils import symbol
        >>> from skcomplex.core.engine.problems import ExpressionPrinter
        >>> x = symbol('x')
        >>> ExpressionPrinter().prepare().run([(x + 1) ** 2, 3 * (x + 1) ** 2]).code_
        ([('_cse0', '(x + 1)**2')], ['_cse0', '3*_cse0'])
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

    def _prepare(self, context: Sequence[Expression] | None) -> None:
        """Set the dialect, the symbols the code reads, and the code name of every symbol.

        It first raises an error for an unknown dialect or a compiled printer it cannot make.

        Args:
            context:
                The symbols the code reads, or `None`.

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
        self.dialect_ = DIALECTS[self.dialect]
        self.symbols_ = list(context or ())
        if self.compiled:
            mapping = {item: f'_a{index}' for index, item in enumerate(self.symbols_)} | {self.time: 't'}
            self.names_: Callable[[Expression], str] = lambda item: mapping[item]
        else:
            self.names_ = self.names or str

    def _run(self, inputs: Sequence[Expression] | None) -> None:
        """Set the assignments and the code of the expressions.

        A compiled printer sets the NumPy function instead.

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
        self.canonical_: dict[PrintedKey, str] = {}
        counts = self._get_subexpressions(expressions)
        assignments: list[tuple[str, str]] = []

        def bind(item: Expression) -> None:
            """Bind the repeated subexpressions of an expression, innermost first."""
            if item.operator() is None:
                return
            key = self._get_key(item)
            if key in self.bound_:
                return
            for operand in self._get_operands(item):
                bind(operand)
            if counts[key] > 1:
                name = f'{self.prefix}{len(assignments)}'
                assignments.append((name, self._print(item)[0]))
                self.bound_[key] = name

        for expression in expressions if self.shared else ():
            bind(expression)
        codes = [self._print(expression)[0] for expression in expressions]
        self.code_ = self._get_function(assignments, codes) if self.compiled else (assignments, codes)

    @staticmethod
    def _get_function_of(
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
    def _get_values_of(computed: object) -> list[npt.ArrayLike]:
        """Return the values of the expressions that a compiled printer computed.

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

    def _get_function(self, assignments: list[tuple[str, str]], codes: list[str]) -> Callable[..., list[npt.ArrayLike]]:
        """Return the NumPy or JAX function `f(t, *symbols)` of the printed assignments.

        The function runs the printed assignments and returns the codes.
        """
        body = [*(f'{name} = {code}' for name, code in assignments), f'return [{", ".join(codes)}]']
        conditions = any(CONDITION in code for _, code in assignments) or any(CONDITION in code for code in codes)
        if self.dialect == 'numpy' and conditions:
            body = [QUIET_ERRORS, *(f'    {line}' for line in body)]
        arguments = ', '.join(self.names_(item) for item in self.symbols_)
        lines = [f'def _function(t, {arguments}):', *(f'    {line}' for line in body)]
        loader = SourceLoader('skcomplex_functions').prepare()
        source = IMPORTS[self.dialect] + '\n'.join(lines) + '\n'
        function: Callable[..., list[npt.ArrayLike]] = loader.run(source).module_._function
        return function

    def _wrap(self, printed: tuple[str, int], precedence: int) -> str:
        """Return code, in parentheses when it binds less tightly than a precedence."""
        code, own = printed
        return code if own >= precedence else f'({code})'

    def _get_number(self, value: Number) -> tuple[str, int]:
        """Return the code of a number."""
        if isinstance(value, Integer | int):
            return (str(int(value)), ATOM if value >= 0 else NEGATION)
        if isinstance(value, Rational) and self.dialect_.exact_rationals:
            code = f'{abs(int(value.numerator()))}/{int(value.denominator())}'
            return (f'-{code}', NEGATION) if value < 0 else (code, PRODUCT)
        number = None if isinstance(value, complex) else self._get_real(value)
        if number is None:
            if not self.dialect_.complex_numbers:
                name = f'complex number {value}'
                raise self.dialect_.get_unknown(name)
            return (repr(complex(value)), ATOM)
        return (repr(number), ATOM if number >= 0 else NEGATION)

    def _get_template(self, name: str, arguments: Sequence[Expression]) -> tuple[str, int]:
        """Return the code of a function or relation from its template."""
        templates = {**self.dialect_.functions, **self.dialect_.relations}
        if name not in templates:
            raise self.dialect_.get_unknown(name)
        call = CALL.fullmatch(templates[name]) is not None
        codes = [
            self._print(argument)[0] if call else self._wrap(self._print(argument), ATOM) for argument in arguments
        ]
        if name in self.dialect_.folds:
            code = codes[0]
            for other in codes[1:]:
                code = templates[name].format(code, other)
            return (code, ATOM)
        return (templates[name].format(*codes), ATOM)

    def _get_product(self, item: Expression) -> tuple[str, int]:
        """Return the code of a product.

        Factors of negative powers go in its denominator.
        """
        coefficient: Number = 1
        numerator: list[str] = []
        denominator: list[str] = []
        for factor in self._get_operands(item) if self._is_product(item) else [item]:
            if factor.is_numeric():
                coefficient *= self._get_numeric(factor)
                continue
            exponent = factor.operands()[1] if factor.operator() is operator.pow else None
            if exponent is not None and exponent.is_numeric() and self._is_negative(self._get_numeric(exponent)):
                base = factor.operands()[0]
                denominator.append(self._wrap(self._get_power(base, -exponent), PRODUCT + 1))
                continue
            numerator.append(self._wrap(self._print(factor), PRODUCT))
        negative = self._is_negative(coefficient)
        coefficient = -coefficient if negative else coefficient
        if isinstance(coefficient, Rational) and not isinstance(coefficient, Integer):
            if coefficient.numerator() != 1:
                numerator.insert(0, str(int(coefficient.numerator())))
            denominator.insert(0, str(int(coefficient.denominator())))
        elif coefficient != 1:
            numerator.insert(0, self._get_number(coefficient)[0])
        code = '*'.join(numerator) or '1'
        if denominator:
            divisor = denominator[0] if len(denominator) == 1 else f'({"*".join(denominator)})'
            code = f'{code}/{divisor}'
        if negative:
            return (f'-{code}', NEGATION)
        return (code, PRODUCT)

    def _get_power(self, base: Expression, exponent: Expression) -> tuple[str, int]:
        """Return the code of a power, with square roots for exponents of a half."""
        if exponent.is_numeric() and self._get_numeric(exponent) == Rational((1, 2)):
            return self._get_template('sqrt', [base])
        if exponent.is_numeric() and self._get_numeric(exponent) == Rational((-1, 2)):
            return (f'1/{self._get_template("sqrt", [base])[0]}', PRODUCT)
        if exponent.is_numeric() and self._get_numeric(exponent) == 1:
            return self._print(base)
        code = f'{self._wrap(self._print(base), ATOM)}{self.dialect_.power}{self._wrap(self._print(exponent), ATOM)}'
        return (code, POWER)

    def _get_total(self, item: Expression) -> tuple[str, int]:
        """Return the code of a sum.

        Its terms come in canonical order, with the negated ones last.
        """
        terms = [self._print(term) for term in self._get_operands(item)]
        terms.sort(key=lambda term: term[1] == NEGATION and term[0].startswith('-'))
        code = self._wrap(terms[0], SUM)
        for term in terms[1:]:
            if term[1] == NEGATION and term[0].startswith('-'):
                code += f' - {term[0][1:]}'
            else:
                code += f' + {self._wrap(term, SUM + 1)}'
        return (code, SUM)

    def _get_leaf(self, item: Expression) -> tuple[str, int]:
        """Return the code of a bound subexpression, a symbol, a number, or a constant."""
        name = self.bound_.get(self._get_key(item)) if self.bound_ else None
        if name is not None:
            return (name, ATOM)
        if item.is_symbol():
            return (self.names_(item), ATOM)
        return self._get_number(self._get_numeric(item) if item.is_numeric() else float(item))

    def _print(self, item: Expression) -> tuple[str, int]:
        """Return the code of an expression and how tightly it binds."""
        if (
            item.is_symbol()
            or item.is_numeric()
            or item.is_constant()
            or (self.bound_ and self._get_key(item) in self.bound_)
        ):
            return self._get_leaf(item)
        function = item.operator()
        if self._is_sum(item):
            return self._get_total(item)
        if self._is_product(item):
            return self._get_product(item)
        if function is operator.pow:
            return self._get_product(item) if self._is_reciprocal(item) else self._get_power(*item.operands())
        if function in RELATIONS:
            return self._get_template(RELATIONS[function], item.operands())
        name = function.name() if hasattr(function, 'name') else str(function)
        return self._get_template(name, item.operands())

    def _get_canonical(self, item: Expression) -> str:
        """Return the canonical form of an expression.

        It is the printed form of the expression with the operands of sums and products sorted. Sage prints the operands
        of sums and products in an order that depends on the process, such as on the order the symbols were created in,
        while the canonical form depends only on the expression.
        """
        if item.operator() is None:
            return str(item)
        key = self._get_key(item)
        if key not in self.canonical_:
            function = item.operator()
            name = function.name() if hasattr(function, 'name') else getattr(function, '__name__', str(function))
            operands = [self._get_canonical(operand) for operand in item.operands()]
            if self._is_sum(item) or self._is_product(item):
                operands.sort()
            self.canonical_[key] = f'{name}({", ".join(operands)})'
        return self.canonical_[key]

    def _get_operands(self, item: Expression) -> list[Expression]:
        """Return the operands of an expression.

        Those of a sum or a product come in canonical order, with numbers last.
        """
        operands = list(item.operands())
        if self._is_sum(item) or self._is_product(item):
            operands.sort(key=lambda operand: (bool(operand.is_numeric()), self._get_canonical(operand)))
        return operands

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
        """Check whether a number is real and negative.

        Complex numbers are thus never negated.
        """
        number = None if isinstance(value, complex) else ExpressionPrinter._get_real(value)
        return number is not None and number < 0

    @staticmethod
    def _get_real(value: RingElement | float) -> float | None:
        """Return the float of a real number, or `None` for a complex Sage number."""
        try:
            return float(value)
        except TypeError:
            return None

    @staticmethod
    def _get_numeric(item: Expression) -> Number:
        """Return the Python or Sage number of a numeric expression."""
        return item.pyobject()

    @staticmethod
    def _is_reciprocal(item: Expression) -> bool:
        """Check whether an expression is a power with a negative number as exponent.

        Such a power is printed as a denominator.
        """
        if item.operator() is not operator.pow:
            return False
        exponent = item.operands()[1]
        return bool(exponent.is_numeric() and exponent.pyobject() < 0)

    @staticmethod
    def _get_key(item: Expression) -> PrintedKey:
        """Return the key of an expression in tables.

        The key equals the keys of structurally identical expressions. Sage's `==` builds a relation whose truth Maxima
        may be asked to prove, so tables of expressions are keyed by the structural hash and the printed form of an
        expression instead.
        """
        return hash(item), str(item)

    @staticmethod
    def _get_subexpressions(expressions: Sequence[Expression]) -> Counter[PrintedKey]:
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
            key = ExpressionPrinter._get_key(item)
            counts[key] += 1
            if counts[key] == 1:
                stack.extend(item.operands())
        return counts
