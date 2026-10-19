"""Print Sage expressions in the syntax of LAMMPS Lepton expressions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR

from ..base import BackendError, Domain, Expression, symbol, symbols_of
from ..compile import Dialect, print_expressions

LEPTON = Dialect(
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


class LeptonPrinter:
    """Printer of Sage expressions as Lepton expressions, with `^` for powers and Lepton function names.

    Conditions become `step` and `select`, and remainders become `floor`, because Lepton has neither comparisons nor
    a remainder.

    Examples:
        >>> from sage.all__sagemath_symbolics import exp
        >>> from skcomplex.core.backends import LeptonPrinter
        >>> from skcomplex.core.base import symbol
        >>> r = symbol('r')
        >>> LeptonPrinter().doprint(2 * exp(-r / 3) + r**2)
        'r^2 + 2*exp(-r/3)'
    """

    def doprint(self, expression: Expression) -> str:
        """Return the Lepton code of an expression.

        Args:
            expression:
                The expression.

        Returns:
            code:
                The Lepton code.

        Raises:
            BackendError:
                The expression uses a function Lepton does not know.
        """
        return print_expressions([expression], LEPTON, shared=False)[1][0]


def lepton_expression(expression: Expression, distance: Expression) -> str:
    """Return a pair energy as a Lepton expression of the distance `r`.

    Args:
        expression:
            The pair energy, whose only symbol is the distance.

        distance:
            The distance symbol.

    Returns:
        expression:
            The Lepton expression.

    Raises:
        BackendError:
            The energy has symbols other than the distance or functions Lepton does not know.

    Examples:
        >>> from skcomplex.core.backends import lepton_expression
        >>> from skcomplex.core.base import Domain, symbol
        >>> x = symbol('x', Domain.POSITIVE)
        >>> lepton_expression(4 * (x**-12 - x**-6), x)
        '-4/r^6 + 4/r^12'
    """
    expression = SR(expression)
    others = symbols_of([expression]) - {distance}
    if others:
        names = ', '.join(sorted(map(str, others)))
        error_msg = f'The pair energy depends on {names}, which have no value for LAMMPS.'
        raise BackendError(error_msg)
    return LeptonPrinter().doprint(expression.subs({distance: symbol('r', Domain.POSITIVE)}).expand())
