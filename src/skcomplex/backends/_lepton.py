"""Print SymPy expressions in the syntax of LAMMPS Lepton expressions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sympy
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter

from ..core import BackendError

FUNCTIONS = {
    'exp': 'exp',
    'log': 'log',
    'sqrt': 'sqrt',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'asin': 'asin',
    'acos': 'acos',
    'atan': 'atan',
    'sinh': 'sinh',
    'cosh': 'cosh',
    'tanh': 'tanh',
    'erf': 'erf',
    'erfc': 'erfc',
    'Abs': 'abs',
    'Min': 'min',
    'Max': 'max',
}


class LeptonPrinter(StrPrinter):
    """Printer of SymPy expressions as Lepton expressions, with `^` for powers and Lepton function names.

    Examples:
        >>> import sympy
        >>> from skcomplex.backends import LeptonPrinter
        >>> r = sympy.Symbol('r')
        >>> LeptonPrinter().doprint(2 * sympy.exp(-r / 3) + r**2)
        'r^2 + 2*exp(-r/3)'
    """

    def _print_Pow(self, expr: sympy.Pow, rational: bool = False) -> str:
        """Print a power with `^` and both sides in parentheses where needed."""
        base, exponent = expr.as_base_exp()
        if exponent == sympy.Rational(1, 2):
            return f'sqrt({self._print(base)})'
        precedence = PRECEDENCE['Pow']
        return f'{self.parenthesize(base, precedence)}^{self.parenthesize(exponent, precedence + 1)}'

    def _print_Rational(self, expr: sympy.Rational) -> str:
        """Print a rational as a division of two integers."""
        return f'{expr.p}/{expr.q}'

    def _print_Function(self, expr: sympy.Function) -> str:
        """Print a function with its Lepton name, or raise an error when Lepton has none."""
        name = type(expr).__name__
        if name not in FUNCTIONS:
            error_msg = f'The function {name} has no Lepton equivalent.'
            raise BackendError(error_msg)
        return f'{FUNCTIONS[name]}({", ".join(self._print(argument) for argument in expr.args)})'

    def emptyPrinter(self, expr: object) -> str:
        """Raise an error for an expression that has no Lepton equivalent."""
        error_msg = f'{type(expr).__name__} has no Lepton equivalent.'
        raise BackendError(error_msg)

    def _print_Piecewise(self, expr: sympy.Piecewise) -> str:
        """Raise an error, because Lepton expressions have no conditional branches."""
        error_msg = 'Piecewise expressions have no Lepton equivalent.'
        raise BackendError(error_msg)


def lepton_expression(expression: sympy.Expr, distance: sympy.Symbol) -> str:
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
        >>> import sympy
        >>> from skcomplex.backends import lepton_expression
        >>> x = sympy.Symbol('x', positive=True)
        >>> lepton_expression(4 * (x**-12 - x**-6), x)
        '-4/r^6 + 4/r^12'
    """
    others = expression.free_symbols - {distance}
    if others:
        names = ', '.join(sorted(map(str, others)))
        error_msg = f'The pair energy depends on {names}, which have no value for LAMMPS.'
        raise BackendError(error_msg)
    return LeptonPrinter().doprint(sympy.expand(expression.subs(distance, sympy.Symbol('r', positive=True))))
