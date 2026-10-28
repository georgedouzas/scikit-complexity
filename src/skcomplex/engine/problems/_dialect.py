"""Syntax of a target language for printed expressions, with the dialects of NumPy, scalar Python, and Lepton."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...core.errors import BackendError, ValidationError
from ._base import BaseValue

_PYTHON_RELATIONS = {
    'lt': '({0} < {1})',
    'le': '({0} <= {1})',
    'gt': '({0} > {1})',
    'ge': '({0} >= {1})',
    'eq': '({0} == {1})',
    'ne': '({0} != {1})',
}


@dataclass(frozen=True)
class Dialect(BaseValue):
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


_NUMPY_DIALECT = Dialect(
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
    _PYTHON_RELATIONS,
    complex_numbers=True,
)


_SCALAR_DIALECT = Dialect(
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
    _PYTHON_RELATIONS,
)


_LEPTON_DIALECT = Dialect(
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


DIALECTS = {'numpy': _NUMPY_DIALECT, 'scalar': _SCALAR_DIALECT, 'lepton': _LEPTON_DIALECT}
