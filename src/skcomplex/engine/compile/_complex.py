"""Split complex expressions and laws into real expressions of their real and imaginary parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from sage.all__sagemath_symbolics import SR, Expression, I, arctan2, sqrt

from ...errors import ValidationError
from ..types import Parts
from ._members import _SHIFT

_LINEAR = frozenset({'laplacian', 'inverse_laplacian', 'partial', 'upwind', 'derivative', 'lag', 'delayed'})
_SUMS = frozenset({'total', 'mean'})
_REAL_OPERANDS = frozenset(
    {
        'mod',
        'either',
        'both',
        'maximum',
        'minimum',
        'variance',
        'count',
        'floor',
        'ceil',
        'sgn',
        'sign',
        'heaviside',
        'unit_step',
        'min_symbolic',
        'max_symbolic',
    },
)
_WHERE = 'where'
_UNSPLIT = frozenset({'real_part', 'imag_part', 'conjugate', 'arg', 'abs'})
_LEFTOVER = _UNSPLIT - {'abs'}
_STAND_IN = '_split_part_'


def _name_of(expression: Expression) -> str:
    """Return the name of the function at the top of an expression, or an empty string."""
    return str(getattr(expression.operator(), 'name', lambda: '')())


def _names_in(expression: Expression) -> set[str]:
    """Return the names of the functions applied anywhere in an expression."""
    names, stack = set(), [expression]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        names.add(_name_of(item))
        stack.extend(item.operands())
    return names


def _is_zero(expression: Expression) -> bool:
    """Check whether a real expression is zero as written, without asking Maxima."""
    return bool(SR(expression).is_trivial_zero())


def _checked(part: Expression, expression: Expression, owner: str) -> Expression:
    """Return a real or imaginary part with every modulus written out, after checking that no complex value is left."""
    part = _modulus_form(part)
    left = sorted(_names_in(part) & _LEFTOVER)
    if not part.has(I) and not left:
        return part
    error_msg = (
        f'The {owner} applies a function to a complex value in {expression} that the compiler cannot write with real '
        'and imaginary parts. Use arithmetic, the elementary functions, real_part, imag_part, conjugate, abs, and '
        'arg on complex values.'
    )
    raise ValidationError(error_msg)


def _stand_in(part: Expression, stand_ins: dict[Expression, Expression]) -> Expression:
    """Return a real symbol that stands for a real part while Sage splits the function applied to it."""
    if part.operator() is None:
        return part
    item = SR.symbol(f'{_STAND_IN}{len(stand_ins)}')
    item.decl_assume('real')
    stand_ins[item] = part
    return item


def _real(parts: Parts, expression: Expression, owner: str) -> Expression:
    """Return the real part of a value whose imaginary part must be zero, as an operand that only takes real values."""
    real, imaginary = parts
    if _is_zero(imaginary):
        return real
    error_msg = (
        f'The {owner} gives the complex value {expression} where only real values have a meaning. Take its modulus '
        '`abs(...)`, its real part `real_part(...)`, or its imaginary part `imag_part(...)` first.'
    )
    raise ValidationError(error_msg)


def _split_relation(expression: Expression, owner: str) -> Expression:
    """Return a relation of real operands, rejecting the order of complex values."""
    sides = []
    for side in (expression.lhs(), expression.rhs()):
        real, imaginary = _split_complex(side, owner)
        if not _is_zero(imaginary):
            error_msg = (
                f'The {owner} compares the complex value {side} in {expression}, and complex values have no order. '
                'Compare its modulus `abs(...)` or its real part `real_part(...)`.'
            )
            raise ValidationError(error_msg)
        sides.append(real)
    return expression.operator()(*sides)


def _split_part(name: str, operand: Expression, owner: str) -> Parts:
    """Return the parts of the modulus, argument, real part, imaginary part, or conjugate of a value."""
    real, imaginary = _split_complex(operand, owner)
    if name == 'abs':
        return (abs(real) if _is_zero(imaginary) else sqrt(real**2 + imaginary**2)), SR(0)
    values = {
        'real_part': (real, SR(0)),
        'imag_part': (imaginary, SR(0)),
        'conjugate': (real, -imaginary),
        'arg': (arctan2(imaginary, real), SR(0)),
    }
    return values[name]


def _split_where(function: Callable[..., Expression], operands: list[Expression], owner: str) -> Parts:
    """Return the parts of a choice between two values, whose condition takes real values."""
    condition = _real(_split_complex(operands[0], owner), operands[0], owner)
    (real, imaginary), (other_real, other_imaginary) = (_split_complex(item, owner) for item in operands[1:])
    return function(condition, real, other_real), function(condition, imaginary, other_imaginary)


def _split_function(expression: Expression, owner: str) -> Parts | None:
    """Return the parts of a call of a modulus, argument, part, linear operator, or reduction, or `None` otherwise."""
    name, operands, function = _name_of(expression), list(expression.operands()), expression.operator()
    if name in _UNSPLIT:
        return _split_part(name, operands[0], owner)
    if name in _LINEAR or name in _SUMS:
        real, imaginary = _split_complex(operands[0], owner)
        rest = [_real(_split_complex(item, owner), item, owner) for item in operands[1:]]
        return function(real, *rest), function(imaginary, *rest)
    if name == _SHIFT:
        real, imaginary = _split_complex(operands[0], owner)
        return function(real, *operands[1:]), function(imaginary, *operands[1:3], 0.0)
    if name == _WHERE:
        return _split_where(function, operands, owner)
    if name in _REAL_OPERANDS:
        return function(*(_real(_split_complex(item, owner), item, owner) for item in operands)), SR(0)
    return None


def _modulus_form(expression: Expression) -> Expression:
    """Return an expression with every modulus of a complex value written with its real and imaginary parts.

    Args:
        expression:
            The expression.

    Returns:
        expression:
            The expression with `abs(a + I*b)` written as `sqrt(a^2 + b^2)`.
    """
    if expression.operator() is None or not expression.has(I):
        return expression
    operands = [_modulus_form(operand) for operand in expression.operands()]
    if _name_of(expression) == 'abs':
        return sqrt(operands[0].real_part() ** 2 + operands[0].imag_part() ** 2)
    return expression.operator()(*operands)


def _split_complex(expression: object, owner: str) -> Parts:
    """Return the real and imaginary parts of an expression of real symbols as real expressions.

    Sage writes the parts of arithmetic and of the elementary functions. The modulus, argument, parts, and conjugate,
    the linear operators of fields and companions, sums and means over groups, and member offset reads are split
    term by term, and relations and the other functions take real operands.

    Args:
        expression:
            The expression, whose symbols are all real.

        owner:
            The law and the path of the component that holds the expression, used in error messages.

    Returns:
        parts:
            The real part and the imaginary part.

    Raises:
        ValidationError:
            The expression orders complex values, gives a complex value to a function that takes real values, or applies
            a function whose parts Sage cannot write.
    """
    parsed: Expression = SR(expression)
    if not parsed.has(I):
        return parsed, SR(0)
    if parsed.is_relational():
        return _split_relation(parsed, owner), SR(0)
    if parsed.operator() is None:
        number = parsed.pyobject()
        if isinstance(number, complex):
            return SR(number.real), SR(number.imag)
        return parsed.real_part(), parsed.imag_part()
    parts = _split_function(parsed, owner)
    if parts is not None:
        return parts
    stand_ins: dict[Expression, Expression] = {}
    operands = [
        _stand_in(real, stand_ins) + I * _stand_in(imaginary, stand_ins)
        for real, imaginary in (_split_complex(item, owner) for item in parsed.operands())
    ]
    rebuilt = parsed.operator()(*operands)
    real, imaginary = _checked(rebuilt.real_part(), parsed, owner), _checked(rebuilt.imag_part(), parsed, owner)
    return real.subs(stand_ins), imaginary.subs(stand_ins)
