"""Split complex expressions and laws into real expressions of their real and imaginary parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import numbers
from collections.abc import Callable

import numpy as np
from sage.all__sagemath_symbolics import SR, I, arctan2, sqrt

from ..base import Expression, SymbolicVector, ValidationError, components_of
from ..model import Boundary, Call, Equation, Event, Jump, Law, Noise, Rate, Rule, Update
from ._lowering import symbolic_components

LINEAR = frozenset({'laplacian', 'inverse_laplacian', 'partial', 'upwind', 'derivative', 'lag', 'delayed'})
SUMS = frozenset({'total', 'mean'})
REAL_OPERANDS = frozenset(
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
SHIFT = 'shift_member'
WHERE = 'where'
UNSPLIT = frozenset({'real_part', 'imag_part', 'conjugate', 'arg', 'abs'})
LEFTOVER = UNSPLIT - {'abs'}
STAND_IN = '_split_part_'

type Parts = tuple[Expression, Expression]


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


def modulus_form(expression: Expression) -> Expression:
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
    operands = [modulus_form(operand) for operand in expression.operands()]
    if _name_of(expression) == 'abs':
        return sqrt(operands[0].real_part() ** 2 + operands[0].imag_part() ** 2)
    return expression.operator()(*operands)


def _checked(part: Expression, expression: Expression, owner: str) -> Expression:
    """Return a real or imaginary part with every modulus written out, after checking that no complex value is left."""
    part = modulus_form(part)
    left = sorted(_names_in(part) & LEFTOVER)
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
    item = SR.symbol(f'{STAND_IN}{len(stand_ins)}')
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
        real, imaginary = split_complex(side, owner)
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
    real, imaginary = split_complex(operand, owner)
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
    condition = _real(split_complex(operands[0], owner), operands[0], owner)
    (real, imaginary), (other_real, other_imaginary) = (split_complex(item, owner) for item in operands[1:])
    return function(condition, real, other_real), function(condition, imaginary, other_imaginary)


def _split_function(expression: Expression, owner: str) -> Parts | None:
    """Return the parts of a call of a modulus, argument, part, linear operator, or reduction, or `None` otherwise."""
    name, operands, function = _name_of(expression), list(expression.operands()), expression.operator()
    if name in UNSPLIT:
        return _split_part(name, operands[0], owner)
    if name in LINEAR or name in SUMS:
        real, imaginary = split_complex(operands[0], owner)
        rest = [_real(split_complex(item, owner), item, owner) for item in operands[1:]]
        return function(real, *rest), function(imaginary, *rest)
    if name == SHIFT:
        real, imaginary = split_complex(operands[0], owner)
        return function(real, *operands[1:]), function(imaginary, *operands[1:3], 0.0)
    if name == WHERE:
        return _split_where(function, operands, owner)
    if name in REAL_OPERANDS:
        return function(*(_real(split_complex(item, owner), item, owner) for item in operands)), SR(0)
    return None


def has_complex(value: object) -> bool:
    """Check whether a law value or one of its components holds a complex number.

    Args:
        value:
            An expression, a vector, a number, a sequence of them, or any other value.

    Returns:
        is_complex:
            Whether the value holds the imaginary unit or a number with a nonzero imaginary part.
    """
    if isinstance(value, Expression):
        return bool(value.has(I))
    if isinstance(value, SymbolicVector | list | tuple | np.ndarray):
        return any(has_complex(item) for item in np.ravel(np.asarray(list(value), dtype=object)))
    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        return complex(value).imag != 0
    return False


def split_complex(expression: object, owner: str) -> Parts:
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
        for real, imaginary in (split_complex(item, owner) for item in parsed.operands())
    ]
    rebuilt = parsed.operator()(*operands)
    real, imaginary = _checked(rebuilt.real_part(), parsed, owner), _checked(rebuilt.imag_part(), parsed, owner)
    return real.subs(stand_ins), imaginary.subs(stand_ins)


def joined_complex(expression: object, owner: str) -> Expression:
    """Return an expression as its real part plus the imaginary unit times its imaginary part, both real expressions.

    Args:
        expression:
            The expression, whose symbols are all real.

        owner:
            The owner of the expression, used in error messages.

    Returns:
        expression:
            The expression with every complex function written out, ready to print with complex numbers.

    Raises:
        ValidationError:
            The expression cannot be split into real and imaginary parts.
    """
    real, imaginary = split_complex(expression, owner)
    return real if _is_zero(imaginary) else real + I * imaginary


def _split_value(
    targets: list[Expression],
    values: list[Expression],
    description: str,
    state_of: Callable[[Expression], str],
) -> tuple[list[Expression], list[Expression]]:
    """Return the real targets and real values of a law, two of each for every complex target."""
    real_targets, real_values = [], []
    for target, value in zip(targets, values, strict=True):
        real, imaginary = split_complex(value, description)
        if has_complex(target):
            real_targets += [target.real_part(), target.imag_part()]
            real_values += [real, imaginary]
            continue
        if not _is_zero(imaginary):
            error_msg = (
                f'The {description} gives the complex value {value} to the real state {state_of(target)}. Take its '
                'real part `real_part(...)`, its imaginary part `imag_part(...)`, or its modulus `abs(...)`, or '
                'declare the state with `domain=Domain.COMPLEX`.'
            )
            raise ValidationError(error_msg)
        real_targets.append(target)
        real_values.append(real)
    return real_targets, real_values


def _split_assignment(
    law: Rate | Noise | Update | Boundary,
    owner: str,
    state_of: Callable[[Expression], str],
) -> list[Law]:
    """Return the laws of one real target component each that a rate, noise, update, or boundary law stands for."""
    description = f'{type(law).__name__} law of {owner}'
    targets = components_of(law.target)
    values = None if isinstance(law.value, Call) else symbolic_components(law.value, len(targets), description)
    if values is None:
        error_msg = (
            f'The {description} on {state_of(targets[0])} calls a function with complex values or on a complex state. '
            'Functions take and give real values, so pass `real_part(...)` and `imag_part(...)` instead.'
        )
        raise ValidationError(error_msg)
    real_targets, real_values = _split_value(targets, values, description, state_of)
    if isinstance(law, Rate | Noise):
        return [type(law)(target, value, law.order) for target, value in zip(real_targets, real_values, strict=True)]
    return [type(law)(target, value) for target, value in zip(real_targets, real_values, strict=True)]


def _split_event(event: Event, owner: str, state_of: Callable[[Expression], str]) -> Event:
    """Return an event with a real guard and a reset of real states."""
    description = f'Event of {owner}'
    guard = _real(split_complex(event.guard, description), SR(event.guard), description)
    if event.reset is None:
        return dataclasses.replace(event, guard=guard)
    targets = [SR(target) for target in event.reset]
    values = [SR(value) for value in event.reset.values()]
    real_targets, real_values = _split_value(targets, values, description, state_of)
    return dataclasses.replace(event, guard=guard, reset=dict(zip(real_targets, real_values, strict=True)))


def _split_jump(jump: Jump, owner: str) -> Jump:
    """Return a jump with a real rate, after checking that it changes real states."""
    description = f'Jump law of {owner}'
    if any(has_complex(target) for target in jump.changes):
        error_msg = f'The {description} changes a complex state. Jumps change real states by real increments.'
        raise ValidationError(error_msg)
    return Jump(_real(split_complex(jump.rate, description), SR(jump.rate), description), jump.changes)


def _split_equation(equation: Equation, owner: str) -> list[Law]:
    """Return the real equations of the real and imaginary parts of a complex equation."""
    description = f'Equation of {owner}'
    real_lhs, imaginary_lhs = split_complex(equation.lhs, description)
    real_rhs, imaginary_rhs = split_complex(equation.rhs, description)
    equations: list[Law] = [Equation(real_lhs, real_rhs)]
    if not (_is_zero(imaginary_lhs) and _is_zero(imaginary_rhs)):
        equations.append(Equation(imaginary_lhs, imaginary_rhs))
    return equations


def _law_fields(law: Law) -> list[object]:
    """Return the values of the fields of a law, with the arguments of a called function."""
    values = [getattr(law, item.name) for item in dataclasses.fields(law)] if dataclasses.is_dataclass(law) else []
    return [item for value in values for item in (value.arguments if isinstance(value, Call) else (value,))]


def _mapped_items(law: Law) -> list[object]:
    """Return the states an event resets with their new values, or the states a jump changes, or nothing."""
    if isinstance(law, Event) and law.reset is not None:
        return [*law.reset, *law.reset.values()]
    if isinstance(law, Jump):
        return list(law.changes)
    return []


def split_law(law: Law, owner: str, state_of: Callable[[Expression], str]) -> list[Law]:
    """Return the laws of real states and real values that a law of complex states and values stands for.

    A law with no complex value is returned unchanged. A law with a complex value becomes one law per real component
    of its target, two for every complex component, and an equation with complex sides becomes the equations of their
    parts.

    Args:
        law:
            The law.

        owner:
            The path of the component that returned the law.

        state_of:
            The function that returns the dotted name of the state a target symbol stands for.

    Returns:
        laws:
            The real laws.

    Raises:
        ValidationError:
            A real state takes a complex value, a law orders complex values, a callable or a jump targets a complex
            state, or a law of a kind that takes only real values holds a complex value.
    """
    if isinstance(law, Rule) or not any(has_complex(item) for item in [*_law_fields(law), *_mapped_items(law)]):
        return [law]
    if isinstance(law, Rate | Noise | Update | Boundary):
        return _split_assignment(law, owner, state_of)
    if isinstance(law, Event):
        return [_split_event(law, owner, state_of)]
    if isinstance(law, Jump):
        return [_split_jump(law, owner)]
    if isinstance(law, Equation):
        return _split_equation(law, owner)
    error_msg = (
        f'A {type(law).__name__} law of {owner} holds a complex value. {type(law).__name__} laws take real states and '
        'real values.'
    )
    raise ValidationError(error_msg)
