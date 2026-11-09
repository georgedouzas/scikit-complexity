"""Split the laws of complex states into laws of their real and imaginary parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers
from collections.abc import Callable

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, I, arctan2, sqrt
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import BaseLaw, BoundaryLaw, CrossingLaw, EquationLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....errors import ValidationError
from ....specs import CallSpec
from ..._utils import components_of
from ...types import Parts
from .._base import BasePass
from .._compilation import Compilation
from .._laws import AGENT_LAWS
from .._members import SHIFT
from ._kernels import KernelsPass

LINEAR = frozenset({'laplacian', 'inverse_helmholtz', 'partial', 'upwind', 'derivative', 'lag', 'delayed'})
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
WHERE = 'where'
UNSPLIT = frozenset({'real_part', 'imag_part', 'conjugate', 'arg', 'abs'})
LEFTOVER = UNSPLIT - {'abs'}
STAND_IN = '_split_part_'


class LowerComplexPass(BasePass):
    """Pass that splits every law of a complex state into laws of its parts.

    The parts are the real and imaginary parts.
    """

    @staticmethod
    def _get_name_of(expression: Expression) -> str:
        """Return the name of the function at the top of an expression, or an empty string."""
        return str(getattr(expression.operator(), 'name', lambda: '')())

    @staticmethod
    def _get_names_in(expression: Expression) -> set[str]:
        """Return the names of the functions applied anywhere in an expression."""
        names, stack = set(), [expression]
        while stack:
            item = stack.pop()
            if item.operator() is None:
                continue
            names.add(LowerComplexPass._get_name_of(item))
            stack.extend(item.operands())
        return names

    @staticmethod
    def _is_zero(expression: Expression) -> bool:
        """Check whether a real expression is zero as written, without asking Maxima."""
        return bool(SR(expression).is_trivial_zero())

    @staticmethod
    def _get_checked(part: Expression, expression: Expression, owner: str) -> Expression:
        """Return a real or imaginary part with every modulus written out.

        It first checks that no complex value is left.
        """
        part = LowerComplexPass._get_modulus_form(part)
        left = sorted(LowerComplexPass._get_names_in(part) & LEFTOVER)
        if not part.has(I) and not left:
            return part
        error_msg = (
            f'The {owner} applies a function to a complex value in {expression} that the compiler cannot write with '
            'real '
            'and imaginary parts. Use arithmetic, the elementary functions, real_part, imag_part, conjugate, abs, and '
            'arg on complex values.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _get_stand_in(part: Expression, stand_ins: dict[Expression, Expression]) -> Expression:
        """Return a real symbol that stands for a real part.

        The symbol stands for the part while Sage splits the function applied to it.
        """
        if part.operator() is None:
            return part
        item = SR.symbol(f'{STAND_IN}{len(stand_ins)}')
        item.decl_assume('real')
        stand_ins[item] = part
        return item

    @staticmethod
    def _get_real(parts: Parts, expression: Expression, owner: str) -> Expression:
        """Return the real part of a value whose imaginary part must be zero.

        The part is returned as an operand that only takes real values.
        """
        real, imaginary = parts
        if LowerComplexPass._is_zero(imaginary):
            return real
        error_msg = (
            f'The {owner} gives the complex value {expression} where only real values have a meaning. Take its modulus '
            '`abs(...)`, its real part `real_part(...)`, or its imaginary part `imag_part(...)` first.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _split_relation(expression: Expression, owner: str) -> Expression:
        """Return a relation of real operands, rejecting the order of complex values."""
        sides = []
        for side in (expression.lhs(), expression.rhs()):
            real, imaginary = LowerComplexPass._split_complex(side, owner)
            if not LowerComplexPass._is_zero(imaginary):
                error_msg = (
                    f'The {owner} compares the complex value {side} in {expression}, and complex values have no order. '
                    'Compare its modulus `abs(...)` or its real part `real_part(...)`.'
                )
                raise ValidationError(error_msg)
            sides.append(real)
        return expression.operator()(*sides)

    @staticmethod
    def _split_part(name: str, operand: Expression, owner: str) -> Parts:
        """Return the parts of a modulus, argument, part, or conjugate of a value.

        The parts are those of the modulus, argument, real part, imaginary part, or conjugate.
        """
        real, imaginary = LowerComplexPass._split_complex(operand, owner)
        if name == 'abs':
            return (abs(real) if LowerComplexPass._is_zero(imaginary) else sqrt(real**2 + imaginary**2)), SR(0)
        values = {
            'real_part': (real, SR(0)),
            'imag_part': (imaginary, SR(0)),
            'conjugate': (real, -imaginary),
            'arg': (arctan2(imaginary, real), SR(0)),
        }
        return values[name]

    @staticmethod
    def _split_where(function: Callable[..., Expression], operands: list[Expression], owner: str) -> Parts:
        """Return the parts of a choice between two values.

        The condition of the choice takes real values.
        """
        condition = LowerComplexPass._get_real(LowerComplexPass._split_complex(operands[0], owner), operands[0], owner)
        (real, imaginary), (other_real, other_imaginary) = (
            LowerComplexPass._split_complex(item, owner) for item in operands[1:]
        )
        return function(condition, real, other_real), function(condition, imaginary, other_imaginary)

    @staticmethod
    def _split_function(expression: Expression, owner: str) -> Parts | None:
        """Return the parts of a call of a modulus, argument, part, operator, or reduction.

        The operator is linear, and the result is `None` for any other call.
        """
        name, operands, function = (
            LowerComplexPass._get_name_of(expression),
            list(expression.operands()),
            expression.operator(),
        )
        if name in UNSPLIT:
            return LowerComplexPass._split_part(name, operands[0], owner)
        if name in LINEAR or name in SUMS:
            real, imaginary = LowerComplexPass._split_complex(operands[0], owner)
            rest = [
                LowerComplexPass._get_real(LowerComplexPass._split_complex(item, owner), item, owner)
                for item in operands[1:]
            ]
            return function(real, *rest), function(imaginary, *rest)
        if name == SHIFT:
            real, imaginary = LowerComplexPass._split_complex(operands[0], owner)
            return function(real, *operands[1:]), function(imaginary, *operands[1:3], 0.0)
        if name == WHERE:
            return LowerComplexPass._split_where(function, operands, owner)
        if name in REAL_OPERANDS:
            return function(
                *(
                    LowerComplexPass._get_real(LowerComplexPass._split_complex(item, owner), item, owner)
                    for item in operands
                ),
            ), SR(0)
        return None

    @staticmethod
    def _get_modulus_form(expression: Expression) -> Expression:
        """Return an expression with every modulus of a complex value written out.

        Every modulus is written with the real and imaginary parts of its value.

        Args:
            expression:
                The expression.

        Returns:
            expression:
                The expression with `abs(a + I*b)` written as `sqrt(a^2 + b^2)`.
        """
        if expression.operator() is None or not expression.has(I):
            return expression
        operands = [LowerComplexPass._get_modulus_form(operand) for operand in expression.operands()]
        if LowerComplexPass._get_name_of(expression) == 'abs':
            return sqrt(operands[0].real_part() ** 2 + operands[0].imag_part() ** 2)
        return expression.operator()(*operands)

    @staticmethod
    def _split_complex(expression: object, owner: str) -> Parts:
        """Return the real and imaginary parts of an expression of real symbols.

        The parts are real expressions. Sage writes the parts of arithmetic and of the elementary functions. The
        modulus, argument, parts, and conjugate, the linear operators of fields and companions, sums and means over
        groups, and member offset reads are split term by term, and relations and the other functions take real
        operands.

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
                The expression orders complex values, gives a complex value to a function that takes real values, or
                applies a function whose parts Sage cannot write.
        """
        parsed: Expression = SR(expression)
        if not parsed.has(I):
            return parsed, SR(0)
        if parsed.is_relational():
            return LowerComplexPass._split_relation(parsed, owner), SR(0)
        if parsed.operator() is None:
            number = parsed.pyobject()
            if isinstance(number, complex):
                return SR(number.real), SR(number.imag)
            return parsed.real_part(), parsed.imag_part()
        parts = LowerComplexPass._split_function(parsed, owner)
        if parts is not None:
            return parts
        stand_ins: dict[Expression, Expression] = {}
        operands = [
            LowerComplexPass._get_stand_in(real, stand_ins) + I * LowerComplexPass._get_stand_in(imaginary, stand_ins)
            for real, imaginary in (LowerComplexPass._split_complex(item, owner) for item in parsed.operands())
        ]
        rebuilt = parsed.operator()(*operands)
        real, imaginary = LowerComplexPass._get_checked(
            rebuilt.real_part(),
            parsed,
            owner,
        ), LowerComplexPass._get_checked(rebuilt.imag_part(), parsed, owner)
        return real.subs(stand_ins), imaginary.subs(stand_ins)

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered',)

    def _run(self, inputs: object = None) -> None:
        """Split the collected laws of complex states."""
        compilation = self.compilation_
        compilation.lowered = [
            (
                spec,
                [item for law in laws for item in self._split_law(law, spec.owner, spec._get_state_name)],
                block,
            )
            for spec, laws, block in Compilation._get_required(compilation.collected, 'collected')
        ]

    @staticmethod
    def _split_value(
        targets: list[Expression],
        values: list[Expression],
        description: str,
        state_of: Callable[[Expression], str],
    ) -> tuple[list[Expression], list[Expression]]:
        """Return the real targets and real values of a law.

        There are two of each for every complex target.
        """
        real_targets, real_values = [], []
        for target, value in zip(targets, values, strict=True):
            real, imaginary = LowerComplexPass._split_complex(value, description)
            if LowerComplexPass._has_complex(target):
                real_targets += [target.real_part(), target.imag_part()]
                real_values += [real, imaginary]
                continue
            if not LowerComplexPass._is_zero(imaginary):
                error_msg = (
                    f'The {description} gives the complex value {value} to the real state {state_of(target)}. Take its '
                    'real part `real_part(...)`, its imaginary part `imag_part(...)`, or its modulus `abs(...)`, or '
                    'declare the state with `domain=CC`.'
                )
                raise ValidationError(error_msg)
            real_targets.append(target)
            real_values.append(real)
        return real_targets, real_values

    @staticmethod
    def _split_assignment(
        law: RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw,
        owner: str,
        state_of: Callable[[Expression], str],
    ) -> list[BaseLaw]:
        """Return the laws of one real target component each that a law stands for.

        The law is a rate, noise, update, or boundary law.
        """
        description = f'{type(law).__name__} of {owner}'
        targets = components_of(law._get_targets()[0])
        values = (
            None
            if isinstance(law.value, CallSpec)
            else KernelsPass._get_symbolic_components(law.value, len(targets), description)
        )
        if values is None:
            error_msg = (
                f'The {description} on {state_of(targets[0])} calls a function with complex values or on a complex '
                'state. Functions take and give real values, so pass `real_part(...)` and `imag_part(...)` instead.'
            )
            raise ValidationError(error_msg)
        real_targets, real_values = LowerComplexPass._split_value(targets, values, description, state_of)
        return [
            BasePass._get_replaced(law, target=target, value=value)
            for target, value in zip(real_targets, real_values, strict=True)
        ]

    @staticmethod
    def _split_equation(equation: EquationLaw, owner: str) -> list[BaseLaw]:
        """Return the real equations of the real and imaginary parts of a complex equation."""
        description = f'Equation of {owner}'
        real_lhs, imaginary_lhs = LowerComplexPass._split_complex(equation.lhs_, description)
        real_rhs, imaginary_rhs = LowerComplexPass._split_complex(equation.rhs_, description)
        real = LowerComplexPass._is_zero(imaginary_lhs) and LowerComplexPass._is_zero(imaginary_rhs)
        target = equation.target if real else None
        equations: list[BaseLaw] = [EquationLaw(real_lhs, real_rhs, target=target).prepare()]
        if not real:
            equations.append(EquationLaw(imaginary_lhs, imaginary_rhs).prepare())
        return equations

    @staticmethod
    def _split_event(event: CrossingLaw, owner: str, state_of: Callable[[Expression], str]) -> CrossingLaw:
        """Return an event with a real guard and a reset of real states."""
        description = f'Event of {owner}'
        guard = LowerComplexPass._get_real(
            LowerComplexPass._split_complex(event.guard, description),
            SR(event.guard),
            description,
        )
        if event.reset is None:
            return BasePass._get_replaced(event, guard=guard)
        targets = [SR(target) for target in event.reset]
        values = [SR(value) for value in event.reset.values()]
        real_targets, real_values = LowerComplexPass._split_value(targets, values, description, state_of)
        return BasePass._get_replaced(event, guard=guard, reset=dict(zip(real_targets, real_values, strict=True)))

    @staticmethod
    def _split_jump(jump: JumpLaw, owner: str) -> JumpLaw:
        """Return a jump with a real rate, after checking that it changes real states."""
        description = f'JumpLaw of {owner}'
        if any(LowerComplexPass._has_complex(target) for target in jump.changes or {}):
            error_msg = f'The {description} changes a complex state. Jumps change real states by real increments.'
            raise ValidationError(error_msg)
        return JumpLaw(
            LowerComplexPass._get_real(
                LowerComplexPass._split_complex(jump.rate, description),
                SR(jump.rate),
                description,
            ),
            jump.changes,
        )

    @staticmethod
    def _has_complex(value: object) -> bool:
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
        if isinstance(value, np.ndarray) and value.dtype != object:
            return bool(np.iscomplexobj(value) and np.any(value.imag != 0))
        if isinstance(value, np.ndarray):
            return any(LowerComplexPass._has_complex(item) for item in value.ravel())
        if isinstance(value, FreeModuleElement | list | tuple):
            return any(LowerComplexPass._has_complex(item) for item in np.ravel(np.asarray(list(value), dtype=object)))
        if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
            return complex(value).imag != 0
        return False

    @staticmethod
    def _split_law(law: BaseLaw, owner: str, state_of: Callable[[Expression], str]) -> list[BaseLaw]:
        """Return the laws of real states and values that a complex law stands for.

        The complex law has complex states and values. A law with no complex value is returned unchanged. A law with a
        complex value becomes one law per real component of its target, two for every complex component, and an equation
        with complex sides becomes the equations of their parts.

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
        if isinstance(law, AGENT_LAWS) or not any(LowerComplexPass._has_complex(item) for item in law._get_values()):
            return [law]
        if isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
            return LowerComplexPass._split_assignment(law, owner, state_of)
        if isinstance(law, CrossingLaw):
            return [LowerComplexPass._split_event(law, owner, state_of)]
        if isinstance(law, JumpLaw):
            return [LowerComplexPass._split_jump(law, owner)]
        if isinstance(law, EquationLaw):
            return LowerComplexPass._split_equation(law, owner)
        error_msg = (
            f'A {type(law).__name__} of {owner} holds a complex value. {type(law).__name__}s take real states and '
            'real values.'
        )
        raise ValidationError(error_msg)
