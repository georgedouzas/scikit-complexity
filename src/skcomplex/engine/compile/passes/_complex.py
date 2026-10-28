"""Split the laws of complex states into laws of their real and imaginary parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers
from collections.abc import Callable
from functools import partial
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, I
from sage.modules.free_module_element import FreeModuleElement

from ....core.errors import ValidationError
from ....core.laws import BaseLaw, BoundaryLaw, CrossingLaw, EquationLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.laws.utils import Call
from ...problems import components_of
from .._compilation import BlockLaws, Compilation
from .._complex import _is_zero, _real, _split_complex
from .._kernel_spec import KernelSpec
from .._laws import _AGENT_LAWS
from .._pipeline import BasePass
from ._kernels import KernelsPass


class LowerComplexPass(BasePass):
    """Pass that splits every law of a complex state into the laws of its real and imaginary parts."""

    reads = ('collected',)
    writes = ('lowered',)

    def _run(self, compilation: Compilation) -> None:
        """Split the collected laws of complex states.

        Args:
            compilation:
                The compilation state.
        """
        compilation.lowered = [
            (
                spec,
                [
                    item
                    for law in laws
                    for item in self._split_law(law, spec.owner, partial(KernelSpec._state_name, spec))
                ],
                block,
            )
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]

    @staticmethod
    def _split_value(
        targets: list[Expression],
        values: list[Expression],
        description: str,
        state_of: Callable[[Expression], str],
    ) -> tuple[list[Expression], list[Expression]]:
        """Return the real targets and real values of a law, two of each for every complex target."""
        real_targets, real_values = [], []
        for target, value in zip(targets, values, strict=True):
            real, imaginary = _split_complex(value, description)
            if LowerComplexPass._has_complex(target):
                real_targets += [target.real_part(), target.imag_part()]
                real_values += [real, imaginary]
                continue
            if not _is_zero(imaginary):
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
        """Return the laws of one real target component each that a rate, noise, update, or boundary law stands for."""
        description = f'{type(law).__name__} of {owner}'
        targets = components_of(law._get_targets()[0])
        values = (
            None
            if isinstance(law.value, Call)
            else KernelsPass._symbolic_components(law.value, len(targets), description)
        )
        if values is None:
            error_msg = (
                f'The {description} on {state_of(targets[0])} calls a function with complex values or on a complex '
                'state. Functions take and give real values, so pass `real_part(...)` and `imag_part(...)` instead.'
            )
            raise ValidationError(error_msg)
        real_targets, real_values = LowerComplexPass._split_value(targets, values, description, state_of)
        return [
            BasePass._replaced(law, target=target, value=value)
            for target, value in zip(real_targets, real_values, strict=True)
        ]

    @staticmethod
    def _split_equation(equation: EquationLaw, owner: str) -> list[BaseLaw]:
        """Return the real equations of the real and imaginary parts of a complex equation."""
        description = f'Equation of {owner}'
        real_lhs, imaginary_lhs = _split_complex(equation.lhs_, description)
        real_rhs, imaginary_rhs = _split_complex(equation.rhs_, description)
        equations: list[BaseLaw] = [EquationLaw(real_lhs, real_rhs).check().prepare()]
        if not (_is_zero(imaginary_lhs) and _is_zero(imaginary_rhs)):
            equations.append(EquationLaw(imaginary_lhs, imaginary_rhs).check().prepare())
        return equations

    @staticmethod
    def _split_event(event: CrossingLaw, owner: str, state_of: Callable[[Expression], str]) -> CrossingLaw:
        """Return an event with a real guard and a reset of real states."""
        description = f'Event of {owner}'
        guard = _real(_split_complex(event.guard, description), SR(event.guard), description)
        if event.reset is None:
            return BasePass._replaced(event, guard=guard)
        targets = [SR(target) for target in event.reset]
        values = [SR(value) for value in event.reset.values()]
        real_targets, real_values = LowerComplexPass._split_value(targets, values, description, state_of)
        return BasePass._replaced(event, guard=guard, reset=dict(zip(real_targets, real_values, strict=True)))

    @staticmethod
    def _split_jump(jump: JumpLaw, owner: str) -> JumpLaw:
        """Return a jump with a real rate, after checking that it changes real states."""
        description = f'JumpLaw of {owner}'
        if any(LowerComplexPass._has_complex(target) for target in jump.changes):
            error_msg = f'The {description} changes a complex state. Jumps change real states by real increments.'
            raise ValidationError(error_msg)
        return JumpLaw(_real(_split_complex(jump.rate, description), SR(jump.rate), description), jump.changes)

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
        """Return the laws of real states and real values that a law of complex states and values stands for.

        A law with no complex value is returned unchanged. A law with a complex value becomes one law per real component
        of its target, two for every complex component, and an equation with complex sides becomes the equations of
        their parts.

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
        if isinstance(law, _AGENT_LAWS) or not any(
            LowerComplexPass._has_complex(item) for item in BasePass._values_of(law)
        ):
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
