"""Define the law of a random event that changes states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ..spaces._continuous_time import _ContinuousTime
from ._base import BaseLaw
from ._call_spec import CallSpec
from ._requirement_spec import RequirementSpec


@dataclass(frozen=True)
class JumpLaw(BaseLaw):
    """A law of a random event that changes values by whole amounts.

    It fires at random times and changes values at once, such as `JumpLaw(s.beta * s.S * s.I, {s.S: -1, s.I: +1})`
    inside an interaction of an epidemic, where one person at a time moves from the susceptible `S` to the infected `I`
    at the rate `beta S I`. With `rate`, the event fires at that rate, with exponential times between firings, and adds
    `changes` to the values. With `waiting`, it fires after a waiting time drawn anew at every firing, sets the values
    of `reset`, and the trajectory counts the firings of each member in `<label>.renewals`. Rate and noise laws move
    the values between firings. Molecules react one at a time in a cell, people are born and die in a small
    population, customers arrive at a queue, and a walker jumps between sites.

    Read more in the user guide, under Laws.

    Args:
        rate:
            How often the event fires per unit of time, a symbolic expression, a `CallSpec`, or a function
            `f(t, values)`, or `None` with a waiting time.

        changes:
            A mapping from the symbol of a scalar value to the number it adds when the event fires. In a law of a
            group, a key may also be the value of the member at an offset, `shift_member(s.n, k)`.

        waiting:
            The time until the next firing, usually a draw from a distribution of positive numbers, such as
            `draw('wait', 'pareto', b=alpha)`, or `None` for a jump at a rate.

        reset:
            With a waiting time, a mapping from the symbol of a value to its new value, which may hold draws, or
            `None`.

        label:
            The name of the event, such as `'infection'`, or the empty string for a law without one. With a waiting
            time, it names the firings in the trajectory, and the empty string names them by the path of the element.

    See Also:
        RateLaw: Changes a value smoothly at a rate.
        EventLaw: Sets values when an expression crosses zero.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import JumpLaw
        >>> S, I, beta = var('S I beta')
        >>> JumpLaw(beta * S * I, {S: -1, I: 1}, label='infection').changes
        {S: -1, I: 1}
    """

    law_tag: ClassVar[str] = 'jump'

    rate: object = None
    changes: Mapping[Expression, object] | None = None
    waiting: object = None
    reset: Mapping[Expression, object] | None = None
    _: KW_ONLY
    label: str = ''

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the jump changes.

        They are the states of its increments, or of its reset.
        """
        return tuple(self.changes or {}) if self.waiting is None else tuple(self.reset or {})

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(_ContinuousTime,))

    def _check_clock(self) -> None:
        """Raise an error for a jump with both or neither of a rate and a waiting time.

        It also raises it for `changes` after a waiting time or a `reset` at a rate.
        """
        if (self.rate is None) == (self.waiting is None):
            error_msg = 'A jump fires at a rate or after a waiting time. Give exactly one of them.'
            raise ValidationError(error_msg)
        if self.waiting is not None and self.changes:
            given, fix = 'after a waiting time takes `changes`', 'Give the new values as `reset`.'
        elif self.waiting is None and self.reset is not None:
            given = 'at a rate takes a `reset`'
            fix = 'Give the numbers it adds as `changes`, or a waiting time in place of the rate.'
        else:
            return
        error_msg = (
            f'A jump that fires {given}, but a jump at a rate adds `changes` and a jump after a waiting time sets '
            f'`reset`. {fix}'
        )
        raise ValidationError(error_msg)

    def _check_reset(self) -> None:
        """Raise an error for a `CallSpec` as the waiting time or a function in the reset."""
        if isinstance(self.waiting, CallSpec):
            error_msg = 'The waiting time of a jump takes a function `f(rng, size=, t=, **values)`, not a `CallSpec`.'
            raise ValidationError(error_msg)
        if any(self._is_function(value) for value in (self.reset or {}).values()):
            error_msg = (
                'The reset of a jump after a waiting time takes symbolic values, because they may read the draws of '
                'the waiting time. Give symbolic values, or a function as the `waiting` time.'
            )
            raise ValidationError(error_msg)

    def _check_changes(self) -> None:
        """Raise an error for an increment that holds a symbol."""
        for state, increment in (self.changes or {}).items():
            if not self._is_function(increment) and not SR(increment).is_numeric():
                error_msg = (
                    f'The jump of {state} has the increment {increment}, which holds a symbol. Give the increment as a '
                    'number or as a function `f(t, values)` of time and the values of the states and parameters.'
                )
                raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for bad timing, a NumPy array, or a symbolic increment."""
        super()._check()
        self._check_clock()
        self._check_reset()
        self._check_array('rate')
        self._check_array('waiting')
        self._check_changes()

    def _get_value(self, inputs: object) -> object:
        """Return the increments or the reset of the jump at given symbol values.

        The symbols are those it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            changes:
                The increment or the new value of every state the jump names.
        """
        changes = dict(self.changes or {}) if self.waiting is None else dict(self.reset or {})
        return self._evaluate(changes, inputs)
