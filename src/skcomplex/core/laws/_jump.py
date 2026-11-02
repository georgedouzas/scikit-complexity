"""Define jump laws, random events that change states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import CallSpec, RequirementSpec
from ..spaces import ContinuousTime
from ._base import BaseLaw


class JumpLaw(BaseLaw):
    """Random event that changes states, on an exponential clock of a given rate or on a clock of drawn waiting times.

    With a `rate`, the event fires at the times of a Poisson process of that propensity and changes states by fixed
    increments, as in continuous-time Markov chains. With a `waiting` time instead, it fires after a waiting time drawn
    anew at every firing, and applies its `reset`, as in renewal processes such as the jumps of a random walk. Every
    member of a group then fires on its own clock, and the trajectory records every firing and counts them in
    `<label>.renewals`. Jumps combine with rate and noise laws, which evolve the states between firings.

    Args:
        rate:
            The propensity of the event, a symbolic expression, a `Call`, or a callable `f(t, values)`, or `None` with a
            waiting time.

        changes:
            A mapping from the symbol of a scalar state to its increment when the event fires. In a law of a group, a
            target may also be a member offset read `shift_member(namespace.n, k)`.

        waiting:
            The waiting time until the next firing, usually a draw from a distribution with nonnegative support, such
            as `draw('wait', 'pareto', b=alpha)`, or `None` for the exponential clock of `rate`.

        reset:
            With a waiting time, a mapping from state symbol to its new value, which may hold draws, or `None`.

        label:
            With a waiting time, the name of the firings in the trajectory, or the empty string for the path of the
            element.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import JumpLaw
        >>> from skcomplex.core.laws.utils import draw
        >>> s, i = var('s'), var('i')
        >>> JumpLaw(0.1 * s * i, {s: -1, i: 1}).check().prepare().targets_
        (s, i)
        >>> n = var('n')
        >>> JumpLaw(waiting=draw('wait', 'expon'), reset={n: 0}).check().prepare().targets_
        (n,)
    """

    def __init__(
        self,
        rate: object = None,
        changes: Mapping[Expression, object] | None = None,
        waiting: object = None,
        reset: Mapping[Expression, object] | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.rate = rate
        self.changes = changes
        self.waiting = waiting
        self.reset = reset

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the jump changes: the states of its increments, or of its reset."""
        return tuple(self.changes or {}) if self.waiting is None else tuple(self.reset or {})

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _check_clock(self) -> None:
        """Raise an error for a jump with both or neither clock, or with an option of the clock it does not use."""
        if (self.rate is None) == (self.waiting is None):
            error_msg = 'A jump fires at a rate or after a waiting time. Give exactly one of them.'
            raise ValidationError(error_msg)
        if self.waiting is not None and self.changes:
            given, fix = 'changes', 'Give the new values of the states as `reset`.'
        elif self.waiting is None and (self.reset is not None or self.label):
            given = 'a reset' if self.reset is not None else 'a label'
            fix = 'Give the increments of the states as `changes`, or a waiting time instead of the rate.'
        else:
            return
        clock = 'a waiting-time clock' if self.waiting is not None else 'an exponential clock'
        error_msg = (
            f'A jump on {clock} takes {given}, but changes belong to the exponential clock of a rate, and resets and '
            f'labels to the waiting-time clock. {fix}'
        )
        raise ValidationError(error_msg)

    def _check_reset(self) -> None:
        """Raise an error for a function in the reset of a renewal, whose values lower with its symbolic draws."""
        if isinstance(self.waiting, CallSpec):
            error_msg = 'The waiting time of a jump takes a function `f(rng, size=, t=, **values)`, not a `CallSpec`.'
            raise ValidationError(error_msg)
        if any(self._is_function(value) for value in (self.reset or {}).values()):
            error_msg = (
                'The reset of a jump on a waiting-time clock takes symbolic values, because they share draws with the '
                'waiting time. Give symbols, or a function as the `waiting` time.'
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
        """Raise an error for a jump with an invalid clock, a rate that is a NumPy array, or a symbolic increment."""
        self._check_clock()
        self._check_reset()
        self._check_array('rate')
        self._check_array('waiting')
        self._check_changes()

    def _run(self, inputs: object) -> None:
        """Set `value_` to the increments of the jump, or its reset at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        changes = dict(self.changes or {}) if self.waiting is None else dict(self.reset or {})
        self.value_ = self._evaluate(changes, inputs)
