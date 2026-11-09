"""Define pulses, resets of states at given or periodic times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from .....core.components.interactions import CrossingInteraction
from .....core.components.laws import BaseLaw, CrossingLaw
from .....core.components.laws.utils import periodic_guard
from .....core.components.spaces import BaseSpace
from .....core.errors import ValidationError


class PulseInteraction(CrossingInteraction):
    """Pulses that set states of the target to new values at given or periodic times.

    Engineering names them pulses or impulses, as the kicks of a sampled controller or of a pulsed circuit. The pulses
    fall at every time of `times` and, with a `period`, at `start + j * period` for `j` from 0, `count` times or without
    end. A pulse at the initial time of a run does not fire. The new values are functions of the namespace, which holds
    `namespace.target`, the states and parameters of the target, and the time `namespace.t`. A pulse sets the states of
    every member of a group at once, and the trajectory records every pulse among its events, under the label of the
    interaction.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose states the pulses set.

        reset:
            A mapping from the name of a state of the target to the function `f(namespace)` of its new value.

        times:
            The times of the pulses.

        period:
            The period of the periodic pulses, or `None` for none.

        start:
            The time of the first periodic pulse.

        count:
            The number of periodic pulses, or `None` for pulses without end.

    Raises:
        ValidationError:
            The pulses set no state, a new value is not a function, the pulses have no time, the period is not
            positive, or the count is not a positive integer.

    Examples:
        >>> from skcomplex.systems.engineering.common.signals import PulseInteraction
        >>> reset = {'c': lambda s: s.target.c + 1.0}
        >>> dose = PulseInteraction('dose', target='body', reset=reset, period=8.0, count=3)
        >>> dose.start
        0.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (CrossingLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: str,
        reset: Mapping[str, Callable[[SimpleNamespace], object]],
        times: Sequence[float] = (),
        period: float | None = None,
        start: float = 0.0,
        count: int | None = None,
    ) -> None:
        super().__init__(label, target=target, guard=self._get_guard, reset=reset)
        self.times = times
        self.period = period
        self.start = start
        self.count = count

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for pulses without states or times, or a bad period or count.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The pulses set no state, a new value is not a function, the pulses have no time, the period is not
                positive, or the count is not a positive integer.
        """
        if not self.reset or not all(callable(function) for function in self.reset.values()):
            error_msg = f'The pulses {self.label!r} set {self.reset!r}. Map state names to functions of the namespace.'
            raise ValidationError(error_msg)
        if not self.times and self.period is None:
            error_msg = f'The pulses {self.label!r} have no time. Give `times`, or a `period`.'
            raise ValidationError(error_msg)
        if self.period is not None and not self.period > 0:
            error_msg = f'The period of the pulses {self.label!r} is {self.period!r}. Give a positive number.'
            raise ValidationError(error_msg)
        if self.count is not None and (
            isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1
        ):
            error_msg = f'The count of the pulses {self.label!r} is {self.count!r}. Give a positive integer, or None.'
            raise ValidationError(error_msg)
        super()._prepare(context)

    def _get_guard(self, namespace: SimpleNamespace) -> Expression:
        """Return the guard that crosses zero at every periodic pulse."""
        return periodic_guard(namespace.t, self.period, self.start, self.count)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the crossing laws of time at the pulses.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and the time `namespace.t`.

        Returns:
            laws:
                One crossing law per given time and one for the periodic pulses.
        """
        reset = {
            getattr(self._get_member(namespace, 'target'), name): SR(f(namespace)) for name, f in self.reset.items()
        }
        laws: list[BaseLaw] = [CrossingLaw(namespace.t - time, reset, 1, label=self.label) for time in self.times]
        if self.period is not None:
            laws.extend(super()._get_laws(namespace))
        return laws
