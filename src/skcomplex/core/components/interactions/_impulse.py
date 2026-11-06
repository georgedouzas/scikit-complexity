"""Define impulses, resets of states at given or periodic times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, Expression, pi, sin

from ...errors import ValidationError
from ...specs import RoleSpec
from ...types import ElementKind
from ..laws import BaseLaw, CrossingLaw
from ..laws.utils import where
from ._interaction import Interaction


class ImpulseInteraction(Interaction):
    """Impulses that set states of the target to new values at given times, or at periodic times.

    The impulses fall at every time of `times` and, with a `period`, at `start + j * period` for `j` from 0, `count`
    times or without end. An impulse at the initial time of a run does not fire. The new values are functions of the
    namespace, which holds `namespace.target`, the states and parameters of the target, and the time `namespace.t`. An
    impulse sets the states of every member of a group at once, and the trajectory records every impulse among its
    events, under the label of the interaction.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose states the impulses set.

        reset:
            A mapping from the name of a state of the target to the function `f(namespace)` of its new value.

        times:
            The times of the impulses.

        period:
            The period of the periodic impulses, or `None` for none.

        start:
            The time of the first periodic impulse.

        count:
            The number of periodic impulses, or `None` for impulses without end.

    Raises:
        ValidationError:
            The impulses set no state, a new value is not a function, the impulses have no time, the period is not
            positive, or the count is not a positive integer.

    Examples:
        >>> from skcomplex.core.components.interactions import ImpulseInteraction
        >>> reset = {'c': lambda s: s.target.c + 1.0}
        >>> dose = ImpulseInteraction('dose', target='body', reset=reset, period=8.0, count=3)
        >>> dose.start
        0.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (CrossingLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('reset',))],
        reset: Mapping[str, Callable[[SimpleNamespace], object]],
        times: Sequence[float] = (),
        period: float | None = None,
        start: float = 0.0,
        count: int | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.reset = reset
        self.times = times
        self.period = period
        self.start = start
        self.count = count

    def _check(self) -> None:
        """Raise an error for impulses without states or times, or with a period or a count they cannot take.

        Raises:
            ValidationError:
                The impulses set no state, a new value is not a function, the impulses have no time, the period is not
                positive, or the count is not a positive integer.
        """
        super()._check()
        if not self.reset or not all(callable(function) for function in self.reset.values()):
            error_msg = (
                f'The impulses {self.label!r} set {self.reset!r}. Map state names to functions of the namespace.'
            )
            raise ValidationError(error_msg)
        if not self.times and self.period is None:
            error_msg = f'The impulses {self.label!r} have no time. Give `times`, or a `period`.'
            raise ValidationError(error_msg)
        if self.period is not None and not self.period > 0:
            error_msg = f'The period of the impulses {self.label!r} is {self.period!r}. Give a positive number.'
            raise ValidationError(error_msg)
        if self.count is not None and (
            isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1
        ):
            error_msg = f'The count of the impulses {self.label!r} is {self.count!r}. Give a positive integer, or None.'
            raise ValidationError(error_msg)

    def _get_periodic_guard(self, time: Expression, period: float) -> Expression:
        """Return a guard that crosses zero at every periodic impulse and stays away from zero outside them."""
        guard = where(time < self.start - period / 2, -1, sin(pi * (time - self.start) / period))
        if self.count is None:
            return guard
        last = self.count - SR(1) / 2
        return where(time > self.start + last * period, sin(pi * last), guard)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the crossing laws of time at the impulses.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and the time `namespace.t`.

        Returns:
            laws:
                One crossing law per given time and one for the periodic impulses.
        """
        reset = {
            getattr(self._get_member(namespace, 'target'), name): SR(f(namespace)) for name, f in self.reset.items()
        }
        laws: list[BaseLaw] = [CrossingLaw(namespace.t - time, reset, 1, label=self.label) for time in self.times]
        if self.period is not None:
            laws.append(CrossingLaw(self._get_periodic_guard(namespace.t, self.period), reset, 0, label=self.label))
        return laws
