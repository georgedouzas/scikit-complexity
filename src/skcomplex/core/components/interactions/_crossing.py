"""Define the interaction that fires an event when a guard crosses zero."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR

from ...errors import ValidationError
from ...specs import RoleSpec
from ...types import ElementKind
from ..laws import BaseLaw, CrossingLaw
from ..spaces import BaseSpace
from ._interaction import Interaction

DIRECTIONS = (-1, 0, 1)


class CrossingInteraction(Interaction):
    """An interaction that resets states or ends the run when a guard crosses zero.

    The guard is a function of the states and the time whose sign change marks an event, such as the height of a falling
    ball. At the crossing the event sets states of the target to new values or ends the run. A guard that reads the
    members of a group gives each member its own event. Solvers locate the crossing within their tolerance, and the
    trajectory records every crossing among its events, under the label of the interaction and with the member.

    Unlike `JumpInteraction`, which fires at random times, it fires at the exact time a condition on the states becomes
    true. Choose it when a continuous run has discrete events, such as an impact, a threshold, or a stop.

    It appears in many fields. A bouncing ball reverses its velocity at the floor, a neuron fires and resets when its
    potential reaches a threshold, and a thermostat switches the heater at a set temperature.

    Args:
        label:
            The name of the interaction, unique among the parts of its system, such as `'bounce'`.

        target:
            The path of the element or group whose states the guard reads and the event sets.

        guard:
            The function `guard(namespace)` of the expression whose crossing of zero triggers the event.

        reset:
            A mapping from the name of a state of the target to the function `f(namespace)` of its new value, applied
            all at once, or `None` to keep the states.

        direction:
            `1` for crossings from negative to positive, `-1` for the opposite, and `0` for both.

        terminal:
            Whether the crossing ends the run.

    Raises:
        ValidationError:
            The direction is not -1, 0, or 1, or the guard or a new value is not a function.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import CrossingInteraction
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class ClockElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, s):
        ...         return [RateLaw(s.x, 1.0)]
        >>> wrap = CrossingInteraction(
        ...     'wrap', target='clock', guard=lambda s: s.target.x - 1, reset={'x': lambda s: 0}, direction=1
        ... )
        >>> system = CompositeSystem(elements=[ClockElement('clock')], interactions=[wrap])
        >>> trajectory = simulate(system, until=2.2, dt=0.5)
        >>> [round(event.time, 6) for event in trajectory.events]
        [1.0, 2.0]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (CrossingLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('reset',))],
        guard: Callable[[SimpleNamespace], object],
        reset: Mapping[str, Callable[[SimpleNamespace], object]] | None = None,
        direction: int = 0,
        terminal: bool = False,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.guard = guard
        self.reset = reset
        self.direction = direction
        self.terminal = terminal

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an invalid direction, guard, or new value.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The direction is not -1, 0, or 1, or the guard or a new value is not a function.
        """
        super()._prepare(context)
        if self.direction not in DIRECTIONS:
            error_msg = f'The direction of crossing {self.label!r} is {self.direction!r}. Use -1, 0, or 1.'
            raise ValidationError(error_msg)
        if not callable(self.guard) or not all(callable(function) for function in (self.reset or {}).values()):
            error_msg = f'The guard and the new values of crossing {self.label!r} must be functions of the namespace.'
            raise ValidationError(error_msg)

    @staticmethod
    def _get_law_value(value: object) -> object:
        """Return what a function of the namespace returned.

        A function stays as it is, or the result is a symbolic expression.
        """
        return value if BaseLaw._is_function(value) else SR(value)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the crossing law of the guard with the new values of the states.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The crossing law.
        """
        reset = {
            getattr(self._get_member(namespace, 'target'), name): self._get_law_value(f(namespace))
            for name, f in (self.reset or {}).items()
        }
        return [
            CrossingLaw(
                self._get_law_value(self.guard(namespace)),
                reset or None,
                self.direction,
                self.terminal,
                label=self.label,
            ),
        ]
