"""Define jumps, random events that change states by signed amounts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, JumpLaw, UpdateLaw
from ..laws.utils import uniform, where
from ._interaction import Interaction

CLOCKS = ('rate', 'probability', 'waiting')


class JumpInteraction(Interaction):
    """Random event that changes states of the target by signed amounts, at a rate, by a probability, or after a wait.

    With a `rate`, the event fires at the times of a Poisson process of that propensity in continuous time. With a
    `probability`, it fires in a step of discrete time with that probability. With a `waiting` time, usually a draw from
    a distribution with nonnegative support, it fires after a time drawn anew at every firing, as in renewal processes,
    and every member of a bound group fires on its own clock. When the event fires it adds the `changes` to the states
    and, with a waiting time, sets the states of `reset` to their new values. Every function reads `namespace.target`,
    the states and parameters of the target, and the time `namespace.t`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose states the event changes.

        changes:
            A mapping from the name of a state of the target to the signed amount the event adds to it.

        rate:
            The function `rate(namespace)` of the propensity of the event, or `None`.

        probability:
            The function `probability(namespace)` of the probability that the event fires in a step, or `None`.

        waiting:
            The function `waiting(namespace)` of the time until the next firing, or `None`.

        reset:
            With a waiting time, a mapping from the name of a state of the target to the function `f(namespace)` of its
            new value, one value per component of a vector state, or `None`.

    Raises:
        ValidationError:
            The event has not exactly one clock, changes nothing, or resets states without a waiting time, or a
            function is not callable.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.interactions import JumpInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> arrivals = JumpInteraction('arrivals', target='queue', changes={'n': 1}, rate=lambda s: 2.0)
        >>> system = System(elements=[CompartmentElement('queue')], interactions=[arrivals])
        >>> simulation = Simulation(system, dt=1.0, random_state=0).check().prepare()
        >>> counts = simulation.run(until=5).trajectory_.values('queue.n')
        >>> bool(all(count == round(count) for count in counts))
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        changes: Mapping[str, float] | None = None,
        rate: Callable[[SimpleNamespace], object] | None = None,
        probability: Callable[[SimpleNamespace], object] | None = None,
        waiting: Callable[[SimpleNamespace], object] | None = None,
        reset: Mapping[str, Callable[[SimpleNamespace], object]] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.changes = changes
        self.rate = rate
        self.probability = probability
        self.waiting = waiting
        self.reset = reset

    def _get_clocks(self) -> list[str]:
        """Return the names of the clocks the event is given."""
        return [clock for clock in CLOCKS if getattr(self, clock) is not None]

    def _check_jump(self) -> None:
        """Raise an error for an event without exactly one clock, without changes, or with a reset of no waiting time.

        Raises:
            ValidationError:
                The event has not exactly one clock, changes nothing, or resets states without a waiting time, or a
                function is not callable.
        """
        clocks = self._get_clocks()
        if len(clocks) != 1:
            error_msg = f'The jump {self.label!r} has the clocks {clocks}. Give exactly one of {", ".join(CLOCKS)}.'
            raise ValidationError(error_msg)
        if not callable(getattr(self, clocks[0])):
            error_msg = f'The {clocks[0]} of the jump {self.label!r} must be a function of the namespace.'
            raise ValidationError(error_msg)
        if self.reset is not None and self.waiting is None:
            error_msg = f'The jump {self.label!r} resets states, which only a waiting time does. Give `waiting`.'
            raise ValidationError(error_msg)
        if not self.changes and not self.reset:
            error_msg = f'The jump {self.label!r} changes no state. Give `changes`, or `reset` with a waiting time.'
            raise ValidationError(error_msg)
        if not all(callable(function) for function in (self.reset or {}).values()):
            error_msg = f'The reset of the jump {self.label!r} maps states to {self.reset!r}. Give functions.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an event without exactly one clock, without changes, or with a reset of no waiting time.

        Raises:
            ValidationError:
                The event has not exactly one clock, changes nothing, or resets states without a waiting time, or a
                function is not callable.
        """
        super()._check()
        self._check_jump()

    def _get_changes(self, namespace: SimpleNamespace) -> dict[Expression, float]:
        """Return the amount the event adds to every state it changes."""
        return {getattr(namespace.target, name): amount for name, amount in (self.changes or {}).items()}

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the propensity of the event."""
        return SR(self.rate(namespace)) if self.rate is not None else SR(0)

    def _get_steps(self, namespace: SimpleNamespace, changes: dict[Expression, float]) -> list[BaseLaw]:
        """Return the updates of the states by their amounts in a step, when one draw falls below the probability."""
        probability = SR(self.probability(namespace)) if self.probability is not None else SR(0)
        name = re.sub(r'\W', '_', f'jump_{self.label}')
        fired = where(uniform(name) < probability, 1, 0)
        return [UpdateLaw(state, state + amount * fired) for state, amount in changes.items()]

    def _get_reset(self, state: object, value: object) -> dict[Expression, Expression]:
        """Return the new value of every component of a state, one per component of a vector state.

        Raises:
            ValidationError:
                The new value of a vector state has not one component per component of the state.
        """
        if not isinstance(state, FreeModuleElement):
            return {SR(state): SR(value)}
        components = list(value) if isinstance(value, FreeModuleElement | list | tuple) else [value] * len(state)
        if len(components) != len(state):
            error_msg = f'The jump {self.label!r} resets {len(state)} components with {len(components)} values.'
            raise ValidationError(error_msg)
        return {SR(item): SR(component) for item, component in zip(state, components, strict=True)}

    def _get_renewal(self, namespace: SimpleNamespace, changes: dict[Expression, float]) -> JumpLaw:
        """Return the renewal of the event, which adds the changes and sets the reset states after every wait."""
        reset: dict[Expression, object] = {state: state + amount for state, amount in changes.items()}
        for name, function in (self.reset or {}).items():
            reset.update(self._get_reset(getattr(namespace.target, name), function(namespace)))
        waiting = SR(self.waiting(namespace)) if self.waiting is not None else SR(0)
        return JumpLaw(waiting=waiting, reset=reset, label=self.label)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jump law of the event at a rate or after a wait, or the updates of a step with a probability.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The laws of the event.
        """
        changes = self._get_changes(namespace)
        if self.waiting is not None:
            return [self._get_renewal(namespace, changes)]
        if self.probability is not None:
            return self._get_steps(namespace, changes)
        return [JumpLaw(self._get_rate(namespace), changes)]
