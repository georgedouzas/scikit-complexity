"""Define sign adjustments, which step a state of discrete time up or down by the sign of a signal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, ClassVar, SupportsFloat

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import RoleSpec, Tags
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, RuleLaw, UpdateLaw
from ..laws.utils import clip, sign
from ..types import AgentViewProtocol, SystemViewProtocol
from ._base import SCHEDULES
from ._interaction import Interaction

MODES = ('multiplicative', 'additive')
FORMS = ('symbolic', 'agent')

type Value = float | Callable[..., object]


class SignAdjustmentInteraction(Interaction):
    """Step of a state of discrete time by the sign of a signal, kept within bounds.

    In the mode `'multiplicative'` the next value of the state `x` is `x (1 + step sign(signal))`, and in the mode
    `'additive'` it is `x + step sign(signal)`, so the state rises while the signal is positive, falls while it is
    negative, and stays while it is zero, by trial and error rather than by the size of the signal. The next value is
    then limited to `[lower, upper]`.

    In the `'symbolic'` form, the signal, the step, and the bounds are numbers or functions of the namespace, where
    `namespace.target` holds the states and parameters of the target. In the `'agent'` form, the target is an agent
    group and they are numbers or functions `f(agent, system)` that return numbers. The agents step at once from the
    states at the start of the rule under the schedule `'synchronous'`, one after the other in their order or in a new
    random order every step under `'sequential'` or `'random-sequential'`, each reading the states the agents before
    it set, or each in its own activation under `'member'`, together with the rules of its element and the other
    member interactions, in the order the system lists them.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state steps.

        state:
            The name of the state of the target.

        signal:
            The function of the signal whose sign sets the direction of the step.

        step:
            The size of the step, relative in the mode `'multiplicative'`, a number or a function.

        mode:
            `'multiplicative'` or `'additive'`.

        lower:
            The lowest next value, a number or a function, or `None` for none.

        upper:
            The highest next value, a number or a function, or `None` for none.

        form:
            `'symbolic'` for an update law of the state of the target, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

        schedule:
            The order of the agents in the agent form, `'synchronous'`, `'sequential'`, `'random-sequential'`, or
            `'member'`.

    Raises:
        ValidationError:
            The mode, the form, or the schedule is unknown, the symbolic form is not synchronous, or the signal is not
            callable, raised by `prepare`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import SignAdjustmentInteraction
        >>> from skcomplex.core.components.spaces import DiscreteTime
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class QuoteElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        >>> search = SignAdjustmentInteraction(
        ...     'search', target='quote', state='x', signal=lambda s: 2 - s.target.x, step=0.5, upper=2.0
        ... )
        >>> system = System(elements=[QuoteElement('quote')], interactions=[search], time=DiscreteTime())
        >>> simulate(system, until=3).values('quote.x').tolist()
        [1.0, 1.5, 2.0, 2.0]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the rules of the agent form or of the updates of the symbolic adjustment."""
        return self._get_variant_tags((RuleLaw,) if self._get_argument('form') == 'agent' else (UpdateLaw,))

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        state: str,
        signal: Callable[..., object],
        step: Value = 0.1,
        mode: str = 'multiplicative',
        lower: Value | None = None,
        upper: Value | None = None,
        form: str = 'symbolic',
        phase: str | None = None,
        schedule: str = 'synchronous',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.signal = signal
        self.step = step
        self.mode = mode
        self.lower = lower
        self.upper = upper
        self.form = form
        self.phase = phase
        self.schedule = schedule

    def _check(self) -> None:
        """Raise an error for an unknown mode, form, or schedule, or a signal that cannot be called.

        Raises:
            ValidationError:
                The mode, the form, or the schedule is unknown, the symbolic form is not synchronous, or the signal is
                not callable.
        """
        super()._check()
        for name, value, variants in (('mode', self.mode, MODES), ('form', self.form, FORMS)):
            if value not in variants:
                error_msg = f'The {name} of {self.label!r} is {value!r}. Use one of {", ".join(variants)}.'
                raise ValidationError(error_msg)
        self._check_variant('schedule', self.schedule, SCHEDULES)
        if self.form == 'symbolic' and self.schedule != 'synchronous':
            error_msg = f'The schedule {self.schedule!r} of {self.label!r} runs agent rules. Use `form="agent"`.'
            raise ValidationError(error_msg)
        if not callable(self.signal):
            error_msg = f'The signal of {self.label!r} must be callable, but it is {self.signal!r}.'
            raise ValidationError(error_msg)

    def _get_stepped(self, value: Expression, step: Expression, direction: Expression) -> Expression:
        """Return the value stepped in a direction by the mode."""
        if self.mode == 'additive':
            return value + step * direction
        return value * (1 + step * direction)

    def _read(self, item: Value | None, *arguments: object) -> object:
        """Return a number, the value of a function at the arguments, or `None`."""
        return item(*arguments) if callable(item) else item

    def _get_number(self, item: Value | None, agent: AgentViewProtocol, system: SystemViewProtocol) -> float | None:
        """Return a number or the value of a function of an agent as a number, or `None`."""
        found = self._read(item, agent, system)
        if found is None:
            return None
        if not isinstance(found, SupportsFloat):
            error_msg = f'A function of {self.label!r} returns {found!r} for an agent. Return a number.'
            raise ValidationError(error_msg)
        return float(found)

    def _get_value(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the next value of the state of an agent."""
        direction = float(np.sign(self._get_number(self.signal, agent, system) or 0.0))
        step = self._get_number(self.step, agent, system) or 0.0
        current = float(getattr(agent, self.state))
        value = current + step * direction if self.mode == 'additive' else current * (1 + step * direction)
        lower, upper = self._get_number(self.lower, agent, system), self._get_number(self.upper, agent, system)
        if upper is not None:
            value = min(upper, value)
        if lower is not None:
            value = max(lower, value)
        return value

    def _set_value(self, agent: AgentViewProtocol, value: float) -> None:
        """Set the state of an agent to its next value."""
        setattr(agent, self.state, value)

    def _adjust_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Step the state of one agent, in its own activation."""
        if agent.alive:
            self._set_value(agent, self._get_value(agent, system))

    def adjust(self, system: SystemViewProtocol) -> None:
        """Step the state of every agent of the target, by the schedule.

        Args:
            system:
                The system view of the interaction.
        """
        self._update_agents(system, self.schedule, self._get_value, self._set_value)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the update of the state to its stepped value, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The update law of the state, or the rule that steps the agents.
        """
        if self.form == 'agent' and self.schedule == 'member':
            return [RuleLaw(self._adjust_member, phase=self.phase, members=self._get_role('target'))]
        if self.form == 'agent':
            return [RuleLaw(self.adjust, phase=self.phase)]
        value = SR(getattr(self._get_member(namespace, 'target'), self.state))
        direction = sign(self._read(self.signal, namespace))
        stepped = self._get_stepped(value, SR(self._read(self.step, namespace)), direction)
        bounded = clip(stepped, self._read(self.lower, namespace), self._read(self.upper, namespace))
        return [UpdateLaw(value, bounded)]
