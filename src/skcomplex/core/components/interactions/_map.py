"""Define maps.

A map sets the next value of a state after one discrete time step.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, SupportsFloat, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ...errors import ValidationError
from ...specs import BaseSpec, RoleSpec, Tags
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, RuleLaw, UpdateLaw
from ..laws.utils import uniform
from ..spaces import BaseSpace
from ..types import AgentViewProtocol, SystemViewProtocol
from ._base import SCHEDULES
from ._interaction import Interaction

MODES = ('replace', 'increment')
FORMS = ('symbolic', 'agent')

type Value = float | npt.NDArray[np.float64]


class MapInteraction(Interaction):
    """Map of a state of discrete time.

    Its next value is a function, or the state plus it. In the mode `'replace'` the next value of the state is `f`, and
    in the mode `'increment'` it is the state plus `f`. Under the `'synchronous'` schedule every member of a bound group
    updates at once, from the states at the start of the step. Under the `'sequential'` schedule the members update one
    after the other, in their order, and under the `'random-sequential'` schedule in a new random order every step, each
    member reading the values the members before it set. Under the `'member'` schedule of the agent form, every member
    updates in its own activation, together with the rules of its element and the other member interactions, in the
    order the system lists them.

    In the `'symbolic'` form, `function(namespace)` returns a symbolic expression, or a vector of them for a vector
    state, where `namespace.target` holds the states and parameters of the target, `namespace.sources` the namespaces
    of the sources, and `namespace.t` the time, and may hold draws such as `normal('noise')`. In the `'agent'` form,
    the target is an agent group and `function(agent, system)` returns the value as a number, or as an array of the
    shape of a vector state, so it can read the agent, its neighbours, and the history and storage of the system view.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state the map sets.

        state:
            The name of the state of the target.

        function:
            The function `f(namespace)` in the symbolic form, or `f(agent, system)` in the agent form.

        sources:
            The paths of the elements, or members of groups such as `cells[0]`, that the function reads.

        mode:
            `'replace'` or `'increment'`.

        schedule:
            `'synchronous'`, `'sequential'`, `'random-sequential'`, or `'member'` for the agent form.

        form:
            `'symbolic'` for update laws of the states of the target, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The mode, the schedule, or the form is unknown, the symbolic form has the schedule `'member'`, the
            function is not callable, or the agent form returns a value that does not match the state, raised by
            `run` of the simulation.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import MapInteraction
        >>> from skcomplex.core.components.spaces import DiscreteTime
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.specs import StateSpec
        >>> from typing import Annotated, SupportsFloat
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class ValueElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 0.5):
        ...         super().__init__(label)
        ...         self.x = x
        >>> logistic = MapInteraction(
        ...     'map', target='value', state='x', function=lambda s: 3 * s.target.x * (1 - s.target.x)
        ... )
        >>> system = System(elements=[ValueElement('value')], interactions=[logistic], time=DiscreteTime())
        >>> simulate(system, until=2).values('value.x').tolist()
        [0.5, 0.75, 0.5625]
        >>> agents = MapInteraction('map', target='walkers', state='x', function=lambda a, m: a.x + 1, form='agent')
        >>> type(agents.prepare().run(SimpleNamespace()).laws_[0]).__name__
        'RuleLaw'
        >>> from skcomplex.core.components.elements import GroupElement
        >>> class WeightsElement(Element):
        ...     def __init__(self, label, visual=None, *, w: Annotated[object, StateSpec(vector=True)] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.w = w
        >>> group = GroupElement('weights', element=WeightsElement, size=1, values={'w': [1.0, 2.0]})
        >>> doubling = MapInteraction(
        ...     'double', target='weights', state='w', function=lambda a, m: [2 * x for x in a.w], form='agent'
        ... )
        >>> trajectory = simulate(System(elements=[group], interactions=[doubling]), until=2)
        >>> trajectory.agents('weights', 2).values['w'].tolist()
        [[4.0, 8.0]]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the agent-form rules or of the symbolic map updates."""
        return self._get_variant_tags((RuleLaw,) if self._get_argument('form') == 'agent' else (UpdateLaw,))

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        state: str,
        function: Callable[..., object],
        sources: Annotated[Sequence[str], RoleSpec(variadic=True, elements=(ElementKind.ANY,))] = (),
        mode: str = 'replace',
        schedule: str = 'synchronous',
        form: str = 'symbolic',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.function = function
        self.sources = sources
        self.mode = mode
        self.schedule = schedule
        self.form = form
        self.phase = phase

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown mode, schedule, or form, or a bad function.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The mode, the schedule, or the form is unknown, the symbolic form has the schedule `'member'`, or the
                function is not callable.
        """
        super()._prepare(context)
        self._check_variant('mode', self.mode, MODES, (BaseSpec,))
        for name, value, variants in (
            ('schedule', self.schedule, SCHEDULES),
            ('form', self.form, FORMS),
        ):
            if value not in variants:
                error_msg = f'The {name} of map {self.label!r} is {value!r}. Use one of {", ".join(variants)}.'
                raise ValidationError(error_msg)
        if self.schedule == 'member' and self.form != 'agent':
            error_msg = (
                f'The map {self.label!r} updates every member in its activation, which only the agent form does.'
            )
            raise ValidationError(error_msg)
        if not callable(self.function):
            error_msg = f'The function of map {self.label!r} must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)

    def _apply_mode(self, current: Value, value: Value) -> Value:
        """Return the next value of a state by the mode.

        It comes from its current value and the value of the function.
        """
        if self.mode == 'increment':
            return current + value
        if self.mode == 'replace':
            return value
        raise self._get_variant_error(self.mode, 'mode', '_apply_mode')

    def _get_vector(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> Value:
        """Return the next value of a vector state of an agent."""
        current = np.asarray(getattr(agent, self.state), dtype=np.float64)
        value = np.asarray(self.function(agent, system), dtype=np.float64)
        if value.shape != current.shape:
            error_msg = (
                f'The function of map {self.label!r} returns the shape {value.shape} for the state {self.state!r} of '
                f'the shape {current.shape}. Return an array of the shape of the state.'
            )
            raise ValidationError(error_msg)
        return self._apply_mode(current, value)

    def _get_value(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> Value:
        """Return the next value of the state of an agent.

        It is a number or the array of a vector state.
        """
        if np.ndim(getattr(agent, self.state)) > 0:
            return self._get_vector(agent, system)
        found = self.function(agent, system)
        if not isinstance(found, SupportsFloat):
            error_msg = f'The function of map {self.label!r} returns {found!r} for an agent. Return a number.'
            raise ValidationError(error_msg)
        value = float(found)
        return self._apply_mode(float(getattr(agent, self.state)), value)

    def _set_value(self, agent: AgentViewProtocol, value: Value) -> None:
        """Set the state of an agent to its next value."""
        setattr(agent, self.state, value)

    def _update_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Set the state of one member to its next value."""
        if agent.alive:
            self._set_value(agent, self._get_value(agent, system))

    def update(self, system: SystemViewProtocol) -> None:
        """Set the state of every agent of the target to its next value, by the schedule.

        Args:
            system:
                The system view of the interaction.
        """
        self._update_agents(system, self.schedule, self._get_value, self._set_value)

    def _get_order(self) -> object:
        """Return the order of the members of the symbolic form.

        It is none, their own order, or a uniform draw.
        """
        if self.schedule == 'synchronous':
            return None
        if self.schedule == 'sequential':
            return SR(0)
        return uniform(re.sub(r'\W', '_', f'order_{self.label}'))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the update of the state to its next value, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and `namespace.sources`.

        Returns:
            laws:
                The update law of the state, or the rule that runs the map of the agents.
        """
        if self.form == 'agent' and self.schedule == 'member':
            return [RuleLaw(self._update_member, phase=self.phase, members=self._get_role('target'))]
        if self.form == 'agent':
            return [RuleLaw(self.update, phase=self.phase)]
        value = getattr(self._get_member(namespace, 'target'), self.state)
        mapped = cast(Value, self.function(namespace))
        next_value = self._apply_mode(value, mapped)
        return [UpdateLaw(value, next_value, increment=self.mode == 'increment', order=self._get_order())]
