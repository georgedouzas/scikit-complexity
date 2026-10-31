"""Define maps, which set the next value of a state after one discrete time step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, SupportsFloat

from sage.all__sagemath_symbolics import SR

from ...errors import ValidationError
from ...specs import RoleSpec
from ..base import BasePart
from ..laws import RuleLaw, UpdateLaw
from ..laws.utils import uniform
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

MODES = ('replace', 'increment')
SCHEDULES = ('synchronous', 'sequential', 'random-sequential')
FORMS = ('symbolic', 'agent')


class MapInteraction(Interaction):
    """Map of a state of discrete time, whose next value is a function, or the state plus it.

    In the mode `'replace'` the next value of the state is `f`, and in the mode `'increment'` it is the state plus `f`.
    Under the `'synchronous'` schedule every member of a bound group updates at once, from the states at the start of
    the step. Under the `'sequential'` schedule the members update one after the other, in their order, and under the
    `'random-sequential'` schedule in a new random order every step, each member reading the values the members before
    it set.

    In the `'symbolic'` form, `function(namespace)` returns a symbolic expression, or a vector of them for a vector
    state, where `namespace.target` holds the states and parameters of the target, `namespace.sources` the namespaces
    of the sources, and `namespace.t` the time, and may hold draws such as `normal('noise')`. In the `'agent'` form,
    the target is an agent group and `function(agent, model)` returns the value as a number, so it can read the agent,
    its neighbours, and the history and storage of the model view.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state the map sets.

        state:
            The name of the state of the target.

        function:
            The function `f(namespace)` in the symbolic form, or `f(agent, model)` in the agent form.

        sources:
            The paths of the elements, or members of groups such as `cells[0]`, that the function reads.

        mode:
            `'replace'` or `'increment'`.

        schedule:
            `'synchronous'`, `'sequential'`, or `'random-sequential'`.

        form:
            `'symbolic'` for update laws of the states of the target, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The mode, the schedule, or the form is unknown, or the function is not callable.

    Examples:
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.interactions import MapInteraction
        >>> from skcomplex.core.spaces import DiscreteTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> from typing import Annotated, SupportsFloat
        >>> class ValueElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 0.5):
        ...         super().__init__(label)
        ...         self.x = x
        >>> logistic = MapInteraction(
        ...     'map', target='value', state='x', function=lambda s: 3 * s.target.x * (1 - s.target.x)
        ... )
        >>> system = System(elements=[ValueElement('value')], interactions=[logistic], time=DiscreteTime())
        >>> Simulation(system).check().prepare().run(until=2).trajectory_.values('value.x').tolist()
        [0.5, 0.75, 0.5625]
        >>> agents = MapInteraction('map', target='walkers', state='x', function=lambda a, m: a.x + 1, form='agent')
        >>> type(agents.check().prepare().run(SimpleNamespace()).laws_[0]).__name__
        'RuleLaw'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str,
        function: Callable[..., object],
        sources: Annotated[Sequence[str], RoleSpec(variadic=True)] = (),
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

    def _check(self) -> None:
        """Raise an error for an unknown mode, schedule, or form, or a function that cannot be called.

        Raises:
            ValidationError:
                The mode, the schedule, or the form is unknown, or the function is not callable.
        """
        super()._check()
        for name, value, variants in (
            ('mode', self.mode, MODES),
            ('schedule', self.schedule, SCHEDULES),
            ('form', self.form, FORMS),
        ):
            if value not in variants:
                error_msg = f'The {name} of map {self.label!r} is {value!r}. Use one of {", ".join(variants)}.'
                raise ValidationError(error_msg)
        if not callable(self.function):
            error_msg = f'The function of map {self.label!r} must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)

    def _get_value(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> float:
        """Return the next value of the state of an agent."""
        found = self.function(agent, model)
        if not isinstance(found, SupportsFloat):
            error_msg = f'The function of map {self.label!r} returns {found!r} for an agent. Return a number.'
            raise ValidationError(error_msg)
        value = float(found)
        return float(getattr(agent, self.state)) + value if self.mode == 'increment' else value

    def update(self, model: ModelViewProtocol) -> None:
        """Set the state of every agent of the target to its next value, by the schedule.

        Args:
            model:
                The model view of the interaction.
        """
        agents = list(model.agents('target'))
        if self.schedule == 'synchronous':
            values = [self._get_value(agent, model) for agent in agents]
            for agent, value in zip(agents, values, strict=True):
                setattr(agent, self.state, value)
            return
        order = model.rng.permutation(len(agents)) if self.schedule == 'random-sequential' else range(len(agents))
        for index in order:
            setattr(agents[index], self.state, self._get_value(agents[index], model))

    def _get_order(self) -> object:
        """Return the order of the members of the symbolic form: none, their own order, or a uniform draw."""
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
        if self.form == 'agent':
            return [RuleLaw(self.update, phase=self.phase)]
        value = getattr(namespace.target, self.state)
        mapped = self.function(namespace)
        increment = self.mode == 'increment'
        return [UpdateLaw(value, value + mapped if increment else mapped, increment=increment, order=self._get_order())]
