"""Define regrowth, which restores the resource of every member of a group towards its capacity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RuleLaw
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

FORMS = ('delay', 'linear', 'logistic')


class RegrowthInteraction(Interaction):
    """Regrowth of the resource of every member of a group towards its capacity, every step of discrete time.

    The resource `r` of a member with the capacity `K` regrows by one of three forms:

    - `'delay'` counts down the clock of a member whose resource is below its capacity by one every step, and when the
      clock reaches zero it restores the resource to the capacity and sets the clock to the `delay`.
    - `'linear'` adds the `rate` to the resource, `r' = min(K, r + rate)`.
    - `'logistic'` adds logistic growth to the resource, `r' = min(K, r + rate r (1 - r / K))`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        resources:
            The path of the group whose members hold a resource.

        form:
            The form of the regrowth, `'delay'`, `'linear'`, or `'logistic'`.

        resource:
            The name of the state that holds the resource of a member.

        capacity:
            The name of the state that holds the capacity of a member.

        clock:
            The name of the state that holds the steps left until the resource of a member is restored, which the form
            `'delay'` reads.

        rate:
            The growth per step of the forms `'linear'` and `'logistic'`.

        delay:
            The number of steps the form `'delay'` takes to restore a resource.

        phase:
            The phase of the step the regrowth runs in, or `None` for the default phase.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element, GroupElement
        >>> from skcomplex.core.interactions import RegrowthInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> class StockElement(Element):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         visual=None,
        ...         *,
        ...         level: Annotated[float, StateSpec()] = 0.0,
        ...         top: Annotated[float, StateSpec()] = 3.0,
        ...     ):
        ...         super().__init__(label, visual=visual)
        ...         self.level = level
        ...         self.top = top
        >>> cells = GroupElement('cells', element=StockElement, size=2, values={'level': [0.0, 2.5]})
        >>> regrowth = RegrowthInteraction(
        ...     'regrowth', resources='cells', form='linear', resource='level', capacity='top', rate=1.0
        ... )
        >>> system = System(elements=[cells], interactions=[regrowth])
        >>> trajectory = Simulation(system, random_state=0).check().prepare().run(until=2).trajectory_
        >>> trajectory.agents('cells', 2).values['level'].tolist()
        [2.0, 3.0]
    """

    def __init__(
        self,
        label: str,
        *,
        resources: Annotated[str, RoleSpec()],
        form: str = 'delay',
        resource: str = 'resource',
        capacity: str = 'capacity',
        clock: str = 'clock',
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.resources = resources
        self.form = form
        self.resource = resource
        self.capacity = capacity
        self.clock = clock
        self.rate = rate
        self.delay = delay
        self.phase = phase

    def _check(self) -> None:
        """Raise an error for an unknown form, or a resource, capacity, or clock that is not the name of a state.

        Raises:
            ValidationError:
                The form is unknown, or a state name is not a string.
        """
        super()._check()
        if self.form not in FORMS:
            error_msg = f'Unknown form {self.form!r} of {self.label!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        for name in ('resource', 'capacity', 'clock'):
            value = getattr(self, name)
            if not isinstance(value, str):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give the name of a state.'
                raise ValidationError(error_msg)

    def _count_down(self, member: AgentViewProtocol, capacity: float, model: ModelViewProtocol) -> None:
        """Count down the clock of a member below its capacity, and restore its resource when the clock ends."""
        if getattr(member, self.resource) >= capacity:
            return
        left = getattr(member, self.clock) - 1
        setattr(member, self.clock, left)
        if left <= 0:
            setattr(member, self.resource, capacity)
            setattr(member, self.clock, model.parameter('delay'))

    def _grow(self, member: AgentViewProtocol, capacity: float, rate: float) -> None:
        """Grow the resource of a member by the linear or the logistic form, up to its capacity."""
        level = getattr(member, self.resource)
        growth = rate if self.form == 'linear' else rate * level * (1 - level / capacity)
        setattr(member, self.resource, min(capacity, level + growth))

    def _regrow(self, model: ModelViewProtocol) -> None:
        """Regrow the resource of every member of the group by one step."""
        rate = model.parameter('rate')
        for member in model.agents('resources'):
            capacity = getattr(member, self.capacity)
            if self.form == 'delay':
                self._count_down(member, capacity, model)
            else:
                self._grow(member, capacity, rate)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the regrowth.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule that regrows the resources every step.
        """
        return [RuleLaw(self._regrow, phase=self.phase)]
