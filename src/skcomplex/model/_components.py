"""Define the elements and interactions that systems are built from."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, dataclass_transform

from sklearn.base import BaseEstimator

from ..core import Domain, VisualSpec, parameter, role, state
from ._laws import Law
from ._symbols import Symbols


@dataclass_transform(kw_only_default=True, field_specifiers=(dataclasses.field, state, parameter, role))
@dataclasses.dataclass(eq=False, repr=False)
class Component(BaseEstimator):
    """Base class of elements and interactions.

    Every subclass becomes a dataclass whose fields after `label` are keyword-only, so the constructor stores every
    argument unmodified under its own name.

    Args:
        label:
            The label of the component, unique among the children of its system.
    """

    label: str

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Turn the subclass into a keyword-only dataclass."""
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(kw_only=True, eq=False, repr=False)(cls)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the laws of the component.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class Element(Component):
    """Base class of entities that carry state, such as particles, populations, agents, or firms.

    Subclasses declare their states with `state` and their parameters with `parameter`, and can override `laws` to
    return their intrinsic dynamics. The class variable `visual` names how the state is drawn.

    Args:
        label:
            The label of the element, unique among the children of its system.

    Examples:
        >>> from skcomplex.core import Domain, state
        >>> from skcomplex.model import Element
        >>> class Population(Element):
        ...     n: float = state(1.0, domain=Domain.NONNEGATIVE)
        >>> Population('rabbits', n=40)
        Population(label='rabbits', n=40)
    """

    visual: ClassVar[VisualSpec] = VisualSpec()

    @classmethod
    def group(
        cls,
        label: str,
        size: int,
        *,
        visual: VisualSpec | None = None,
        **values: object,
    ) -> 'Group':
        """Return a group of identical members of this element type.

        Args:
            label:
                The label of the group.

            size:
                The number of members.

            visual:
                The visual of the group, or `None` to use the visual of the element type.

            **values:
                Per-member arrays whose first dimension is `size`, or values shared by every member.

        Returns:
            group:
                The group.

        Examples:
            >>> from skcomplex.models.ecology import Population
            >>> Population.group('herds', size=3, n=[10, 20, 30])
            Group('herds', element=Population, size=3)
        """
        return Group(label, element=cls, size=size, values=values, visual=visual)


class AgentType(Element):
    """Base class of agents, whose behaviour is a list of rules and whose groups change size during a run.

    States are attributes of every agent and parameters are shared by every agent of a group. `laws` returns `Rule`
    laws whose functions take the agent and the model. Agent groups are created with `group` like any element group.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, cell coordinates, a network node, or a position, or one location per agent of a
            group, or `None` for random places.

    Examples:
        >>> from skcomplex.core import state
        >>> from skcomplex.model import AgentType, Rule
        >>> class Walker(AgentType):
        ...     steps: float = state(0.0)
        ...
        ...     def laws(self, s):
        ...         return [Rule(self.walk)]
        ...
        ...     def walk(self, agent, model):
        ...         agent.steps += 1
        >>> Walker.group('walkers', 5)
        Group('walkers', element=Walker, size=5)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents')
    location: object = None


class Aggregate(Element):
    """Summary of a group, the count of its members or a reduction of one of its states, held in the state `value`.

    In a co-simulation the scheduler computes the value at every synchronisation point, so partitions that cannot
    see a group's members, such as differential equations next to agents, read the group through the aggregate.

    Args:
        label:
            The label of the aggregate.

        group:
            The path of the agent group or element group, relative to the system that holds the aggregate.

        reduce:
            `'count'`, `'sum'`, `'mean'`, or a function of the array of the variable's values.

        variable:
            The state the reduction reads, unused for `'count'`.

        value:
            The initial value, before the first synchronisation point.

    Examples:
        >>> from skcomplex.model import Aggregate
        >>> Aggregate('size', group='farmers').reduce
        'count'
    """

    group: str = ''  # type: ignore[assignment]  # the summarised group's label replaces the group constructor
    reduce: str | Callable[[Any], float] = 'count'
    variable: str | None = None
    value: float = state(default=0.0)


class Group(BaseEstimator):
    """Group of identical elements of one type, whose variables are arrays with one row per member.

    Groups are usually created with `Element.group`. A value is either an array whose first dimension is the size of
    the group, or one value shared by every member. Variables without a value take the default of the element type.

    Args:
        label:
            The label of the group, unique among the children of its system.

        element:
            The element type of the members.

        size:
            The number of members.

        values:
            A mapping from state or parameter name to its per-member or shared value.

        visual:
            The visual of the group, or `None` to use the visual of the element type.

    Examples:
        >>> from skcomplex.model import Group
        >>> from skcomplex.models.ecology import Population
        >>> Group('herds', element=Population, size=3).size
        3
    """

    def __init__(
        self,
        label: str,
        *,
        element: type[Element],
        size: int,
        values: Mapping[str, Any] | None = None,
        visual: VisualSpec | None = None,
    ) -> None:
        self.label = label
        self.element = element
        self.size = size
        self.values = values
        self.visual = visual

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label, element type, and size."""
        return f'Group({self.label!r}, element={self.element.__name__}, size={self.size})'


class Interaction(Component):
    """Base class of laws that couple elements to each other or to the environment.

    Subclasses declare their roles with `role` and their parameters with `parameter`, and override `laws`. Inside
    `laws`, `s.<role>` is the namespace of the element bound to the role and `s.<parameter>` is a symbol.

    Args:
        label:
            The label of the interaction, unique among the children of its system.

    Examples:
        >>> from skcomplex.core import Domain, parameter, role
        >>> from skcomplex.model import Interaction, Rate
        >>> class Growth(Interaction):
        ...     target: str = role()
        ...     r: float = parameter(1.0, domain=Domain.POSITIVE)
        ...
        ...     def laws(self, s):
        ...         return [Rate(s.target.n, s.r * s.target.n)]
        >>> Growth('growth', target='rabbits', r=0.5)
        Growth(label='growth', r=0.5, target='rabbits')
    """


class PairInteraction(Interaction):
    """Base class of interactions between pairs of members within a cutoff distance.

    The roles `first` and `second` bind to groups or elements, the same group for pairs inside one group. Subclasses
    return rate laws from `laws`, written for one pair, where `s.first` and `s.second` are the two members. The
    compiled system sums each law over every pair whose positions are closer than `cutoff`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or element of the first members.

        second:
            The path of the group or element of the second members.

        cutoff:
            The distance beyond which pairs do not interact.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

    Examples:
        >>> from skcomplex.model import PairInteraction, Rate
        >>> class Repulsion(PairInteraction):
        ...     def laws(self, s):
        ...         return [Rate(s.first.v, s.first.x - s.second.x)]
        >>> Repulsion('push', first='gas', second='gas', cutoff=0.1).cutoff
        0.1
    """

    position: ClassVar[str] = 'x'
    first: str = role()
    second: str = role()
    cutoff: float = parameter(1.0, domain=Domain.POSITIVE)
    period: float | None = None
