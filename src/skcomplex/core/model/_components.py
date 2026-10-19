"""Define the elements and interactions that systems are built from."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, dataclass_transform

import numpy as np
from sklearn.base import BaseEstimator

from ..base import Domain, VisualSpec, companions_of, declaration_of, parameter, role, state
from ._lattices import LatticeSpace
from ._laws import Law
from ._spaces import Space
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
        """Add the companion states of higher-order and lagged states, and turn the subclass into a dataclass."""
        super().__init_subclass__(**kwargs)
        annotations = dict(inspect.get_annotations(cls))
        for name, value in list(vars(cls).items()):
            declaration = declaration_of(value)
            if declaration is None or (declaration.order == 1 and not declaration.lags):
                continue
            default = value.default if value.default is not dataclasses.MISSING else 0.0
            for companion, field in companions_of(name, declaration, default).items():
                if companion not in vars(cls):
                    setattr(cls, companion, field)
                    annotations[companion] = Any
        cls.__annotations__ = annotations
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


def _is_declaration(value: object) -> bool:
    """Check whether a value is a field declared with `state`, `parameter`, or `role`."""
    return isinstance(value, dataclasses.Field)


def _method(laws: Callable[..., list[Law]]) -> Callable[..., list[Law]]:
    """Return a `laws` method from a function of the namespace, or of the component and the namespace."""
    if len(inspect.signature(laws).parameters) >= 2:  # noqa: PLR2004
        return laws

    def method(self: Component, s: Symbols) -> list[Law]:
        """Return the laws of the function."""
        return laws(s)

    method.__doc__ = laws.__doc__ or 'Return the laws of the component.'
    return method


def _define(
    base: type['Component'],
    *,
    name: str,
    fields: Mapping[str, Any],
    laws: Callable[..., list[Law]] | None,
    attributes: Mapping[str, Any],
    module: str,
) -> type:
    """Return a new component type with declared fields, a laws function, and class attributes."""
    namespace: dict[str, Any] = {'__annotations__': dict.fromkeys(fields, Any), '__module__': module, **attributes}
    namespace.update(fields)
    namespace['__qualname__'] = name
    if laws is not None:
        namespace['laws'] = _method(laws)
    return type(name, (base,), namespace)


def _caller_module() -> str:
    """Return the module name of the code that defines a component type, so the type pickles by reference there."""
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
    return caller.f_globals.get('__name__', __name__) if caller is not None else __name__


class Element(Component):
    """Base class of entities that carry state, such as particles, populations, agents, or firms.

    Subclasses declare their states with `state` and their parameters with `parameter`, and can override `laws` to
    return their intrinsic dynamics. The class variable `visual` names how the state is drawn.

    Args:
        label:
            The label of the element, unique among the children of its system.

    Examples:
        >>> from skcomplex.core.base import Domain, state
        >>> from skcomplex.core.model import Element
        >>> class Population(Element):
        ...     n: float = state(1.0, domain=Domain.NONNEGATIVE)
        >>> Population('rabbits', n=40)
        Population(label='rabbits', n=40)
    """

    visual: ClassVar[VisualSpec] = VisualSpec()

    @classmethod
    def define(
        cls,
        name: str,
        *,
        states: Mapping[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
        laws: Callable[..., list[Law]] | None = None,
        visual: VisualSpec | None = None,
    ) -> type['Element']:
        """Return a new element type built from declarations, without writing a class.

        The type behaves like a subclass in groups, cloning, and parameter introspection. Assign it to a module-level
        name equal to `name` for it to pickle.

        Args:
            name:
                The name of the type.

            states:
                A mapping from state name to a `state` declaration or to a default value.

            parameters:
                A mapping from parameter name to a `parameter` declaration or to a default value.

            laws:
                A function `laws(s)` or `laws(self, s)` that returns the intrinsic laws of the type.

            visual:
                The visual of the type, or `None` to inherit it.

        Returns:
            element_type:
                The new element type, a subclass of the type `define` is called on.

        Examples:
            >>> from skcomplex.core.base import Domain, state
            >>> from skcomplex.core.model import Element, Rate
            >>> Decaying = Element.define(
            ...     'Decaying',
            ...     states={'n': state(1.0, domain=Domain.NONNEGATIVE)},
            ...     parameters={'k': 0.5},
            ...     laws=lambda s: [Rate(s.n, -s.k * s.n)],
            ... )
            >>> Decaying('atoms', n=10.0)
            Decaying(label='atoms', n=10.0)
        """
        fields = {
            **{key: value if _is_declaration(value) else state(value) for key, value in (states or {}).items()},
            **{key: value if _is_declaration(value) else parameter(value) for key, value in (parameters or {}).items()},
        }
        attributes = {} if visual is None else {'visual': visual}
        return _define(cls, name=name, fields=fields, laws=laws, attributes=attributes, module=_caller_module())

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
            >>> from skcomplex.models.biology.ecology import Population
            >>> Population.group('herds', size=3, n=[10, 20, 30])
            Group('herds', element=Population, size=3)
        """
        return Group(label, element=cls, size=size, values=values, visual=visual)

    @classmethod
    def lattice(
        cls,
        label: str,
        space: LatticeSpace,
        *,
        visual: VisualSpec | None = None,
        **values: Any,  # noqa: ANN401
    ) -> 'Group':
        """Return a group of cells of this element type, one member per site of a lattice.

        Laws of the element type are written for one cell, and local rules read the cells around it. Lattice groups
        stream to Rerun as images unless a visual is given.

        Args:
            label:
                The label of the group.

            space:
                The lattice.

            visual:
                The visual of the group, or `None` for an image of every state.

            **values:
                Arrays of the lattice shape, or values shared by every cell.

        Returns:
            group:
                The group.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.base import Domain, state
            >>> from skcomplex.core.model import Element, LatticeSpace
            >>> Cell = Element.define('Cell', states={'alive': state(0, domain=Domain.BOOLEAN)})
            >>> cells = Cell.lattice('cells', LatticeSpace((3, 4)), alive=np.eye(3, 4))
            >>> cells.size, cells.values['alive'].shape
            (12, (12,))
        """
        flat = {
            name: np.ravel(value) if np.shape(value) == space.dimensions else value for name, value in values.items()
        }
        visual = VisualSpec('image') if visual is None else visual
        return Group(label, element=cls, size=space.size, values=flat, visual=visual, space=space)


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
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.model import AgentType, Rule
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
        >>> from skcomplex.core.model import Aggregate
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

        space:
            The space whose sites the members occupy, such as a lattice with one member per site, or `None` for the
            space of the enclosing system.

    Examples:
        >>> from skcomplex.core.model import Group
        >>> from skcomplex.models.biology.ecology import Population
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
        space: Space | None = None,
    ) -> None:
        self.label = label
        self.element = element
        self.size = size
        self.values = values
        self.visual = visual
        self.space = space

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
        >>> from skcomplex.core.base import Domain, parameter, role
        >>> from skcomplex.core.model import Interaction, Rate
        >>> class Growth(Interaction):
        ...     target: str = role()
        ...     r: float = parameter(1.0, domain=Domain.POSITIVE)
        ...
        ...     def laws(self, s):
        ...         return [Rate(s.target.n, s.r * s.target.n)]
        >>> Growth('growth', target='rabbits', r=0.5)
        Growth(label='growth', r=0.5, target='rabbits')
    """

    @classmethod
    def define(
        cls,
        name: str,
        *,
        roles: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        laws: Callable[..., list[Law]] | None = None,
    ) -> type['Interaction']:
        """Return a new interaction type built from declarations, without writing a class.

        The type behaves like a subclass in cloning and parameter introspection. Assign it to a module-level name equal
        to `name` for it to pickle.

        Args:
            name:
                The name of the type.

            roles:
                The names of the roles, each bound to an element path in the constructor.

            parameters:
                A mapping from parameter name to a `parameter` declaration or to a default value.

            laws:
                A function `laws(s)` or `laws(self, s)` that returns the laws of the interaction.

        Returns:
            interaction_type:
                The new interaction type, a subclass of the type `define` is called on.

        Examples:
            >>> from skcomplex.core.model import Interaction, Rate
            >>> Growth = Interaction.define(
            ...     'Growth',
            ...     roles=['target'],
            ...     parameters={'r': 1.0},
            ...     laws=lambda s: [Rate(s.target.n, s.r * s.target.n)],
            ... )
            >>> Growth('growth', target='rabbits', r=0.5)
            Growth(label='growth', r=0.5, target='rabbits')
        """
        fields = {
            **{key: role() for key in roles},
            **{key: value if _is_declaration(value) else parameter(value) for key, value in (parameters or {}).items()},
        }
        return _define(cls, name=name, fields=fields, laws=laws, attributes={}, module=_caller_module())


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
        >>> from skcomplex.core.model import PairInteraction, Rate
        >>> class Repulsion(PairInteraction):
        ...     def laws(self, s):
        ...         return [Rate(s.first.v, s.first.x - s.second.x)]
        >>> Repulsion('push', first='gas', second='gas', cutoff=0.1).cutoff
        0.1
    """

    position: ClassVar[str] = 'x'
    first: str = role()
    second: str = role()
    cutoff: float = parameter(default=1.0, domain=Domain.POSITIVE)
    period: float | None = None
