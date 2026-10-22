"""Define the components systems are built from: elements, aggregates, and groups of elements."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, dataclass_transform

import numpy as np
from sklearn.base import BaseEstimator

from ..base import (
    ComponentKind,
    LabelError,
    Requires,
    SpaceError,
    VariableKind,
    VisualSpec,
    companions_of,
    declaration_of,
    declarations_of,
    parameter,
    role,
    state,
    symbol,
)
from ..laws._laws import Law, Rule
from ..laws._symbols import Symbols
from ..space._lattices import LatticeSpace
from ..space._spaces import NetworkSpace, Space


def _check_reserved(cls: type, reserved: frozenset[str]) -> None:
    """Raise an error for a state or parameter whose name a component or the views of rules reserve."""
    for name, value in vars(cls).items():
        if name in reserved and declaration_of(value) is not None:
            error_msg = (
                f'{cls.__name__} declares the variable {name!r}, whose name is reserved. Rename it, since '
                f'{", ".join(sorted(reserved))} are reserved names.'
            )
            raise LabelError(error_msg)


@dataclass_transform(kw_only_default=True, field_specifiers=(dataclasses.field, state, parameter, role))
@dataclasses.dataclass(eq=False, repr=False)
class Component(BaseEstimator):
    """Base class of elements and interactions.

    Every subclass becomes a dataclass whose fields after `label` are keyword-only, so the constructor stores every
    argument unmodified under its own name. A state or parameter whose name is in `reserved_names` raises `LabelError`
    when the subclass is created.

    Args:
        label:
            The label of the component, unique among the children of its system.
    """

    reserved_names: ClassVar[frozenset[str]] = frozenset({'label'})
    label: str

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Add the companion states of higher-order and lagged states, and turn the subclass into a dataclass."""
        super().__init_subclass__(**kwargs)
        _check_reserved(cls, cls.reserved_names)
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


def _lattice_visual(element: type, visual: VisualSpec | None, default: str) -> VisualSpec:
    """Return the visual of a lattice group: the given one, the element type's, or the default kind."""
    if visual is not None:
        return visual
    declared: VisualSpec = getattr(element, 'visual', VisualSpec())
    return VisualSpec(default) if declared == VisualSpec() else declared


def _caller_module() -> str:
    """Return the module name of the code that defines a component type, so the type pickles by reference there."""
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
    return caller.f_globals.get('__name__', __name__) if caller is not None else __name__


class Element(Component):
    """Base class of entities that carry state, such as particles, populations, agents, or firms.

    Subclasses declare their states with `state` and their parameters with `parameter`, and can override `laws` to
    return their intrinsic dynamics. The class variable `visual` names how the state is drawn.

    An element whose laws include `Rule` laws, alone or in a group, is an agent: under discrete time it runs on the
    agent stepper, whose rules receive every member as an agent with its states as attributes. Its states then hold
    any value, not only numbers. The class variable `remembers` maps state names to the number of past values the
    stepper keeps of them, which `memory` returns. A member is removed only when its states tagged with a quantity are
    settled to zero, and its states and parameters cannot take the names of the agent view, such as `location` or
    `alive`, or of the attributes Mesa reads, such as `position` or `model`.

    Args:
        label:
            The label of the element, unique among the children of its system.

    Examples:
        >>> from skcomplex.core.base import Domain, state
        >>> from skcomplex.core.components import Element
        >>> class Population(Element):
        ...     n: float = state(1.0, domain=Domain.NONNEGATIVE)
        >>> Population('rabbits', n=40)
        Population(label='rabbits', n=40)
    """

    component_kind: ClassVar[ComponentKind] = ComponentKind.ELEMENT
    requires: ClassVar[Requires] = Requires()

    reserved_names: ClassVar[frozenset[str]] = frozenset({'label', 'visual', 'remembers'})
    visual: ClassVar[VisualSpec] = VisualSpec()
    remembers: ClassVar[Mapping[str, int]] = {}

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
            >>> from skcomplex.core.components import Element
            >>> from skcomplex.core.laws import Rate
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
        /,
        *,
        visual: VisualSpec | None = None,
        variable: bool = False,
        location: object = None,
        **values: object,
    ) -> 'Group':
        """Return a group of identical members of this element type.

        The label and the size are given by position, so a state named `size` takes per-member values by keyword.

        Args:
            label:
                The label of the group.

            size:
                The number of members.

            visual:
                The visual of the group, or `None` to use the visual of the element type.

            variable:
                Whether laws add and remove members during a run, such as the births and deaths of agents.

            location:
                The initial places of agents in the space: cell coordinates, a network node, or a position, one per
                member or one shared by every member, or `None` for random places.

            **values:
                Per-member arrays whose first dimension is `size`, or values shared by every member.

        Returns:
            group:
                The group.

        Examples:
            >>> from skcomplex.models.biology.ecology import Population
            >>> Population.group('herds', 3, n=[10, 20, 30])
            Group('herds', element=Population, size=3)
        """
        return Group(label, element=cls, size=size, values=values, visual=visual, variable=variable, location=location)

    @classmethod
    def lattice(
        cls,
        label: str,
        space: LatticeSpace | NetworkSpace,
        /,
        *,
        visual: VisualSpec | None = None,
        **values: Any,  # noqa: ANN401
    ) -> 'Group':
        """Return a group of cells of this element type, one member per site of a lattice or node of a network.

        Laws of the element type are written for one cell, and local rules read the cells around it. Square lattice
        groups stream to Rerun as images and other lattice groups as sites, unless a visual is given or the element
        type declares one. The label and the lattice are given by position.

        Args:
            label:
                The label of the group.

            space:
                The lattice, or the network whose nodes are the sites, in the order of `graph.nodes`.

            visual:
                The visual of the group, or `None` for the visual of the element type, and, when the type declares
                none, an image of every state of a square lattice and the sites of other lattices.

            **values:
                Arrays of the lattice shape, arrays in node order or mappings from node to value on a network, or
                values shared by every cell.

        Returns:
            group:
                The group.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.base import Domain, state
            >>> from skcomplex.core.components import Element
            >>> from skcomplex.core.space import LatticeSpace
            >>> Cell = Element.define('Cell', states={'alive': state(0, domain=Domain.BOOLEAN)})
            >>> cells = Cell.lattice('cells', LatticeSpace((3, 4)), alive=np.eye(3, 4))
            >>> cells.size, cells.values['alive'].shape
            (12, (12,))
        """
        if isinstance(space, NetworkSpace):
            nodes = list(space.graph.nodes)
            by_node = {
                name: np.asarray([value[node] for node in nodes]) if isinstance(value, Mapping) else value
                for name, value in values.items()
            }
            visual = _lattice_visual(cls, visual, 'sites')
            return Group(label, element=cls, size=len(nodes), values=by_node, visual=visual, space=space, on='sites')
        flat = {
            name: np.ravel(value) if np.shape(value) == space.dimensions else value for name, value in values.items()
        }
        default = 'image' if space.geometry == 'square' else 'sites'
        visual = _lattice_visual(cls, visual, default)
        return Group(label, element=cls, size=space.size, values=flat, visual=visual, space=space, on='sites')


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
        >>> from skcomplex.core.components import Aggregate
        >>> Aggregate('size', group='farmers').reduce
        'count'
    """

    group: str = ''  # type: ignore[assignment]  # the summarised group's label replaces the group constructor
    reduce: str | Callable[[Any], float] = 'count'
    variable: str | None = None
    value: float = state(default=0.0)


PARTS = ('sites', 'edges', 'hyperedges')
BINDINGS = ('members', 'pairs', 'neighbours', 'hyperedges')


def _is_state(item: dataclasses.Field) -> bool:
    """Check whether a dataclass field is declared with `state`."""
    declaration = declaration_of(item)
    return declaration is not None and declaration.kind is VariableKind.STATE


def part_count(space: Space, on: str, label: str) -> int:
    """Return the number of parts of a space a group has one member on.

    Args:
        space:
            The space.

        on:
            `'sites'`, `'edges'`, or `'hyperedges'`.

        label:
            The label of the group, named in errors.

    Returns:
        count:
            The number of sites, edges, or hyperedges.

    Raises:
        SpaceError:
            The space does not have the part.
    """
    graph = getattr(space, 'graph', None)
    if on == 'sites' and graph is not None:
        return int(graph.number_of_nodes())
    if on == 'sites' and hasattr(space, 'size'):
        return int(space.size)
    if on == 'edges' and graph is not None:
        return int(graph.number_of_edges())
    if on == 'hyperedges' and getattr(space, 'hyperedges', ()):
        return len(space.hyperedges)
    needs = 'a network space with hyperedges' if on == 'hyperedges' else 'a network space' if on == 'edges' else 'sites'
    error_msg = f'Group {label!r} lives on {on}, which need {needs}, but its space is {space!r}.'
    raise SpaceError(error_msg)


class _FreeSymbols(Symbols):
    """Namespace that returns a free symbol for every name, so laws can be listed before the states are declared."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Return a free symbol of the name."""
        if name.startswith('__'):
            raise AttributeError(name)
        return symbol(name)


def probe_laws(component: object) -> list[Law] | None:
    """Return the laws of a component for a namespace of free symbols, before its states are declared.

    A group returns the laws of a representative member, and an interaction reads one namespace of free symbols per
    role. Laws that cannot be written with free symbols, such as laws that index a vector state, are not listed.

    Args:
        component:
            The element, group, or interaction.

    Returns:
        laws:
            The laws, or `None` when they cannot be listed.

    Examples:
        >>> from skcomplex.core.components import Element, probe_laws
        >>> from skcomplex.core.laws import Rule
        >>> Walker = Element.define('Walker', laws=lambda s: [Rule(lambda agent, model: None)])
        >>> [type(law).__name__ for law in probe_laws(Walker('walker'))]
        ['Rule']
    """
    if isinstance(component, Group):
        component_type: type = component.element
    elif isinstance(component, Component):
        component_type = type(component)
    else:
        return None
    if component_type.laws is Component.laws:
        return []
    roles = declarations_of(component_type, VariableKind.ROLE)
    try:
        member = (
            component.element(component.label, **(component.values or {}))
            if isinstance(component, Group)
            else component
        )
        return list(member.laws(_FreeSymbols(**{name: _FreeSymbols() for name in roles})))
    except Exception:  # noqa: BLE001
        return None


def runs_rules(component: object) -> bool:
    """Check whether a component runs rules of its own: a variable group, or an element or group with rule laws.

    Such a component is an agent. The compiler also makes agents of the groups whose element type has no laws of its
    own in a system with rules, unless an interaction whose laws are not rules binds them.

    Args:
        component:
            The element, group, interaction, or system.

    Returns:
        runs_rules:
            Whether the component is a variable group, or an element or group whose own laws include a `Rule`.

    Examples:
        >>> from skcomplex.core.components import Element, runs_rules
        >>> from skcomplex.core.laws import Rule
        >>> Walker = Element.define('Walker', laws=lambda s: [Rule(lambda agent, model: None)])
        >>> runs_rules(Walker('walker')), runs_rules(Element.define('Plain')('plain'))
        (True, False)
    """
    if isinstance(component, Group) and component.variable:
        return True
    if not isinstance(component, Element | Group):
        return False
    return any(isinstance(law, Rule) for law in probe_laws(component) or [])


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

        variable:
            Whether laws add and remove members during a run, such as the births and deaths of agents.

        on:
            The part of the space with one member each, `'sites'`, `'edges'`, or `'hyperedges'`, or `None` for members
            that are not tied to parts of the space. With a part and no size, the size is the number of parts.

        location:
            The initial places of agents in the space: cell coordinates, a network node, or a position, one per member
            or one shared by every member, or `None` for random places. Only the agent stepper reads it.

    Raises:
        SpaceError:
            The part is unknown, or the space does not have it, such as edges outside a network.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.components import Group
        >>> from skcomplex.core.space import NetworkSpace
        >>> from skcomplex.models.biology.ecology import Population
        >>> Group('herds', element=Population, size=3).size
        3
        >>> Group('weights', element=Population, space=NetworkSpace(nx.cycle_graph(30)), on='edges').size
        30
    """

    component_kind: ClassVar[ComponentKind] = ComponentKind.GROUP

    def __init__(
        self,
        label: str,
        *,
        element: type[Element],
        size: int | None = None,
        values: Mapping[str, Any] | None = None,
        visual: VisualSpec | None = None,
        space: Space | None = None,
        variable: bool = False,
        on: str | None = None,
        location: object = None,
    ) -> None:
        self.label = label
        self.element = element
        self.size = size if size is not None or on is None or space is None else part_count(space, on, label)
        self.values = values
        self.visual = visual
        self.space = space
        self.variable = variable
        self.on = on
        self.location = location
        if on is not None and on not in PARTS:
            error_msg = f'Group {label!r} lives on {on!r}. Use one of {", ".join(PARTS)}.'
            raise SpaceError(error_msg)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label, element type, and size."""
        return f'Group({self.label!r}, element={self.element.__name__}, size={self.size})'
