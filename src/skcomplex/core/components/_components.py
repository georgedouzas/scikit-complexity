"""Define the components systems are built from: elements, and groups of identical elements."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np

from ..base import (
    BaseComponent,
    BaseInteraction,
    BaseLaw,
    BaseSpace,
    ComponentKind,
    NumericDomain,
    Requires,
    SpaceError,
    ValidationError,
    VariableKind,
    VisualSpec,
    declarations_of,
    symbol,
)
from ..laws import Symbols
from ..space import LatticeSpace, NetworkSpace, SiteSpace, part_count

PARTS = ('sites', 'edges', 'hyperedges')


def _lattice_visual(element: type, visual: VisualSpec | None, default: str) -> VisualSpec:
    """Return the visual of a lattice group: the given one, the element type's, or the default kind."""
    if visual is not None:
        return visual
    declared: VisualSpec = getattr(element, 'visual', VisualSpec())
    return VisualSpec(default) if declared == VisualSpec() else declared


def _check_element(label: str, element: object) -> None:
    """Raise an error for a group whose element type is not an `ElementComponent` subclass."""
    if isinstance(element, type) and issubclass(element, ElementComponent):
        return
    error_msg = f'Group {label!r} has element {element!r}, which is not an Element subclass.'
    raise ValidationError(error_msg)


def _check_size(group: 'GroupComponent') -> None:
    """Raise an error for a group whose size is not a positive integer, except an empty variable group on parts."""
    if group.size is None and group.on is not None:
        return
    empty = group.variable and group.on in {'sites', 'edges'} and group.size == 0
    if empty or (NumericDomain.POSITIVE.contains(group.size) and NumericDomain.INTEGER.contains(group.size)):
        return
    error_msg = f'Group {group.label!r} has size {group.size!r}, which is not a positive integer.'
    raise ValidationError(error_msg)


def _check_value_names(group: 'GroupComponent') -> None:
    """Raise an error for a value whose name the element type of a group does not declare."""
    unknown = sorted(set(group.values or {}) - set(group.element.get_argument_names()))
    if not unknown:
        return
    states = ', '.join(declarations_of(group.element, VariableKind.STATE)) or 'none'
    parameters = ', '.join(declarations_of(group.element, VariableKind.PARAMETER)) or 'none'
    error_msg = (
        f'Group {group.label!r} sets {", ".join(unknown)}, which {group.element.__name__} does not declare. Its '
        f'states are {states}, and its parameters are {parameters}.'
    )
    raise ValidationError(error_msg)


def _check_value_rows(group: 'GroupComponent') -> None:
    """Raise an error for a numeric per-member value whose first dimension is not the size of a group."""
    if group.size is None:
        return
    declarations = group.element._declarations()
    dimensions = group.space.dimensions if isinstance(group.space, LatticeSpace) else None
    for name, value in (group.values or {}).items():
        declaration = declarations.get(name)
        if declaration is None or not isinstance(declaration.domain, NumericDomain):
            continue
        try:
            shape = np.shape(value)
        except ValueError:
            continue
        shared = 1 if declaration.vector else 0
        if len(shape) <= shared or shape[0] == group.size or shape == dimensions:
            continue
        error_msg = (
            f'Group {group.label!r} has {shape[0]} values of {name} for {group.size} members. Give one value per '
            'member, or one value shared by every member.'
        )
        raise ValidationError(error_msg)


def _check_group(group: 'GroupComponent') -> None:
    """Raise an error for a group whose element type, size, value names, or value rows are not valid."""
    _check_element(group.label, group.element)
    _check_size(group)
    _check_value_names(group)
    _check_value_rows(group)


class ElementComponent(BaseComponent):
    """Base class of entities that carry state, such as particles, populations, agents, or firms.

    Subclasses declare their states with `state` and their parameters with `parameter`, and can override `laws` to
    return their intrinsic dynamics. The class variable `visual` names how the state is drawn.

    An element whose laws have the agent trait, such as rules, alone or in a group, is an agent: under discrete time it
    runs on the agent stepper, whose rules receive every member as an agent with its states as attributes. Its states
    then hold any value, not only numbers. The class variable `remembers` maps state names to the number of past values
    the stepper keeps of them, which `memory` returns. A member is removed only when its states tagged with a quantity
    are settled to zero, and its states and parameters cannot take the names of the agent view, such as `location` or
    `alive`, or of the attributes Mesa reads, such as `position` or `model`.

    Args:
        label:
            The label of the element, unique among the children of its system.

    Examples:
        >>> from skcomplex.core.base import NumericDomain, state
        >>> from skcomplex.core.components import ElementComponent
        >>> class PopulationComponent(ElementComponent):
        ...     n: float = state(1.0, domain=NumericDomain.NONNEGATIVE)
        >>> PopulationComponent('rabbits', n=40)
        PopulationComponent(label='rabbits', n=40)
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
        laws: Callable[..., list[BaseLaw]] | None = None,
        visual: VisualSpec | None = None,
    ) -> type['ElementComponent']:
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
            >>> from skcomplex.core.base import NumericDomain, state
            >>> from skcomplex.core.components import ElementComponent
            >>> from skcomplex.core.laws import RateLaw
            >>> DecayingComponent = ElementComponent.define(
            ...     'DecayingComponent',
            ...     states={'n': state(1.0, domain=NumericDomain.NONNEGATIVE)},
            ...     parameters={'k': 0.5},
            ...     laws=lambda s: [RateLaw(s.n, -s.k * s.n)],
            ... )
            >>> DecayingComponent('atoms', n=10.0)
            DecayingComponent(label='atoms', n=10.0)
        """
        attributes = {} if visual is None else {'visual': visual}
        return cls._define(name, states=states, parameters=parameters, laws=laws, attributes=attributes)

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
    ) -> 'GroupComponent':
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
            >>> from skcomplex.models.biology.ecology import PopulationComponent
            >>> PopulationComponent.group('herds', 3, n=[10, 20, 30])
            GroupComponent('herds', element=PopulationComponent, size=3)
        """
        return GroupComponent(
            label,
            element=cls,
            size=size,
            values=values,
            visual=visual,
            variable=variable,
            location=location,
        )

    @classmethod
    def lattice(
        cls,
        label: str,
        space: SiteSpace | NetworkSpace,
        /,
        *,
        visual: VisualSpec | None = None,
        **values: Any,  # noqa: ANN401
    ) -> 'GroupComponent':
        """Return a group of cells of this element type, one member per site of a lattice or node of a network.

        Laws of the element type are written for one cell, and local rules read the cells around it. Square lattice
        groups stream to Rerun as images, and groups on other spaces of sites with the default visual of their space,
        unless a visual is given or the element type declares one. The label and the lattice are given by position.

        Args:
            label:
                The label of the group.

            space:
                The lattice or other space of sites, or the network whose nodes are the sites, in the order of
                `graph.nodes`.

            visual:
                The visual of the group, or `None` for the visual of the element type, and, when the type declares
                none, the default visual of the space, an image of every state of a square lattice and the sites of
                other lattices.

            **values:
                Arrays of the lattice shape, arrays in node order or mappings from node to value on a network, or
                values shared by every cell.

        Returns:
            group:
                The group.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.base import NumericDomain, state
            >>> from skcomplex.core.components import ElementComponent
            >>> from skcomplex.core.space import LatticeSpace
            >>> CellComponent = ElementComponent.define(
            ...     'CellComponent', states={'alive': state(0, domain=NumericDomain.BOOLEAN)}
            ... )
            >>> cells = CellComponent.lattice('cells', LatticeSpace((3, 4)), alive=np.eye(3, 4))
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
            return GroupComponent(
                label,
                element=cls,
                size=len(nodes),
                values=by_node,
                visual=visual,
                space=space,
                on='sites',
            )
        flat = {
            name: np.ravel(value) if np.shape(value) == space.dimensions else value for name, value in values.items()
        }
        visual = _lattice_visual(cls, visual, space.default_visual)
        return GroupComponent(label, element=cls, size=space.size, values=flat, visual=visual, space=space, on='sites')


class GroupComponent(BaseComponent):
    """Group of identical elements of one type, whose variables are arrays with one row per member.

    Groups are usually created with `ElementComponent.group`. A value is either an array whose first dimension is the
    size of the group, or one value shared by every member. Variables without a value take the default of the element
    type.

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

        layer:
            The layer of the network whose edges a group on edges lives on, or `None` for the graph of the network.

        capacity:
            The largest number of members of a variable group on the sites or edges of a network, which compiled
            backends reserve rows for, or `None` for twice the initial size.

    Raises:
        SpaceError:
            The space does not have the part, such as edges outside a network.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.components import GroupComponent
        >>> from skcomplex.core.space import NetworkSpace
        >>> from skcomplex.models.biology.ecology import PopulationComponent
        >>> GroupComponent('herds', element=PopulationComponent, size=3).size
        3
        >>> GroupComponent(
        ...     'weights', element=PopulationComponent, space=NetworkSpace(nx.cycle_graph(30)), on='edges'
        ... ).size
        30
        >>> from skcomplex.core.space import Layer
        >>> space = NetworkSpace(nx.cycle_graph(4), layers=[Layer('work', nx.complete_graph(4))])
        >>> GroupComponent('wages', element=PopulationComponent, space=space, on='edges', layer='work').size
        6
        >>> herds = GroupComponent('herds', element=PopulationComponent, size=3).fit()
        >>> herds.default_visual_ == PopulationComponent.visual
        True
    """

    component_kind: ClassVar[ComponentKind] = ComponentKind.GROUP

    def __init__(
        self,
        label: str,
        *,
        element: type[ElementComponent],
        size: int | None = None,
        values: Mapping[str, Any] | None = None,
        visual: VisualSpec | None = None,
        space: BaseSpace | None = None,
        variable: bool = False,
        on: str | None = None,
        location: object = None,
        layer: str | None = None,
        capacity: int | None = None,
    ) -> None:
        super().__init__(label)
        self.element = element
        self.size = size if size is not None or on is None or space is None else part_count(space, on, label, layer)
        self.values = values
        self.visual = visual
        self.space = space
        self.variable = variable
        self.on = on
        self.location = location
        self.layer = layer
        self.capacity = capacity

    def _check(self) -> None:
        """Raise an error for a part, an element type, a size, or values the group cannot take.

        Raises:
            SpaceError:
                The part is unknown.

            ValidationError:
                The element type is not an `ElementComponent` subclass, the size is not a positive integer, a value
                names a variable the element type does not declare, or a per-member value does not have one row per
                member.
        """
        super()._check()
        if self.on is not None and self.on not in PARTS:
            error_msg = f'Group {self.label!r} lives on {self.on!r}. Use one of {", ".join(PARTS)}.'
            raise SpaceError(error_msg)
        _check_group(self)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label, element type, and size."""
        return f'{type(self).__name__}({self.label!r}, element={self.element.__name__}, size={self.size})'

    def _fit(self) -> None:
        """Set `declarations_` and `default_visual_`, the visual of the group when it sets none.

        The default visual of a group is the visual of its element type.
        """
        super()._fit()
        self.default_visual_ = self.element.visual


def probe_laws(component: object) -> list[BaseLaw] | None:
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
        >>> from skcomplex.core.components import ElementComponent, probe_laws
        >>> from skcomplex.core.laws import RateLaw
        >>> DecayComponent = ElementComponent.define('DecayComponent', laws=lambda s: [RateLaw(s.x, -s.x)])
        >>> [type(law).__name__ for law in probe_laws(DecayComponent('decay'))]
        ['RateLaw']
    """
    if isinstance(component, GroupComponent):
        component_type: type[BaseComponent | BaseInteraction] = component.element
    elif isinstance(component, BaseComponent | BaseInteraction):
        component_type = type(component)
    else:
        return None
    if component_type.laws in (BaseComponent.laws, BaseInteraction.laws):
        return []
    roles = declarations_of(component_type, VariableKind.ROLE)
    try:
        member = (
            component.element(component.label, **(component.values or {}))
            if isinstance(component, GroupComponent)
            else component
        )
        return list(member.run(_FreeSymbols(**{name: _FreeSymbols() for name in roles})))
    except Exception:  # noqa: BLE001
        return None


class _FreeSymbols(Symbols):
    """Namespace that returns a free symbol for every name, so laws can be listed before the states are declared."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Return a free symbol of the name, or a namespace of free symbols for `other`, the neighbour of a member."""
        if name.startswith('__'):
            raise AttributeError(name)
        if name == 'other':
            return _FreeSymbols()
        return symbol(name)
