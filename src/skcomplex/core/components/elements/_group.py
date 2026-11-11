"""Define the element that holds many members of one element class as arrays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import ZZ

from ...errors import SpaceError, ValidationError
from ...specs import RoleSpec, Spec, StateSpec, Tags, VisualSpec
from ..base import BaseComponent
from ..spaces import BaseSpace, NetworkSpace
from ._base import BaseElement
from ._declared import DeclaredElement

PARTS = ('sites', 'edges', 'hyperedges')
MIN_RANK = 2


class GroupElement(BaseElement):
    """An element that holds many members of one element class as arrays.

    Each state and parameter of the group is an array with one row per member, or one value that every member shares. A
    variable without a value takes the default of the element class. The laws of the group are those of the class,
    written for one member and run for every member at once. A group may sit on the sites, the edges, or the hyperedges
    of a space, with one member on each. A hyperedge is a link that joins more than two nodes of a network. With
    `variable=True`, laws add and remove members during a run, as births and deaths do.

    Unlike `Element`, which is one entity, it holds many entities of the same class and runs their laws as arrays.
    Choose it when a model has many alike units, such as the cells of a lattice, the nodes of a network, or a population
    of agents.

    It appears in many fields. The herds of a grazing ecosystem, the assets of a portfolio, the cells of a tissue, and
    the generators of a power grid are groups of one element class.

    Args:
        label:
            The name of the group, unique among the components of its system, such as `'herds'`.

        element:
            The element class of every member, such as a class with one state `x` for the size of a herd.

        size:
            The number of members, such as `100`, or `None` for one member per part of the space that `on` names.

        values:
            The value of each state or parameter by name, with one row per member or one value for all, such as
            `{'x': [40.0, 25.0, 60.0]}`. On sites, a value may also be an array of the lattice shape, or a mapping from
            each node of a network to its value.

        visual:
            How the group is drawn, such as `VisualSpec('sites')`, or `None` for `default_visual_`.

        space:
            The space whose parts the members occupy, such as a lattice with one member per site or a network with one
            member per node in the order of `graph.nodes`, or `None` for the space of the enclosing system.

        variable:
            Whether laws add and remove members during a run, such as the births and deaths of agents.

        on:
            The part of the space that holds one member each, or `None` for members tied to no part. Without a size, the
            size is the number of these parts.

            - `'sites'`: the sites of a lattice or the nodes of a network.
            - `'edges'`: the edges of a network, such as the lines of a power grid.
            - `'hyperedges'`: the hyperedges of a network, such as the groups of a social network.

        location:
            The starting place of each agent, one per member or one for all, or `None` for random places. A place is a
            cell of a lattice, a node of a network, a position, or a polygon that holds the member at a random point
            inside it. Only the agent stepper reads it.

        layer:
            The layer of the network whose edges a group on edges sits on, such as `'work'`, or `None` for the main
            graph.

        capacity:
            The largest number of members of a variable group on the sites or edges of a network, which compiled
            backends reserve rows for, or `None` for twice the initial size.

        rank:
            The largest number of members of one hyperedge of a variable group on hyperedges, such as `3` for triangles,
            or `None` for the size of the initial hyperedges. Compiled backends reserve one column per member up to it.

    Attributes:
        default_visual_:
            The visual of a member, which the group takes when it sets none. When the member sets none, it is the
            default visual of the space of a group on sites, and the plain visual otherwise.

        size_:
            The number of members, which is `size` when it is given and otherwise the number of parts of the space.

        values_:
            The values with one row per member, or one value shared by every member.

    Examples:
        Three herds of grazing animals start at different sizes.

        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec
        >>> class StockElement(Element):
        ...     def __init__(
        ...         self, label, x: Annotated[float, StateSpec()] = 0.0, r: Annotated[float, ParameterSpec()] = 0.1
        ...     ):
        ...         super().__init__(label)
        ...         self.x = x
        ...         self.r = r
        >>> herds = GroupElement('herds', element=StockElement, size=3, values={'x': [40.0, 25.0, 60.0]})
        >>> herds.prepare().size_
        3

        The same class holds the assets of a portfolio, each with its own yearly return.

        >>> from skcomplex.core.components.interactions import RateInteraction
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> holdings = {'x': [100.0, 50.0], 'r': [0.05, 0.02]}
        >>> assets = GroupElement('assets', element=StockElement, size=2, values=holdings)
        >>> returns = RateInteraction('returns', target='assets', state='x', rate=lambda s: s.target.r * s.target.x)
        >>> portfolio = CompositeSystem(elements=[assets], interactions=[returns])
        >>> simulate(portfolio, until=1, dt=1.0).get_values('assets.x').round(2).tolist()
        [[100.0, 50.0], [105.13, 51.01]]

        On the edges of a network one member sits on each edge, such as the lines of a ring of five substations.

        >>> import networkx as nx
        >>> from skcomplex.core.components.spaces import NetworkSpace
        >>> lines = GroupElement('lines', element=StockElement, space=NetworkSpace(nx.cycle_graph(5)), on='edges')
        >>> lines.prepare().size_
        5
    """

    def __init__(
        self,
        label: str,
        *,
        element: type[BaseElement],
        size: int | None = None,
        values: Mapping[str, object] | None = None,
        visual: VisualSpec | None = None,
        space: BaseSpace | None = None,
        variable: bool = False,
        on: str | None = None,
        location: object = None,
        layer: str | None = None,
        capacity: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.element = element
        self.size = size
        self.values = values
        self.space = space
        self.variable = variable
        self.on = on
        self.location = location
        self.layer = layer
        self.capacity = capacity
        self.rank = rank

    def _get_size_of(self, space: BaseSpace | None = None) -> int:
        """Return the size of the group.

        The size is the number of parts of its own space or else of an inherited space.
        """
        if self.size is not None:
            return int(self.size)
        place = self.space if self.space is not None else space
        if place is None:
            error_msg = f'Group {self.label!r} lives on {self.on} but has neither a size nor a space.'
            raise SpaceError(error_msg)
        return place.prepare()._count_parts(self.on or 'sites', self.layer)

    def _get_member(self) -> BaseElement:
        """Return a member of the group, with the values the group gives."""
        return self.element(self.label).set_params(**(self.values or {}))

    def _get_member_specs(self) -> dict[str, Spec]:
        """Return the specs of a member.

        They are read from a member only when it declares variables from data.
        """
        if issubclass(self.element, DeclaredElement):
            return self._get_member()._get_component_specs()
        return self.element._get_specs()

    def get_declaring_element(self) -> type[BaseElement] | BaseElement:
        """Return what declares the states and parameters of a member.

        It is the element class, or a member with the values of the group when the class takes its variables as data.

        Returns:
            element:
                The element class, or a member with the values of the group.

        Examples:
            >>> from skcomplex.core.components.elements import DeclaredElement, GroupElement
            >>> from skcomplex.core.specs import DeclaredVariableSpec, StateSpec
            >>> values = {'variables': {'x': DeclaredVariableSpec(1.0, StateSpec())}}
            >>> group = GroupElement('cells', element=DeclaredElement, size=2, values=values)
            >>> group.get_declaring_element().get_value('x')
            1.0
        """
        return self._get_member() if issubclass(self.element, DeclaredElement) else self.element

    def get_states(self) -> dict[str, StateSpec]:
        """Return the states of a member of the group with their specs.

        Returns:
            states:
                The spec of every state, by name.

        Examples:
            >>> from skcomplex.core.components.elements import DeclaredElement, GroupElement
            >>> from skcomplex.core.specs import DeclaredVariableSpec, StateSpec
            >>> values = {'variables': {'x': DeclaredVariableSpec(1.0, StateSpec())}}
            >>> list(GroupElement('cells', element=DeclaredElement, size=3, values=values).get_states())
            ['x']
        """
        return {name: spec for name, spec in self._get_member_specs().items() if isinstance(spec, StateSpec)}

    def _get_tags(self) -> Tags:
        """Return the tags of the group.

        They are those of its members, on its own space, with a variable membership or not.
        """
        tags = replace(self._get_law_tags(self.element._get_declared_laws()), domains=self._get_domain_tags())
        space = frozenset() if self.space is None else self.space.get_tags().space
        structure = frozenset({'variable membership'}) if self.variable else frozenset()
        return replace(tags, space=space, structure=structure)

    def _get_draws(self) -> bool:
        """Return whether the group draws.

        It draws when it declares it or the class of its members does.
        """
        return self.draws or self.element.draws

    def _get_member_location(self, index: int, size: int) -> object:
        """Return the initial location of one of `size` members.

        It comes from per-member locations or one shared location.
        """
        location = self.location
        array = np.asarray(location, dtype=object if isinstance(location, list | tuple) else None)
        return array[index] if array.ndim >= 1 and array.shape[0] == size else location

    def _get_site_space(self, context: BaseSpace | None) -> BaseSpace | None:
        """Return the space with one member of the group per site, or `None`.

        It is the space of a group on sites with no size of its own, its own space or else the inherited one.
        """
        if self.on != 'sites' or self.size is not None:
            return None
        return self.space if self.space is not None else context

    def _flatten_values(self, space: BaseSpace | None) -> dict[str, object]:
        """Return the values with one row per member.

        On the sites of a space, mappings from node to value on a network and arrays of the shape of a lattice become
        one row per site, except the declared variables of a member that declares them from data. Without such a space
        the group takes its values as given.
        """
        values = dict(self.values or {})
        if space is None:
            return values
        if isinstance(space, NetworkSpace):
            nodes = list(space.graph.nodes)
            declared = issubclass(self.element, DeclaredElement)
            values = {
                name: (
                    np.asarray([value[node] for node in nodes])
                    if isinstance(value, Mapping) and not (declared and name == 'variables')
                    else value
                )
                for name, value in values.items()
            }
        return {
            name: np.asarray(value).ravel() if np.asarray(value).shape == space.shape_ else value
            for name, value in values.items()
        }

    def _check_value_rows(self) -> None:
        """Raise an error for a per-member value without one row per member.

        The value is that of a state or a parameter.
        """
        specs = self._get_member_specs()
        for name, value in self.values_.items():
            spec = specs.get(name)
            if spec is None or isinstance(spec, RoleSpec):
                continue
            try:
                shape = np.asarray(value).shape
            except ValueError:
                continue
            shared = 1 if spec.vector or getattr(spec, 'tensor', None) else 0
            if len(shape) <= shared or shape[0] == self.size_:
                continue
            error_msg = (
                f'Group {self.label!r} has {shape[0]} values of {name} for {self.size_} members. Give one value per '
                'member, or one value shared by every member.'
            )
            raise ValidationError(error_msg)

    def _check_on(self) -> None:
        """Raise an error for an `on` that names no part of a space."""
        if self.on is not None and self.on not in PARTS:
            error_msg = f'Group {self.label!r} lives on {self.on!r}. Use one of {", ".join(PARTS)}.'
            raise SpaceError(error_msg)

    def _check_element(self) -> None:
        """Raise an error for an element type that is not an element class."""
        if not isinstance(self.element, type) or not issubclass(self.element, BaseElement):
            error_msg = f'Group {self.label!r} has element {self.element!r}, which is not an element class.'
            raise ValidationError(error_msg)

    def _check_size(self) -> None:
        """Raise an error for a size that is not a positive integer.

        An empty variable group on parts is the exception.
        """
        if self.size is None and self.on is not None:
            return
        empty = self.variable and self.on in PARTS and self.size == 0
        positive_integer = self.size is not None and self.size in ZZ and self.size > 0
        if not empty and not positive_integer:
            error_msg = f'Group {self.label!r} has size {self.size!r}, which is not a positive integer.'
            raise ValidationError(error_msg)

    def _check_rank(self) -> None:
        """Raise an error for an invalid rank of a group.

        Only a variable group on hyperedges takes a rank, and the rank is a whole number of at least two.
        """
        if self.rank is None:
            return
        if not self.variable or self.on != 'hyperedges':
            error_msg = f'Group {self.label!r} has a rank, which only a variable group on hyperedges takes.'
            raise ValidationError(error_msg)
        if self.rank not in ZZ or self.rank < MIN_RANK:
            error_msg = f'Group {self.label!r} has rank {self.rank!r}. A hyperedge holds a whole number of at least 2.'
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, `size_`, and `values_`.

        It first raises an error for arguments the group cannot take. They are a part of the space, an element type, a
        size, a rank, or values.

        `default_visual_` is the visual of the group when it sets none. The default visual of a group is the visual of a
        member. When the member sets none, it is the default visual of the space of a group on its sites, and the plain
        visual otherwise. The size of a group on parts of a space without a size of its own is the number of parts of
        its space, or else of the space it inherits. The values of a group on sites take the shape of that space.

        Args:
            context:
                The space the group inherits from its system, or `None`.

        Raises:
            SpaceError:
                The group lives on parts but has neither a size nor a space, the space does not have the part, or the
                part of the space is unknown.

            ValidationError:
                A per-member value does not have one row per member. The element type is not an element class, the size
                is not a positive integer, or the rank does not belong to a variable group on hyperedges or is below 2.

            ValueError:
                A value names an argument the constructor of the element type does not take.
        """
        super()._prepare(context)
        self._check_on()
        self._check_element()
        self._check_size()
        self._check_rank()
        member = self._get_member()
        if member.visual is not None:
            self.default_visual_ = member.visual
        self.size_ = self._get_size_of(context)
        space = self._get_site_space(context)
        self.values_ = self._flatten_values(space)
        self._check_value_rows()
        if space is not None and self.default_visual_ == VisualSpec():
            self.default_visual_ = VisualSpec(space.default_visual_ or 'sites')

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the laws of the element type for a namespace of symbols.

        The laws are written for one member.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of a member.
        """
        return self._get_member()._get_laws(namespace)
