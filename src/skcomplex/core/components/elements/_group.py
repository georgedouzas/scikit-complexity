"""Define groups of identical elements.

The variables of a group are arrays with one row per member.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import ZZ

from ...errors import SpaceError, ValidationError
from ...specs import RoleSpec, Spec, StateSpec, Tags, VisualSpec
from ..base import BasePart
from ..spaces import BaseSpace
from ._base import BaseElement
from ._declared import DeclaredElement

PARTS = ('sites', 'edges', 'hyperedges')
MIN_RANK = 2


class GroupElement(BaseElement):
    """Group of identical elements of one type.

    Its variables are arrays with one row per member. A value is either an array whose first dimension is the size of
    the group, or one value shared by every member. Variables without a value take the default of the constructor of the
    element type, and the laws of the group are the laws of its element type, written for one member.

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
            How the group is drawn, or `None` for `default_visual_`.

        space:
            The space whose sites the members occupy, such as a lattice with one member per site, or `None` for the
            space of the enclosing system.

        variable:
            Whether laws add and remove members during a run, such as the births and deaths of agents.

        on:
            The part of the space with one member each, `'sites'`, `'edges'`, or `'hyperedges'`, or `None` for members
            that are not tied to parts of the space. With a part and no size, the size is the number of parts.

        location:
            The initial places of agents in the space: cell coordinates, a network node, a position, or a polygon that
            holds the member at a uniform point inside it, one per member or one shared by every member, or `None` for
            random places. Only the agent stepper reads it.

        layer:
            The layer of the network whose edges a group on edges lives on, or `None` for the graph of the network.

        capacity:
            The largest number of members of a variable group on the sites or edges of a network, which compiled
            backends reserve rows for, or `None` for twice the initial size.

        rank:
            The largest number of members of a hyperedge of a variable group on the hyperedges of a network, the rank
            of its hypergraph, which compiled backends reserve a column per member for, so the group holds hyperedges
            of every size up to it, or `None` for one size, that of the initial hyperedges.

    Attributes:
        default_visual_:
            The visual of a member, or the plain visual when the member sets none, which the group takes when it sets
            none.

        size_:
            The number of members: the size, or for a group on parts without one, the number of parts of its space.

        values_:
            The values with one row per member, or one value shared by every member.

    Examples:
        >>> from typing import Annotated
        >>> import networkx as nx
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.spaces import NetworkSpace
        >>> from skcomplex.core.specs import LayerSpec
        >>> from skcomplex.core.specs import VisualSpec
        >>> class PopulationElement(Element):
        ...     def __init__(self, label, n: Annotated[float, StateSpec()] = 1.0, visual=VisualSpec('line')):
        ...         super().__init__(label, visual=visual)
        ...         self.n = n
        >>> GroupElement('herds', element=PopulationElement, size=3).prepare().size_
        3
        >>> ring = NetworkSpace(nx.cycle_graph(30))
        >>> GroupElement('weights', element=PopulationElement, space=ring, on='edges').prepare().size_
        30
        >>> space = NetworkSpace(nx.cycle_graph(4), layers=[LayerSpec('work', nx.complete_graph(4))])
        >>> wages = GroupElement('wages', element=PopulationElement, space=space, on='edges', layer='work')
        >>> wages.prepare().size_
        6
        >>> herds = GroupElement('herds', element=PopulationElement, size=3).prepare()
        >>> herds.default_visual_
        VisualSpec(kind='line', options={})
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

    def _size_of(self, space: BaseSpace | None = None) -> int:
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
            return self._get_member()._get_part_specs()
        return self.element._get_specs()

    def get_declaring_part(self) -> type[BaseElement] | BaseElement:
        """Return what declares the variables of a member.

        It is its class, or a member when it declares them from data.

        Returns:
            part:
                The element class, or a member with the values of the group.

        Examples:
            >>> from skcomplex.core.components.elements import DeclaredElement, GroupElement
            >>> from skcomplex.core.specs import DeclaredVariableSpec, StateSpec
            >>> values = {'variables': {'x': DeclaredVariableSpec(1.0, StateSpec())}}
            >>> group = GroupElement('cells', element=DeclaredElement, size=2, values=values)
            >>> group.get_declaring_part().get_value('x')
            1.0
        """
        return self._get_member() if issubclass(self.element, DeclaredElement) else self.element

    def get_states(self) -> dict[str, StateSpec]:
        """Return the states of a member of the group with their specs.

        Returns:
            states:
                A mapping from the name of every state to its spec.

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

    def _draws(self) -> bool:
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

    def _flatten_values(self) -> dict[str, object]:
        """Return the values with one row per member.

        A group takes them as given.
        """
        return dict(self.values or {})

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
        """Raise an error for a part that is not one of the parts of a space."""
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

        It first raises an error for arguments the group cannot take. They are a part, an element type, a size, a rank,
        or values.

        `default_visual_` is the visual of the group when it sets none. The default visual of a group is the visual of a
        member, or the plain visual when the member sets none. The size of a group on parts of a space without a size of
        its own is the number of parts of its space, or else of the space it inherits.

        Args:
            context:
                The space the group inherits from its system, or `None`.

        Raises:
            SpaceError:
                The group lives on parts but has neither a size nor a space, or the space does not have the part. The
                part is unknown.

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
        self.size_ = self._size_of(context)
        self.values_ = self._flatten_values()
        self._check_value_rows()

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
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
