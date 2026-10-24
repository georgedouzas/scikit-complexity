"""Flatten a system tree into paths and validate its labels and values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ..base import (
    AnyDomain,
    Domain,
    LabelError,
    SpaceError,
    TimeKind,
    Trait,
    ValidationError,
    Variable,
)
from ..components import Element, Group, probe_laws
from ..interactions import Interaction
from ..laws import has_trait
from ..space import LatticeSpace, NetworkSpace, Space, part_count
from ..system import System

LABEL_PATTERN = re.compile(r'^[A-Za-z0-9_\-]+$')
MAX_SHOWN_COMPONENTS = 10


def _check_label(label: object, scope: str) -> None:
    """Raise an error for a label that is not a non-empty word."""
    if isinstance(label, str) and LABEL_PATTERN.match(label):
        return
    where = scope or 'the root system'
    error_msg = (
        f'Label {label!r} in {where} is not valid. A label must be a non-empty string of letters, digits, '
        'underscores, and hyphens.'
    )
    raise LabelError(error_msg)


def _check_network_lattice(space: NetworkSpace, size: int, where: str) -> None:
    """Raise an error for a lattice group on a network with a capacity or a size other than its number of nodes."""
    if space.capacity is not None:
        error_msg = (
            f'{where} has its sites on a network with capacity {space.capacity}, but sites hold states, not agents. '
            'Leave the capacity as None.'
        )
        raise ValidationError(error_msg)
    if space.graph.number_of_nodes() != size:
        error_msg = f'{where} has size {size}, but its network has {space.graph.number_of_nodes()} nodes.'
        raise ValidationError(error_msg)


def _check_group(group: Group, scope: str) -> None:
    """Raise an error for a group with an element type or size that is not valid."""
    where = f'Group {group.label!r} in {scope or "the root system"}'
    if group.size is None and group.on is not None:
        return
    space = getattr(group, 'space', None)
    if isinstance(space, LatticeSpace):
        try:
            space.check()
        except ValueError as error:
            error_msg = f'{where} has a lattice that is not valid. {error}'
            raise ValidationError(error_msg) from error
        if space.size != group.size:
            error_msg = f'{where} has size {group.size}, but its lattice has {space.size} sites.'
            raise ValidationError(error_msg)
    if (
        isinstance(space, NetworkSpace)
        and not group.variable
        and group.on in {None, 'sites'}
        and group.size is not None
    ):
        _check_network_lattice(space, group.size, where)


def _check_children(system: System, scope: str) -> None:
    """Raise an error for a child of the wrong type or a duplicated label."""
    children: list[tuple[Element | Group | Interaction | System, type | tuple[type, ...]]] = [
        *((element, (Element, Group)) for element in system.elements),
        *((interaction, Interaction) for interaction in system.interactions),
        *((child, System) for child in system.systems),
    ]
    labels: set[str] = set()
    for child, expected_type in children:
        if not isinstance(child, expected_type):
            names = ' or '.join(
                kind.__name__ for kind in (expected_type if isinstance(expected_type, tuple) else (expected_type,))
            )
            error_msg = f'{child!r} in {scope or "the root system"} is not an instance of {names}.'
            raise ValidationError(error_msg)
        if isinstance(child, Group):
            _check_group(child, scope)
        _check_label(child.label, scope)
        if child.label in labels:
            error_msg = f'Label {child.label!r} is used by more than one child of {scope or "the root system"}.'
            raise LabelError(error_msg)
        labels.add(child.label)


def _sized(element: Element | Group, space: Space | None) -> Element | Group:
    """Return a group on parts of the space it inherits with its size counted from them, or the element itself."""
    if not isinstance(element, Group) or element.size is not None:
        return element
    if space is None:
        error_msg = f'Group {element.label!r} lives on {element.on} but has neither a size nor a space.'
        raise SpaceError(error_msg)
    sized = copy.copy(element)
    sized.size = part_count(space, element.on or 'sites', element.label, element.layer)
    return sized


def _with_capacity(element: Element | Group, space: Space | None) -> tuple[Element | Group, int | None]:
    """Return a variable group on a network with its capacity as its size, and its initial size, or the element."""
    if not is_topology_group(element, space):
        return element, None
    group = cast('Group', element)
    initial = int(group.size) if group.size is not None else 0
    capacity = group.capacity if group.capacity is not None else max(2 * initial, 1)
    if capacity < initial:
        error_msg = (
            f'Group {group.label!r} has the capacity {capacity}, below its {initial} initial members. Give a capacity '
            'of at least its size.'
        )
        raise ValidationError(error_msg)
    sized = copy.copy(group)
    sized.size = capacity
    return sized, initial


def _canonical(space: Space | None, seen: list[Space]) -> Space | None:
    """Return the first network space seen that equals a space, so clones of one network share one space."""
    if not isinstance(space, NetworkSpace):
        return space
    for known in seen:
        if isinstance(known, NetworkSpace) and known.equals(space):
            return known
    seen.append(space)
    return space


@dataclass(frozen=True)
class Node:
    """Element or interaction with its place in the system tree.

    Args:
        path:
            The slash-separated path of the component.

        scope:
            The path prefix of the system that holds the component.

        component:
            The element, group, or interaction.

        space:
            The nearest enclosing space.

        agent:
            Whether the component runs on the agent stepper, as a pass of a formalism decides.

        active:
            The initial number of members of a variable group on the sites or edges of a network, whose capacity is
            its size, or `None` for every other component.
    """

    path: str
    scope: str
    component: Element | Group | Interaction
    space: Space | None
    agent: bool = False
    active: int | None = None

    @property
    def members(self) -> int | None:
        """Size of a group, or `None` for a single element or an interaction."""
        return self.component.size if isinstance(self.component, Group) else None

    @property
    def is_agent(self) -> bool:
        """Whether the node is an agent element or an agent group, which runs on the agent stepper."""
        return self.agent

    @property
    def component_type(self) -> type[Element] | type[Interaction]:
        """Element type of a group, or the type of an element or interaction."""
        return self.component.element if isinstance(self.component, Group) else type(self.component)

    def value_of(self, variable: str) -> Any:  # noqa: ANN401
        """Return the value of a variable, or the element type's default for a group that does not set it.

        Args:
            variable:
                The name of the variable.

        Returns:
            value:
                The value given by the user or the default.
        """
        if not isinstance(self.component, Group):
            return getattr(self.component, variable)
        values = self.component.values or {}
        default = self.component.element.get_default(variable)
        if variable not in values:
            return default
        value = values[variable]
        space = getattr(self.component, 'space', None)
        if isinstance(space, LatticeSpace) and np.shape(value) == space.dimensions:
            return np.ravel(value)
        if self.active is not None and np.ndim(value) and len(value) == self.active:
            rows = (self.members or 0) - self.active
            padding = np.broadcast_to(np.asarray(default, dtype=float), (rows, *np.shape(value)[1:]))
            return np.concatenate([np.asarray(value, dtype=float), padding])
        return value


def _flatten(system: System, scope: str, space: Space | None, seen: list[Space] | None = None) -> list[Node]:
    """Return the nodes of a system tree in tree order."""
    if not isinstance(system, System):
        error_msg = f'{system!r} is not an instance of System.'
        raise ValidationError(error_msg)
    _check_children(system, scope)
    seen = [] if seen is None else seen
    space = _canonical(system.space if system.space is not None else space, seen)
    nodes = []
    for element in system.elements:
        place = _canonical(getattr(element, 'space', None) or space, seen)
        sized, active = _with_capacity(_sized(element, space), place)
        nodes.append(Node(f'{scope}{element.label}', scope, sized, place, active=active))
    nodes += [Node(f'{scope}{interaction.label}', scope, interaction, space) for interaction in system.interactions]
    for child in system.systems:
        nodes += _flatten(child, f'{scope}{child.label}/', space, seen)
    return nodes


def _describe(value: object) -> str:
    """Return the repr of a small value, or its shape for an array with many components."""
    if isinstance(value, np.ndarray | list | tuple) and np.size(value) > MAX_SHOWN_COMPONENTS:
        return f'an array of shape {np.shape(value)}'
    return repr(value)


def _coerce_group_value(
    name: str,
    value: object,
    array: npt.NDArray[Any],
    shape: tuple[int, ...],
    members: int,
) -> npt.NDArray[Any]:
    """Return a group value as flat member-major components, broadcasting shared values."""
    if array.shape in ((), shape):
        return np.broadcast_to(array, (members, *shape)).ravel().copy()
    if array.shape == (members, *shape):
        return array.ravel()
    count = array.shape[0] if array.ndim else 1
    error_msg = (
        f'{name} = {_describe(value)} has {count} values of shape {array.shape[1:]} for a group of {members} members '
        f'of shape {shape}. Give one value per member or one value shared by every member.'
    )
    raise ValidationError(error_msg)


def _flat_value(
    name: str,
    value: object,
    array: npt.NDArray[Any],
    shape: tuple[int, ...],
    members: int | None,
) -> npt.NDArray[Any]:
    """Return an encoded value as flat components of the given shape, member after member for a group."""
    if members is not None:
        return _coerce_group_value(name, value, array, shape, members)
    if not shape and array.ndim == 0:
        return array.reshape(1)
    if shape and array.ndim == 0:
        return np.full(shape, array.item(), dtype=array.dtype).ravel()
    if array.shape != shape:
        error_msg = f'{name} = {_describe(value)} has shape {array.shape}, but the variable has shape {shape}.'
        raise ValidationError(error_msg)
    return array.ravel()


def is_topology_group(component: object, space: Space | None) -> bool:
    """Check whether a component is a variable group on the sites or edges of a network without agent laws of its own.

    Compiled backends run such a group with a capacity layout, whose rows hold the members present and room for new
    ones, so laws that change the topology of the network add and remove its members.

    Args:
        component:
            The element, group, or interaction.

        space:
            The space of the component.

    Returns:
        topology_group:
            Whether the group runs with a capacity layout.
    """
    if not (isinstance(component, Group) and component.variable and component.on in {'sites', 'edges'}):
        return False
    if not isinstance(space, NetworkSpace):
        return False
    return not any(has_trait(law, Trait.AGENT) for law in probe_laws(component) or [])


def flatten_system(system: System, scope: str = '', space: Space | None = None) -> list[Node]:
    """Return the elements and interactions of a system tree with their paths.

    Args:
        system:
            The root of the tree.

        scope:
            The path prefix of the system.

        space:
            The space inherited from the parent system.

    Returns:
        nodes:
            The elements and interactions, in tree order, none of them marked as an agent yet.

    Raises:
        ValidationError:
            A child has the wrong type.

        LabelError:
            A label is not valid or is duplicated among siblings.
    """
    return _flatten(system, scope, space)


def declared_times(system: System, scope: str = '', inherited: TimeKind | None = None) -> dict[str, TimeKind | None]:
    """Return the declared time of every system of a tree by path, a nested system inheriting its parent's.

    Args:
        system:
            The root of the tree.

        scope:
            The path of the system.

        inherited:
            The time of the parent system.

    Returns:
        times:
            The time of every system by path, `None` for a system without a declared time.
    """
    time = system.time.kind if system.time is not None else inherited
    times = {scope: time}
    for child in system.systems:
        times.update(declared_times(child, f'{scope}/{child.label}' if scope else child.label, time))
    return times


def shape_of(name: str, declaration: Variable, space: Space | None) -> tuple[int, ...]:
    """Return the shape of a declared variable in its space.

    Args:
        name:
            The dotted name of the variable.

        declaration:
            The declaration of the variable.

        space:
            The nearest enclosing space.

    Returns:
        shape:
            `()` for a scalar and `(n_dim,)` for a vector.

    Raises:
        SpaceError:
            A vector variable has no enclosing space.

        ValidationError:
            The space dimension is not a positive integer.
    """
    if not declaration.vector:
        return ()
    if space is None:
        error_msg = f'{name} is a vector, so the system that holds it or one of its parents must have a space.'
        raise SpaceError(error_msg)
    if not Domain.POSITIVE.contains(space.n_dim) or not Domain.INTEGER.contains(space.n_dim):
        error_msg = f'The space of {name} has n_dim = {space.n_dim!r}, which is not a positive integer.'
        raise ValidationError(error_msg)
    return (int(space.n_dim),)


def coerce_value(
    name: str,
    value: object,
    shape: tuple[int, ...],
    domain: AnyDomain,
    members: int | None = None,
) -> npt.NDArray[np.float64]:
    """Return a value as a flat float array of the given shape after validating it.

    Args:
        name:
            The dotted name of the variable.

        value:
            The value given by the user.

        shape:
            The shape of the variable.

        domain:
            The domain of the variable.

        members:
            The size of the group that holds the variable, or `None` for a single element or an interaction. A group
            value is a per-member array with this first dimension, or one value shared by every member.

    Returns:
        array:
            The flat components of the value, member after member for a group, with the real and imaginary part of
            every component in a row for the complex domain.

    Raises:
        ValidationError:
            The value is not numeric, has the wrong shape, or is outside the domain.
    """
    if not domain.contains(value):
        error_msg = f'{name} = {_describe(value)} is not in the {domain.value} domain.'
        if Domain.COMPLEX.contains(value):
            error_msg = (
                f'{name} = {_describe(value)} has a nonzero imaginary part, which the {domain.value} domain does not '
                f'hold. Declare {name.rpartition(".")[2]} with `domain=Domain.COMPLEX` to give it complex values.'
            )
        raise ValidationError(error_msg)
    flat = _flat_value(name, value, domain.encode(value), shape, members)
    return np.ascontiguousarray(flat).view(float) if domain is Domain.COMPLEX else flat
