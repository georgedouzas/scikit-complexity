"""Flatten a system tree into paths and validate its labels and values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC, ZZ, oo
from sage.sets.real_set import RealSet

from ...core.elements import Element, GroupElement
from ...core.errors import LabelError, RoleError, SkcomplexError, SpaceError, ValidationError
from ...core.interactions import BaseInteraction
from ...core.spaces import BaseSpace, LatticeSpace, NetworkSpace
from ...core.specs import WienerProcessSpec
from ...core.system import System
from ..problems import Variable, VariableKind, contains, declarations_of, default_of, encode
from ._laws import _AGENT_LAWS
from ._probe import probe_laws

_LABEL_PATTERN = re.compile(r'^[A-Za-z0-9_\-]+$')
_MAX_SHOWN_COMPONENTS = 10


def _check_label(label: object, scope: str) -> None:
    """Raise an error for a label that is not a non-empty word."""
    if isinstance(label, str) and _LABEL_PATTERN.match(label):
        return
    where = scope or 'the root system'
    error_msg = (
        f'Label {label!r} in {where} is not valid. A label must be a non-empty string of letters, digits, '
        'underscores, and hyphens.'
    )
    raise LabelError(error_msg)


def _check_network_lattice(space: NetworkSpace, size: int | None, where: str) -> None:
    """Raise an error for a lattice group on a network with a capacity or a size other than its number of nodes."""
    if space.capacity is not None:
        error_msg = (
            f'{where} has its sites on a network with capacity {space.capacity}, but sites hold states, not agents. '
            'Leave the capacity as None.'
        )
        raise ValidationError(error_msg)
    if size is not None and space.graph.number_of_nodes() != size:
        error_msg = f'{where} has size {size}, but its network has {space.graph.number_of_nodes()} nodes.'
        raise ValidationError(error_msg)


def _check_group(group: GroupElement, scope: str) -> None:
    """Raise an error for a group with an element type or size that is not valid."""
    where = f'Group {group.label!r} in {scope or "the root system"}'
    space = getattr(group, 'space', None)
    if isinstance(space, NetworkSpace) and not group.variable and group.on in {None, 'sites'}:
        _check_network_lattice(space, group.size, where)
    if group.size is None:
        return
    if isinstance(space, LatticeSpace):
        try:
            space.check().prepare()
        except ValidationError as error:
            error_msg = f'{where} has a lattice that is not valid. {error}'
            raise ValidationError(error_msg) from error
        if space.size_ != group.size:
            error_msg = f'{where} has size {group.size}, but its lattice has {space.size_} sites.'
            raise ValidationError(error_msg)


def _check_part(part: Element | GroupElement | BaseInteraction, scope: str) -> None:
    """Check the arguments of an element, group, or interaction, raising its error with the path of the part."""
    try:
        part.check()
    except SkcomplexError as error:
        error_msg = f'{scope}{part.label}: {error}'
        raise type(error)(error_msg) from error


def _bound_processes(interaction: BaseInteraction, scope: str) -> list[WienerProcessSpec]:
    """Return the Wiener processes that the noise roles of an interaction bind, after checking every role."""
    processes = []
    for name, declaration in declarations_of(type(interaction), VariableKind.ROLE).items():
        value = getattr(interaction, name)
        if declaration.noise != isinstance(value, WienerProcessSpec):
            expected = 'a WienerProcessSpec' if declaration.noise else 'an element or group path'
            error_msg = (
                f'Interaction {scope}{interaction.label} binds its role {name!r} to {value!r}, but the role takes '
                f'{expected}. Bind {expected}, and declare a role of Wiener processes with `role(noise=True)`.'
            )
            raise RoleError(error_msg)
        if declaration.noise:
            processes.append(value)
    return processes


def _same_process(first: WienerProcessSpec, second: WienerProcessSpec) -> bool:
    """Check whether two Wiener processes of one label have the same dimension and covariance."""
    if first.dimension != second.dimension:
        return False
    return bool(np.array_equal(np.asarray(first.covariance, dtype=object), np.asarray(second.covariance, dtype=object)))


def _check_processes(system: System, scope: str, labels: set[str]) -> None:
    """Raise an error for Wiener processes of one label that differ, or whose label another child of a system uses."""
    found: dict[str, WienerProcessSpec] = {}
    for interaction in system.interactions:
        for process in _bound_processes(interaction, scope):
            _check_label(process.label, scope)
            known = found.setdefault(process.label, process)
            if not _same_process(known, process):
                error_msg = (
                    f'The interactions of {scope or "the root system"} bind two different Wiener processes labelled '
                    f'{process.label!r}. Bind one process to every interaction that shares it, or give them two labels.'
                )
                raise LabelError(error_msg)
    clashes = sorted(set(found) & labels)
    if clashes:
        error_msg = (
            f'The Wiener process {clashes[0]!r} has the label of another child of {scope or "the root system"}. Give '
            'it a label of its own.'
        )
        raise LabelError(error_msg)


def _check_children(system: System, scope: str) -> None:
    """Raise an error for a child of the wrong type or a duplicated label."""
    children: list[tuple[Element | GroupElement | BaseInteraction | System, type | tuple[type, ...]]] = [
        *((element, (Element, GroupElement)) for element in system.elements),
        *((interaction, BaseInteraction) for interaction in system.interactions),
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
        _check_label(child.label, scope)
        if not isinstance(child, System):
            _check_part(child, scope)
        if isinstance(child, GroupElement):
            _check_group(child, scope)
        if child.label in labels:
            error_msg = f'Label {child.label!r} is used by more than one child of {scope or "the root system"}.'
            raise LabelError(error_msg)
        labels.add(child.label)
    _check_processes(system, scope, labels)


def _sized(element: Element | GroupElement, space: BaseSpace | None) -> Element | GroupElement:
    """Return a group prepared with its size, a copy when the size counts the parts of the space it inherits."""
    if not isinstance(element, GroupElement):
        return element
    if element.size is None and element.space is None:
        return copy.copy(element).prepare(space)
    return element.check().prepare(space)


def _with_capacity(
    element: Element | GroupElement,
    space: BaseSpace | None,
) -> tuple[Element | GroupElement, int | None]:
    """Return a variable group on a network with its capacity as its size, and its initial size, or the element."""
    if not _is_topology_group(element, space):
        return element, None
    group = cast('GroupElement', element)
    initial = group.size_
    capacity = group.capacity if group.capacity is not None else max(2 * initial, 1)
    if capacity < initial:
        error_msg = (
            f'Group {group.label!r} has the capacity {capacity}, below its {initial} initial members. Give a capacity '
            'of at least its size.'
        )
        raise ValidationError(error_msg)
    sized = copy.copy(group)
    sized.size_ = capacity
    return sized, initial


def _same_network(first: NetworkSpace, second: NetworkSpace) -> bool:
    """Check whether two prepared network spaces have the same nodes, edges, and layers, such as clones of a network."""
    if list(first.graphs_) != list(second.graphs_):
        return False
    if (first.capacity, first.fill, list(first.hyperedges)) != (second.capacity, second.fill, list(second.hyperedges)):
        return False
    return all(
        list(graph.nodes) == list(second.graphs_[layer].nodes)
        and list(graph.edges) == list(second.graphs_[layer].edges)
        and graph.is_directed() == second.graphs_[layer].is_directed()
        for layer, graph in first.graphs_.items()
    )


def _canonical(space: BaseSpace | None, seen: list[BaseSpace]) -> BaseSpace | None:
    """Return a prepared space, or the first network space seen that equals it, so clones of a network share one.

    A space is prepared here unless it was prepared before, such as a group space whose ball was enumerated once. A
    space whose arguments are not valid stays unprepared, and validation reports it from its check.
    """
    if space is None:
        return None
    for known in seen:
        if known is space:
            return known
    try:
        if not hasattr(space, 'size_'):
            space.check().prepare()
    except ValidationError:
        seen.append(space)
        return space
    for known in seen:
        if isinstance(known, NetworkSpace) and isinstance(space, NetworkSpace) and _same_network(known, space):
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
            The element, group, or interaction, or the Wiener processes that the noise role of an interaction binds.

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
    component: Element | GroupElement | BaseInteraction | WienerProcessSpec
    space: BaseSpace | None
    agent: bool = False
    active: int | None = None

    @property
    def members(self) -> int | None:
        """Size of a group, or `None` for a single element or an interaction."""
        return self.component.size_ if isinstance(self.component, GroupElement) else None

    @property
    def is_agent(self) -> bool:
        """Whether the node is an agent element or an agent group, which runs on the agent stepper."""
        return self.agent

    @property
    def component_type(self) -> type[Element] | type[BaseInteraction]:
        """Element type of a group, or the type of an element or interaction."""
        return self.component.element if isinstance(self.component, GroupElement) else type(self.component)

    def value_of(self, variable: str) -> Any:  # noqa: ANN401
        """Return the value of a variable, or the element type's default for a group that does not set it.

        Args:
            variable:
                The name of the variable.

        Returns:
            value:
                The value given by the user or the default.
        """
        if not isinstance(self.component, GroupElement):
            return (
                getattr(self.component, variable)
                if hasattr(self.component, variable)
                else default_of(self.component, variable)
            )
        values = self.component.values_
        default = default_of(self.component.element, variable)
        if variable not in values:
            return default
        value = values[variable]
        space = getattr(self.component, 'space', None)
        if isinstance(space, LatticeSpace) and np.shape(value) == space.shape_:
            return np.ravel(value)
        if self.active is not None and np.ndim(value) and len(value) == self.active:
            rows = (self.members or 0) - self.active
            padding = np.broadcast_to(np.asarray(default, dtype=float), (rows, *np.shape(value)[1:]))
            return np.concatenate([np.asarray(value, dtype=float), padding])
        return value


def _flatten(system: System, scope: str, space: BaseSpace | None, seen: list[BaseSpace] | None = None) -> list[Node]:
    """Return the nodes of a system tree in tree order."""
    if not isinstance(system, System):
        error_msg = f'{system!r} is not an instance of System.'
        raise ValidationError(error_msg)
    seen = [] if seen is None else seen
    space = _canonical(system.space if system.space is not None else space, seen)
    places = [_canonical(getattr(element, 'space', None) or space, seen) for element in system.elements]
    _check_children(system, scope)
    nodes = []
    for element, place in zip(system.elements, places, strict=True):
        element.check().prepare(place)
        sized, active = _with_capacity(_sized(element, space), place)
        nodes.append(Node(f'{scope}{element.label}', scope, sized, place, active=active))
    nodes += [
        Node(f'{scope}{interaction.label}', scope, interaction.check().prepare(), space)
        for interaction in system.interactions
    ]
    for child in system.systems:
        nodes += _flatten(child, f'{scope}{child.label}/', space, seen)
    return nodes


def _describe(value: object) -> str:
    """Return the repr of a small value, or its shape for an array with many components."""
    if isinstance(value, np.ndarray | list | tuple) and np.size(value) > _MAX_SHOWN_COMPONENTS:
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


def _noise_processes(nodes: list[Node]) -> dict[str, WienerProcessSpec]:
    """Return the Wiener processes that the noise roles of the interactions of a system tree bind, by path.

    Interactions of one system that bind processes of one label share them, at the path of the label in that system.

    Args:
        nodes:
            The nodes of the system tree.

    Returns:
        processes:
            The Wiener processes by path, in the order the interactions first bind them.
    """
    processes: dict[str, WienerProcessSpec] = {}
    for node in nodes:
        if not isinstance(node.component, BaseInteraction):
            continue
        for name, declaration in declarations_of(type(node.component), VariableKind.ROLE).items():
            value = getattr(node.component, name)
            if declaration.noise and isinstance(value, WienerProcessSpec):
                processes.setdefault(f'{node.scope}{value.label}', value)
    return processes


def _is_topology_group(component: object, space: BaseSpace | None) -> bool:
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
    if not (isinstance(component, GroupElement) and component.variable and component.on in {'sites', 'edges'}):
        return False
    if not isinstance(space, NetworkSpace):
        return False
    return not any(isinstance(law, _AGENT_LAWS) for law in probe_laws(component) or [])


def flatten_system(system: System, scope: str = '', space: BaseSpace | None = None) -> list[Node]:
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
            A child has the wrong type, or a component or interaction does not pass its check, named by its path.

        LabelError:
            A label is not valid or is duplicated among siblings.
    """
    return _flatten(system, scope, space)


def _declared_times(system: System, scope: str = '', inherited: type | None = None) -> dict[str, type | None]:
    """Return the class of the declared time of every system of a tree by path, a nested system inheriting its parent's.

    Args:
        system:
            The root of the tree.

        scope:
            The path of the system.

        inherited:
            The time of the parent system.

    Returns:
        times:
            The time class of every system by path, such as `ContinuousTime`, or `None` for a system without one.
    """
    time = type(system.time) if system.time is not None else inherited
    times = {scope: time}
    for child in system.systems:
        times.update(_declared_times(child, f'{scope}/{child.label}' if scope else child.label, time))
    return times


def _shape_of(name: str, declaration: Variable, space: BaseSpace | None) -> tuple[int, ...]:
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
    if not contains(RealSet.open(0, oo), space.n_dim_) or not contains(ZZ, space.n_dim_):
        error_msg = f'The space of {name} has n_dim = {space.n_dim_!r}, which is not a positive integer.'
        raise ValidationError(error_msg)
    return (int(space.n_dim_),)


def _coerce_value(
    name: str,
    value: object,
    shape: tuple[int, ...],
    domain: object,
    members: int | None = None,
    *,
    period: object = None,
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

        period:
            The period the values wrap around, or `None`.

    Returns:
        array:
            The flat components of the value, member after member for a group, with the real and imaginary part of
            every component in a row for the complex domain.

    Raises:
        ValidationError:
            The value is not numeric, has the wrong shape, or is outside the domain.
    """
    if not contains(domain, value):
        error_msg = f'{name} = {_describe(value)} is not in its domain {domain}.'
        if contains(CC, value):
            error_msg = (
                f'{name} = {_describe(value)} has a nonzero imaginary part, which its domain {domain} does not hold. '
                f'Declare {name.rpartition(".")[2]} with `StateSpec(domain=CC)` to give it complex values.'
            )
        raise ValidationError(error_msg)
    flat = _flat_value(name, value, encode(domain, value, period), shape, members)
    return np.ascontiguousarray(flat).view(float) if domain is CC else flat
