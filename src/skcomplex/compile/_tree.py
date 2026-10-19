"""Flatten a system tree into paths and validate its labels and values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..core import AnyDomain, Domain, LabelError, SpaceError, ValidationError, Variable
from ..model import AgentType, Element, Group, Interaction, LatticeSpace, MesaSystem, Space, System

LABEL_PATTERN = re.compile(r'^[A-Za-z0-9_\-]+$')
MAX_SHOWN_COMPONENTS = 10


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
    """

    path: str
    scope: str
    component: Element | Group | Interaction
    space: Space | None

    @property
    def members(self) -> int | None:
        """Size of a group, or `None` for a single element or an interaction."""
        return self.component.size if isinstance(self.component, Group) else None

    @property
    def is_agent(self) -> bool:
        """Whether the node is an agent type or an agent group."""
        return isinstance(self.component, Element | Group) and issubclass(self.component_type, AgentType)

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
        if variable in values:
            value = values[variable]
            space = getattr(self.component, 'space', None)
            if isinstance(space, LatticeSpace) and np.shape(value) == space.dimensions:
                return np.ravel(value)
            return value
        field = next(field for field in dataclasses.fields(self.component.element) if field.name == variable)
        factory = field.default_factory
        return factory() if callable(factory) else field.default


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


def _check_group(group: Group, scope: str) -> None:
    """Raise an error for a group with an element type or size that is not valid."""
    where = f'Group {group.label!r} in {scope or "the root system"}'
    if not (isinstance(group.element, type) and issubclass(group.element, Element)):
        error_msg = f'{where} has element {group.element!r}, which is not an Element subclass.'
        raise ValidationError(error_msg)
    if not (Domain.POSITIVE.contains(group.size) and Domain.INTEGER.contains(group.size)):
        error_msg = f'{where} has size {group.size!r}, which is not a positive integer.'
        raise ValidationError(error_msg)
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
    unknown = set(group.values or {}) - {field.name for field in dataclasses.fields(group.element)} - {'label'}
    if unknown:
        error_msg = f'{where} sets {", ".join(sorted(unknown))}, which {group.element.__name__} does not declare.'
        raise ValidationError(error_msg)


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
            The elements and interactions, in tree order.

    Raises:
        ValidationError:
            A child has the wrong type.

        LabelError:
            A label is not valid or is duplicated among siblings.
    """
    if not isinstance(system, System):
        error_msg = f'{system!r} is not an instance of System.'
        raise ValidationError(error_msg)
    if isinstance(system, MesaSystem):
        error_msg = f'{system!r} in {scope or "the root system"} can only be compiled as the root system.'
        raise ValidationError(error_msg)
    _check_children(system, scope)
    space = system.space if system.space is not None else space
    nodes = [
        Node(f'{scope}{element.label}', scope, element, getattr(element, 'space', None) or space)
        for element in system.elements
    ]
    nodes += [Node(f'{scope}{interaction.label}', scope, interaction, space) for interaction in system.interactions]
    for child in system.systems:
        nodes += flatten_system(child, f'{scope}{child.label}/', space)
    return nodes


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


def _describe(value: object) -> str:
    """Return the repr of a small value, or its shape for an array with many components."""
    if isinstance(value, np.ndarray | list | tuple) and np.size(value) > MAX_SHOWN_COMPONENTS:
        return f'an array of shape {np.shape(value)}'
    return repr(value)


def _coerce_group_value(
    name: str,
    value: object,
    array: npt.NDArray[np.float64],
    shape: tuple[int, ...],
    members: int,
) -> npt.NDArray[np.float64]:
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
            The flat components of the value, member after member for a group.

    Raises:
        ValidationError:
            The value is not numeric, has the wrong shape, or is outside the domain.
    """
    if not domain.contains(value):
        error_msg = f'{name} = {_describe(value)} is not in the {domain.value} domain.'
        raise ValidationError(error_msg)
    array = domain.encode(value)
    if members is not None:
        return _coerce_group_value(name, value, array, shape, members)
    if not shape and array.ndim == 0:
        return array.reshape(1)
    if shape and array.ndim == 0:
        return np.full(shape, float(array)).ravel()
    if array.shape != shape:
        error_msg = f'{name} = {_describe(value)} has shape {array.shape}, but the variable has shape {shape}.'
        raise ValidationError(error_msg)
    return array.ravel()
