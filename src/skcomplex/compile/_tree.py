"""Flatten a system tree into paths and validate its labels and values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..core import Domain, LabelError, SpaceError, ValidationError, Variable
from ..model import Element, Interaction, Space, System

LABEL_PATTERN = re.compile(r'^[A-Za-z0-9_\-]+$')


@dataclass(frozen=True)
class Node:
    """Element or interaction with its place in the system tree.

    Args:
        path:
            The slash-separated path of the component.

        scope:
            The path prefix of the system that holds the component.

        component:
            The element or interaction.

        space:
            The nearest enclosing space.
    """

    path: str
    scope: str
    component: Element | Interaction
    space: Space | None


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


def _check_children(system: System, scope: str) -> None:
    """Raise an error for a child of the wrong type or a duplicated label."""
    children: list[tuple[Element | Interaction | System, type]] = [
        *((element, Element) for element in system.elements),
        *((interaction, Interaction) for interaction in system.interactions),
        *((child, System) for child in system.systems),
    ]
    labels: set[str] = set()
    for child, expected_type in children:
        if not isinstance(child, expected_type):
            error_msg = f'{child!r} in {scope or "the root system"} is not an instance of {expected_type.__name__}.'
            raise ValidationError(error_msg)
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
    _check_children(system, scope)
    space = system.space if system.space is not None else space
    nodes = [Node(f'{scope}{element.label}', scope, element, space) for element in system.elements]
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


def coerce_value(name: str, value: object, shape: tuple[int, ...], domain: Domain) -> npt.NDArray[np.float64]:
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

    Returns:
        array:
            The flat components of the value.

    Raises:
        ValidationError:
            The value is not numeric, has the wrong shape, or is outside the domain.
    """
    if not domain.contains(value):
        error_msg = f'{name} = {value!r} is not in the {domain.value} domain.'
        raise ValidationError(error_msg)
    array = np.asarray(value, dtype=float)
    if not shape and array.ndim == 0:
        return array.reshape(1)
    if shape and array.ndim == 0:
        return np.full(shape, float(array)).ravel()
    if array.shape != shape:
        error_msg = f'{name} = {value!r} has shape {array.shape}, but the variable has shape {shape}.'
        raise ValidationError(error_msg)
    return array.ravel()
