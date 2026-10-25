"""Replicate a system of single elements into groups that run many initial conditions or parameters at once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from ..base import Trait, ValidationError, has_trait
from ..components import ElementComponent, GroupComponent, probe_laws
from ._systems import System


def _has_agent_laws(element: ElementComponent) -> bool:
    """Check whether the laws of an element make it an agent."""
    return any(has_trait(law, Trait.AGENT) for law in probe_laws(element) or [])


def _check_plain(system: System) -> None:
    """Raise an error unless a system holds single elements alone."""
    plain = all(isinstance(item, ElementComponent) and not _has_agent_laws(item) for item in system.elements)
    if system.interactions or system.systems or not plain:
        error_msg = (
            f'Only systems of plain single elements replicate, but {system.label!r} has interactions, child systems, '
            'groups, fields, or agents. Build the ensemble with Element.group and per-member values instead.'
        )
        raise ValidationError(error_msg)


def _member_values(
    values: Mapping[str, npt.ArrayLike],
    system: System,
    size: int,
) -> dict[str, dict[str, Any]]:
    """Return the per-member values of every element by its label, checked against the elements and the size."""
    by_label: dict[str, dict[str, Any]] = {element.label: {} for element in system.elements}
    names = {element.label: set(element.get_params()) - {'label'} for element in system.elements}
    for path, value in values.items():
        label, _, name = path.partition('.')
        if label not in names or name not in names[label]:
            available = ', '.join(f'{item}.{field}' for item, fields in names.items() for field in sorted(fields))
            error_msg = f'{path} is not a state or parameter of the system. Available paths are: {available}.'
            raise ValidationError(error_msg)
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != size:
            error_msg = f'The values of {path} have shape {array.shape}, but they need one row per member, {size} rows.'
            raise ValidationError(error_msg)
        by_label[label][name] = array
    return by_label


def replicate(system: System, size: int, values: Mapping[str, npt.ArrayLike] | None = None) -> System:
    """Return a system whose single elements become groups of members of the same types and labels.

    Every member starts from the values of its element, except the dotted paths given in `values`, which take one
    value per member. A dotted path such as `map.x` names the same state in the replicated system, where it holds one
    value per member, so many initial conditions or parameter values run as one vectorised simulation.

    Args:
        system:
            The system of single elements, without interactions or child systems.

        size:
            The number of members of every group.

        values:
            A mapping from dotted path of a state or parameter to its per-member values, one row per member.

    Returns:
        system:
            The replicated system, with the observables of the original.

    Raises:
        ValidationError:
            The system has interactions, child systems, groups, fields, or agents, a path is not a state or
            parameter, or its values do not have one row per member.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.system import replicate
        >>> from skcomplex.models.mathematics.dynamical_systems import logistic_map
        >>> ensemble = replicate(logistic_map(r=3.7), 4, {'map.x': np.linspace(0.1, 0.4, 4)})
        >>> group = ensemble.elements[0]
        >>> group.label, group.element.__name__, group.size
        ('map', 'LogisticMapComponent', 4)
    """
    _check_plain(system)
    by_label = _member_values(values or {}, system, size)
    groups = [
        GroupComponent(
            element.label,
            element=type(element),
            size=size,
            values={
                **{name: value for name, value in element.get_params().items() if name != 'label'},
                **by_label[element.label],
            },
        )
        for element in system.elements
    ]
    return System(system.label, components=groups, space=system.space, time=system.time)
