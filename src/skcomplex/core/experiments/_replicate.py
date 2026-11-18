"""Replicate a system of single elements into groups, run as one simulation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import inspect
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from ..components.elements import BaseElement, DeclaredElement, Element, GroupElement
from ..components.laws import (
    BoundaryLaw,
    ConstraintLaw,
    EventLaw,
    JumpLaw,
    NoiseLaw,
    RateLaw,
    TransferLaw,
    TransitionLaw,
    UpdateLaw,
)
from ..components.systems import CompositeSystem
from ..engine._utils import declared_variables, probe_laws
from ..engine.problems import BaseProblemSpec
from ..engine.runtime import Trajectory, simulate
from ..enums import VariableKind
from ..errors import ValidationError
from ..specs import ParameterSpec, StateSpec, VariableSpec
from ..systems import BaseSystem, System
from ._runs import _check_source
from ._utils import build_system, get_problem
from .types import Source

PLAIN_LAWS = (
    RateLaw,
    NoiseLaw,
    JumpLaw,
    UpdateLaw,
    BoundaryLaw,
    EventLaw,
    TransitionLaw,
    TransferLaw,
    ConstraintLaw,
)


def _has_formalism_laws(element: Element) -> bool:
    """Return whether an element follows laws other than the plain ones.

    An example is the rules of agents.
    """
    return any(not isinstance(law, PLAIN_LAWS) for law in probe_laws(element) or [])


def _check_plain(system: CompositeSystem) -> None:
    """Raise an error for a system that is not only plain elements and interactions.

    The elements must be plain single elements.

    Args:
        system:
            The system.

    Raises:
        ValidationError:
            The system has child systems, groups, fields, or agents.
    """
    plain = all(isinstance(item, Element) and not _has_formalism_laws(item) for item in system.elements)
    if system.systems or not plain:
        error_msg = (
            f'Only systems of plain single elements and their interactions replicate, but {system.label!r} has child '
            'systems, groups, fields, or agents. Build the ensemble as a GroupElement with per-member values instead.'
        )
        raise ValidationError(error_msg)


def _is_declared_companion(element: Element, variable: VariableSpec) -> bool:
    """Return whether a variable is a companion of a state declared from data."""
    state = variable.derivative_of or variable.lag_of
    return state is not None and state in element._get_declared()


def _value_names(element: Element) -> list[str]:
    """Return the names of the states and parameters an element holds.

    They include the derivatives of higher order states.
    """
    return [
        name
        for name, (variable, _) in declared_variables(element).items()
        if variable.kind != VariableKind.ROLE
        and (hasattr(element, name) or name in element._get_declared() or _is_declared_companion(element, variable))
    ]


def _member_settings(element: Element) -> dict[str, object]:
    """Return what a member takes from an element.

    It takes its values, and the arguments that declare its variables.
    """
    if isinstance(element, DeclaredElement):
        settings = element.get_params(deep=False)
        return {name: value for name, value in settings.items() if name not in {'label', 'visual'}}
    return {name: getattr(element, name) for name in _value_names(element)}


def _member_values(
    values: Mapping[str, npt.ArrayLike],
    system: CompositeSystem,
    size: int,
) -> dict[str, dict[str, object]]:
    """Return the per-member values of every element by its label.

    They are checked against the elements and the size.

    Args:
        values:
            The per-member values by dotted path.

        system:
            The system.

        size:
            The number of members.

    Returns:
        values:
            The per-member values of every element by its label, by state or parameter name.

    Raises:
        ValidationError:
            A path is not a state or parameter of an element, or its values do not have one row per member.
    """
    by_label: dict[str, dict[str, object]] = {element.label: {} for element in system.elements}
    names = {element.label: set(_value_names(element)) for element in system.elements}
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


def _replicate(
    system: CompositeSystem | System,
    size: int,
    values: Mapping[str, npt.ArrayLike] | None = None,
) -> CompositeSystem:
    """Return a system whose single elements become groups of members.

    The members are of the same types and labels. Every member starts from the values of its element, except the dotted
    paths given in `values`, which take one value per member. A dotted path such as `map.x` names the same state in the
    replicated system, where it holds one value per member, so many initial conditions or parameter values run as one
    vectorised simulation. The interactions of the system bind the groups by the labels of their elements, so member `i`
    of every group interacts with member `i` of the others, and every member follows the original system from its own
    values.

    Args:
        system:
            The system of single elements and their interactions, without child systems. A `System` replicates as
            `_replicate_prepared` does, into the composite the engine compiles.

        size:
            The number of members of every group.

        values:
            A mapping from dotted path of a state or parameter to its per-member values, one row per member.

    Returns:
        system:
            The replicated system, with copies of the interactions of the original.

    Raises:
        ValidationError:
            The system has child systems, groups, fields, or agents, a path is not a state or parameter of an element,
            or its values do not have one row per member.
    """
    if isinstance(system, System):
        return _replicate_prepared(system, size, values)._get_composite()
    _check_plain(system)
    by_label = _member_values(values or {}, system, size)
    groups = [
        GroupElement(
            element.label,
            element=type(element),
            size=size,
            values={**_member_settings(element), **by_label[element.label]},
        )
        for element in system.elements
    ]
    return CompositeSystem(
        system.label,
        elements=groups,
        interactions=[copy.copy(interaction) for interaction in system.interactions],
        space=system.space,
        time=system.time,
    )


def _declares(element: Element) -> bool:
    """Return whether an element declares its states and parameters."""
    specs = type(element)._get_specs().values()
    return isinstance(element, DeclaredElement) or any(isinstance(spec, StateSpec | ParameterSpec) for spec in specs)


def _get_value_names(element: Element) -> set[str]:
    """Return the names of the values of a single element.

    They are its value arguments and the entries of `values`.
    """
    if _declares(element):
        return set(_value_names(element))
    return set(element._get_hints()) | set(element.values or {})


def _get_members(element: Element, size: int, values: Mapping[str, object]) -> BaseElement:
    """Return `size` members of a single element.

    The given values hold one value per member. A plain element whose class takes `size` becomes an element of many
    members, with the per-member values as its arguments or entries of `values`. Any other element, such as one that
    declares its states, becomes a group of members of its class.
    """
    if 'size' not in inspect.signature(type(element)).parameters or _declares(element):
        settings = {**_member_settings(element), **values}
        return GroupElement(element.label, element=type(element), size=size, values=settings)
    arguments = {name: value for name, value in values.items() if name in element._get_hints()}
    entries = {name: value for name, value in values.items() if name not in arguments}
    return clone(element).set_params(size=size, values={**(element.values or {}), **entries}, **arguments)


def _replicate_prepared(system: System, size: int, values: Mapping[str, npt.ArrayLike] | None = None) -> System:
    """Return a system whose single elements become elements of many members.

    It is the replication of `_replicate` for a `System`. Every element takes `size` members, which start from the
    values of the element, except the dotted paths given in `values`, which take one value per member.

    Args:
        system:
            The system of single elements and their interactions, without nested systems.

        size:
            The number of members of every element.

        values:
            A mapping from dotted path of a value of an element to its per-member values, one row per member.

    Returns:
        system:
            The replicated system, with clones of the interactions of the original.

    Raises:
        ValidationError:
            The system has nested systems or elements of many members, a path is not a value of an element, or its
            values do not have one row per member.
    """
    plain = all(
        isinstance(element, Element) and element.size is None and element.space is None and element.on is None
        for element in system.elements
    )
    if system.systems or not plain:
        error_msg = (
            f'Only systems of single elements and their interactions replicate, but {system.label!r} has nested '
            'systems or elements of many members. Build the ensemble as an element with `size=` instead.'
        )
        raise ValidationError(error_msg)
    names = {element.label: _get_value_names(element) for element in system.elements}
    by_label: dict[str, dict[str, object]] = {label: {} for label in names}
    for path, value in (values or {}).items():
        label, _, name = path.partition('.')
        if label not in names or name not in names[label]:
            available = ', '.join(f'{item}.{field}' for item, fields in names.items() for field in sorted(fields))
            error_msg = f'{path} is not a value of an element of the system. Available paths are: {available}.'
            raise ValidationError(error_msg)
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != size:
            error_msg = f'The values of {path} have shape {array.shape}, but they need one row per member, {size} rows.'
            raise ValidationError(error_msg)
        by_label[label][name] = array
    elements = [_get_members(element, size, by_label[element.label]) for element in system.elements]
    return System(
        system.label,
        elements=elements,
        interactions=[clone(interaction) for interaction in system.interactions],
        space=system.space,
        solver=system.solver,
        backend=system.backend,
    )


def _run_replicated(
    source: Source,
    values: Mapping[str, npt.ArrayLike],
    until: float | None,
    dt: float | None,
    random_state: int | None,
) -> tuple[Trajectory, BaseProblemSpec]:
    """Return one run of a replicated system, and the problem of the original.

    The system is replicated once per value of every path.

    Args:
        source:
            The system, or the factory of systems.

        values:
            The per-member values by dotted path.

        until:
            The end time of the run, or `None` for the tuned end time of the system.

        dt:
            The time between frames, or `None` for the tuned one.

        random_state:
            The seed of the run.

    Returns:
        trajectory:
            The trajectory of the replicated system.

        problem:
            The compiled problem of the original system, which reads one member of the trajectory.

    Raises:
        ValidationError:
            The system has no tuned end time and none is given.
    """
    system = build_system(_check_source(source), {})
    size = len(np.asarray(next(iter(values.values()))))
    if isinstance(system, BaseSystem):
        end, step = (system.until if until is None else until), (system.dt if dt is None else dt)
    else:
        settings = system.get_run_settings()
        end, step = (settings['until'] if until is None else until), (settings['dt'] if dt is None else dt)
    if end is None:
        error_msg = f'{type(system).__name__} has no tuned end time. Pass `until`.'
        raise ValidationError(error_msg)
    if isinstance(system, BaseSystem):
        original = system.prepare().system_
        if not isinstance(original, System):
            error_msg = f'{type(system).__name__} prepares no System, so it does not replicate.'
            raise ValidationError(error_msg)
        replicated = _replicate_prepared(original, size, values)
        trajectory = replicated.prepare().simulate(end, dt=step, random_state=random_state)
        return trajectory, get_problem(original)
    composite = system.prepare().system_
    replicated_composite = _replicate(composite, size, values)
    return simulate(replicated_composite, until=end, dt=step, random_state=random_state), get_problem(composite)
