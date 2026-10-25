"""Resolve the model source of an experiment into compiled problems with given initial values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...core.base import BaseProblem, ValidationError
from ...core.system import Observable, System
from ..compile import compile_system
from ..runtime import MODELS

type Source = str | Callable[..., System] | System
PROBLEMS: dict[tuple[Any, ...], BaseProblem] = {}


def _problem(
    source: Callable[..., System] | System,
    arguments: Mapping[str, Any],
    system: System | None,
    observables: Sequence[Observable] | str | None = None,
) -> BaseProblem:
    """Return the compiled problem of a source with factory arguments, compiled once per process and arguments."""
    if system is not None:
        key: tuple[Any, ...] = ('system', id(system))
        built = system
    elif isinstance(source, System):
        if arguments:
            error_msg = (
                f'A system source takes dotted paths of states and parameters, not the arguments {sorted(arguments)}.'
            )
            raise ValidationError(error_msg)
        key, built = ('system', id(source)), source
    else:
        key = (source, tuple(sorted((name, repr(value)) for name, value in arguments.items())))
        built = None
    key = (*key, observables if isinstance(observables, str | None) else tuple(map(id, observables)))
    if key not in PROBLEMS:
        built = built if built is not None else source(**arguments)
        PROBLEMS[key] = compile_system(built, observables=observables_for(observables, built))
    return PROBLEMS[key]


def resolve_source(source: Source) -> tuple[Callable[..., System] | System, dict[str, Any]]:
    """Return the factory or system of a model source and its default run settings.

    Args:
        source:
            A registered model name, a factory, or a system.

    Returns:
        source:
            The factory or the system, and the default run settings of a registered model.

    Raises:
        ValidationError:
            The source is none of the three.
    """
    if isinstance(source, str):
        entry = MODELS.resolve(source)
        return entry.factory, dict(entry.defaults)
    if isinstance(source, System) or callable(source):
        return source, {}
    error_msg = f'A model source is a registered model name, a factory, or a system, not {source!r}.'
    raise ValidationError(error_msg)


def split_overrides(values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return overrides split into factory arguments and dotted paths of states and parameters.

    Args:
        values:
            A mapping from factory argument or dotted path to value.

    Returns:
        overrides:
            The factory arguments and the dotted paths.
    """
    arguments = {name: value for name, value in values.items() if '.' not in name}
    paths = {name: value for name, value in values.items() if '.' in name}
    return arguments, paths


def observables_for(observables: Sequence[Observable] | str | None, system: System) -> list[Observable]:
    """Return the observables of a run: given ones, the standard ones of a registered model by name, or none.

    Args:
        observables:
            The observables, the name of a registered model whose standard observables to use, or `None`.

        system:
            The system of the run.

    Returns:
        observables:
            The observables.
    """
    if isinstance(observables, str):
        return MODELS.resolve(observables).observables_of(system)
    return list(observables or ())


def with_values[P: BaseProblem](problem: P, paths: Mapping[str, Any]) -> P:
    """Return a problem whose initial states and parameters take values by dotted path.

    Args:
        problem:
            The compiled problem.

        paths:
            A mapping from dotted state or parameter path to its value.

    Returns:
        problem:
            A copy of the problem with the new initial values.

    Raises:
        ValidationError:
            A path is neither a state nor a parameter, or a value is outside its domain.
    """
    states, parameters = problem.initial_state.copy(), problem.initial_parameters.copy()
    for name, value in paths.items():
        layout, vector = (problem.states, states) if problem.states.has(name) else (problem.parameters, parameters)
        if not layout.has(name):
            names = ', '.join([*problem.states.names, *problem.parameters.names])
            error_msg = f'{name} is neither a state nor a parameter of the model. Available paths are: {names}.'
            raise ValidationError(error_msg)
        entry = layout.entry(name)
        if not entry.domain.contains(value):
            error_msg = f'{name} = {value!r} is not in the {entry.domain.value} domain.'
            raise ValidationError(error_msg)
        vector[entry.indices] = entry.encode(value)
    return dataclasses.replace(problem, initial_state=states, initial_parameters=parameters)
