"""Define observables, named values of a system that a run computes from its states at every frame."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import pickle  # nosec B403: pickles the model's own functions to check that they pickle, and never loads
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from ..base import ValidationError
from ..laws import Symbols

REDUCTIONS: dict[str, Callable[[Any], Any]] = {
    'count': lambda array: np.full(array.shape[1:], float(array.shape[0])),
    'sum': lambda array: array.sum(axis=0),
    'mean': lambda array: array.mean(axis=0),
    'min': lambda array: array.min(axis=0),
    'max': lambda array: array.max(axis=0),
    'var': lambda array: array.var(axis=0),
}


def _check_pickles(label: str, name: str, value: object) -> None:
    """Raise an error when a value of an observable cannot be pickled."""
    try:
        pickle.dumps(value)
    except (pickle.PicklingError, AttributeError, TypeError) as error:
        error_msg = (
            f'The {name} of the observable {label!r} cannot be pickled: {error}. Use a module-level function or a '
            'symbolic expression instead of a lambda or a local function.'
        )
        raise ValidationError(error_msg) from error


def _element_value(element: str, expression: Callable[[Any], object], s: Any) -> object:  # noqa: ANN401
    """Return the value of an expression of an element's namespace and the time, read from its system's namespace."""
    return expression(Symbols(**vars(getattr(s, element)), t=s.t))


def reduce_values(values: Any, reduce: str | Callable[[Any], Any]) -> Any:  # noqa: ANN401
    """Return the reduction of the values of a group's members.

    Args:
        values:
            One value per member, or one row per member for vector values.

        reduce:
            `'count'`, `'sum'`, `'mean'`, `'min'`, `'max'`, `'var'`, or a function of the array of values.

    Returns:
        value:
            The reduction, a number for scalar values and an array for vector values. The reductions of no members
            are 0, except the count.

    Raises:
        ValidationError:
            The reduction is unknown.

    Examples:
        >>> from skcomplex.core.system import reduce_values
        >>> reduce_values([1.0, 2.0, 6.0], 'mean')
        3.0
    """
    array = np.asarray(values) if reduce != 'count' else np.zeros(len(values))
    if callable(reduce):
        return reduce(array)
    if reduce not in REDUCTIONS:
        error_msg = f'Unknown reduction {reduce!r}. Use {", ".join(REDUCTIONS)}, or a function of the values.'
        raise ValidationError(error_msg)
    if not array.size and reduce != 'count':
        return 0.0
    result = np.real_if_close(REDUCTIONS[reduce](array))
    if np.ndim(result):
        return result
    return float(result) if np.isrealobj(result) else complex(result)


@dataclass(frozen=True)
class Observable:
    """Named value of a system that a run computes from its states and parameters at every frame.

    An observable is either an expression of the system's symbols, or a reduction over the members of a group of an
    expression or a variable of each member. A run takes its observables, records them in trajectories under
    `<system path>/<label>`, logs them to Rerun, checks those it names as invariants, and lets laws of other partitions
    read them through a role bound to their path, like an aggregate.

    Args:
        label:
            The label of the observable, unique among the observables of its system.

        expression:
            For a system observable, a function `expression(s)` of the system namespace, where `s.<child>.<variable>`
            is a symbol and `s.<observable>` is an observable declared earlier. For a reduction, a function of one
            member's namespace, or of one agent for agent groups. A symbolic expression is also accepted.

        group:
            The path of the group to reduce over, relative to the system, or `None` for a system observable.

        reduce:
            `'count'`, `'sum'`, `'mean'`, `'min'`, `'max'`, `'var'`, or a function of the array of member values.

        variable:
            The name of the member variable to reduce, instead of an expression.

    Raises:
        ValidationError:
            The expression or a function reduction cannot be pickled, such as a lambda.

    Examples:
        >>> from skcomplex.core.system import Observable
        >>> Observable('mean_wealth', group='traders', variable='wealth').check().reduce
        'mean'
    """

    label: str
    expression: object = None
    group: str | None = field(default=None, kw_only=True)
    reduce: str | Callable[[Any], Any] = field(default='mean', kw_only=True)
    variable: str | None = field(default=None, kw_only=True)

    def check(self) -> Self:
        """Return the observable after checking that its expression and its reduction pickle.

        Returns:
            observable:
                The observable itself.

        Raises:
            ValidationError:
                The expression or a function reduction cannot be pickled, such as a lambda.
        """
        _check_pickles(self.label, 'expression', self.expression)
        if callable(self.reduce):
            _check_pickles(self.label, 'reduction', self.reduce)
        return self


def observe_element(element: str, label: str, expression: Callable[[Any], object]) -> Observable:
    """Return an observable labelled `<element>.<label>` of an expression of the namespace of one element.

    The expression is the same function of an element's or an interaction's namespace `s` that its laws use, such as
    `s.debt / s.equity`, so a law and the observable share one definition. The namespace also holds the time `s.t`. The
    trajectory records the value under `<element>.<label>`, and Rerun logs it under the element.

    Args:
        element:
            The label of an element or an interaction of the root system.

        label:
            The name of the value.

        expression:
            A module-level function of the element's namespace.

    Returns:
        observable:
            The observable of the root system.

    Examples:
        >>> from operator import attrgetter
        >>> from skcomplex.core.system import observe_element
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models import create
        >>> stock = observe_element('hilsa', 'stock', attrgetter('spawners'))
        >>> trajectory = Simulation(create('hilsa_fishery'), dt=1, observables=[stock]).run(1).trajectory_
        >>> trajectory.values('hilsa.stock')[0]
        375.0
    """
    return Observable(f'{element}.{label}', functools.partial(_element_value, element, expression))
