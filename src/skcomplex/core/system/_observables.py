"""Define observables, named quantities of a system computed from its states at every frame."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import pickle  # nosec B403: pickles the model's own functions to check that they pickle, and never loads
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from ..base import ValidationError

REDUCTIONS: dict[str, Callable[[Any], Any]] = {
    'count': lambda array: np.full(array.shape[1:], float(array.shape[0])),
    'sum': lambda array: array.sum(axis=0),
    'mean': lambda array: array.mean(axis=0),
    'min': lambda array: array.min(axis=0),
    'max': lambda array: array.max(axis=0),
    'var': lambda array: array.var(axis=0),
}


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


class Observable(BaseEstimator):
    """Named quantity of a system, computed from its states and parameters at every frame.

    An observable is either an expression of the system's symbols, or a reduction over the members of a group of an
    expression or a variable of each member. It is recorded in trajectories under `<system path>/<label>`, logged to
    Rerun, and readable by laws of other partitions through a role bound to its path, like an aggregate.

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
        >>> Observable('mean_wealth', group='traders', variable='wealth').reduce
        'mean'
    """

    def __init__(
        self,
        label: str,
        expression: Callable[..., Any] | Any | None = None,  # noqa: ANN401
        *,
        group: str | None = None,
        reduce: str | Callable[[Any], Any] = 'mean',
        variable: str | None = None,
    ) -> None:
        self.label = label
        self.expression = expression
        self.group = group
        self.reduce = reduce
        self.variable = variable
        _check_pickles(label, 'expression', expression)
        if callable(reduce):
            _check_pickles(label, 'reduction', reduce)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label and whether the observable reduces a group."""
        target = f', group={self.group!r}' if self.group is not None else ''
        return f'Observable({self.label!r}{target})'
