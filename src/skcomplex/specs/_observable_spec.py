"""Define the observable spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler

from ..errors import ValidationError
from ..types import AnyArray
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ObservableSpec(BaseSpec):
    """Spec of a named value of a system that a run computes from its states and parameters at every frame.

    An observable is an expression of the namespace of the system, of one element when `element` is given, or a
    reduction over the members of a group of an expression or a variable of each member. A run records it in its
    trajectory under `<system path>/<label>`, or `<element>.<label>` for an element, and laws of other partitions read
    it through a role bound to its path.

    Args:
        label:
            The label of the observable, unique among the observables of its system.

        expression:
            A function of the namespace of the system, of the element, or of one member, or a symbolic expression.

        element:
            The label of an element or an interaction of the root system whose namespace the expression reads, or
            `None`.

        group:
            The path of the group to reduce over, relative to the system, or `None` for no reduction.

        reduce:
            `'count'`, `'sum'`, `'mean'`, `'min'`, `'max'`, `'var'`, `'logsumexp'`, or a function of the array of member
            values.

        variable:
            The name of the member variable to reduce, instead of an expression.

        function:
            A function `f(t, values)` of time and the mapping from the dotted name of every state and parameter to its
            value, instead of an expression, for a system without agents. A module-level function, since observables
            are pickled.

    Examples:
        >>> from skcomplex.specs import ObservableSpec
        >>> ObservableSpec('mean_wealth', group='traders', variable='wealth').check().reduce
        'mean'
    """

    label: str
    expression: object = None
    element: str | None = None
    group: str | None = None
    reduce: str | Callable[[AnyArray], object] = 'mean'
    variable: str | None = None
    function: Callable[[float, Mapping[str, object]], object] | None = None

    def _check(self) -> None:
        """Raise an error for an expression or a function reduction that cannot be pickled, such as a lambda.

        Raises:
            ValidationError:
                The expression or the reduction cannot be pickled.
        """
        for name, value in (('expression', self.expression), ('reduction', self.reduce), ('function', self.function)):
            try:
                ForkingPickler.dumps(value)
            except Exception as error:
                error_msg = (
                    f'The {name} of the observable {self.label!r} cannot be pickled: {error}. Use a module-level '
                    'function or a symbolic expression instead of a lambda or a local function.'
                )
                raise ValidationError(error_msg) from error
