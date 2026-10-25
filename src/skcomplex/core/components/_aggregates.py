"""Define aggregates, the quantities that summarise a group for partitions that cannot see its members."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

from ..base import BaseQuantity, ValidationError


@dataclass(frozen=True)
class AggregateQuantity(BaseQuantity):
    """Summary of a group, the count of its members or a reduction of one of their states.

    A system declares its aggregates among its components. In a co-simulation the scheduler computes the value at every
    synchronisation point, so partitions that cannot see a group's members, such as differential equations next to
    agents, read the group through the aggregate. A role binds the path of the aggregate, and laws read its value as
    `s.<role>.value`, which the trajectory records under `<aggregate path>.value`.

    Args:
        label:
            The label of the aggregate, unique among the children of its system.

        expression:
            Unused, since an aggregate counts the members or reduces one of their states, so `None`.

        group:
            The path of the agent group or element group, relative to the system that holds the aggregate.

        reduce:
            `'count'`, `'sum'`, `'mean'`, or a function of the array of the variable's values.

        variable:
            The state the reduction reads, unused for `'count'`.

    Raises:
        ValidationError:
            The aggregate has an expression, or names no group.

    Examples:
        >>> from skcomplex.core.components import AggregateQuantity
        >>> AggregateQuantity('size', group='farmers').check().reduce
        'count'
    """

    expression: object = None
    group: str = field(default='', kw_only=True)
    reduce: str | Callable[[Any], float] = field(default='count', kw_only=True)
    variable: str | None = field(default=None, kw_only=True)

    def _check(self) -> Self:
        """Return the aggregate after checking that it names a group and has no expression."""
        if not self.group:
            error_msg = f'The aggregate {self.label!r} names no group. Give the path of the group it summarises.'
            raise ValidationError(error_msg)
        if self.expression is not None:
            error_msg = (
                f'The aggregate {self.label!r} has an expression, but aggregates count the members of a group or '
                'reduce one of their states. Give `variable` instead.'
            )
            raise ValidationError(error_msg)
        return self
