"""Define aggregates, the elements that summarise a group for partitions that cannot see its members."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any

from ..base import state
from ._components import ElementComponent


class Aggregate(ElementComponent):
    """Summary of a group, the count of its members or a reduction of one of its states, held in the state `value`.

    In a co-simulation the scheduler computes the value at every synchronisation point, so partitions that cannot
    see a group's members, such as differential equations next to agents, read the group through the aggregate.

    Args:
        label:
            The label of the aggregate.

        group:
            The path of the agent group or element group, relative to the system that holds the aggregate.

        reduce:
            `'count'`, `'sum'`, `'mean'`, or a function of the array of the variable's values.

        variable:
            The state the reduction reads, unused for `'count'`.

        **values:
            The values of the declared variables by name:

            value:
                The initial value, before the first synchronisation point.

    Examples:
        >>> from skcomplex.core.components import Aggregate
        >>> Aggregate('size', group='farmers').reduce
        'count'
    """

    value: float = state(default=0.0)
    group: str  # type: ignore[assignment]  # the summarised group's label replaces the group constructor

    def __init__(
        self,
        label: str,
        *,
        group: str = '',
        reduce: str | Callable[[Any], float] = 'count',
        variable: str | None = None,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.group = group
        self.reduce = reduce
        self.variable = variable
