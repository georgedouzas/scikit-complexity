"""Define the orders spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ...specs._base_spec import BaseSpec


@dataclass(frozen=True)
class _OrdersSpec(BaseSpec):
    """Orders and lags of the states of a system by entry name.

    It also holds the entry names of its algebraic states and the orders of its states of a fractional order.
    """

    orders: dict[str, int]
    lags: dict[str, int]
    algebraic: frozenset[str] = frozenset()
    fractional: dict[str, float] = field(default_factory=dict)
