"""Define the orders spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _OrdersSpec(BaseSpec):
    """Orders and lags of the states of a system by entry name.

    It also holds the entry names of its algebraic states, and the state that is the rate of each state that declares
    one.
    """

    orders: dict[str, int]
    lags: dict[str, int]
    algebraic: frozenset[str] = frozenset()
    rates: dict[str, str] = field(default_factory=dict)
