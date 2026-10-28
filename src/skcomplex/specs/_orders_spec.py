"""Define the orders spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _OrdersSpec(BaseSpec):
    """Orders and lags of the states of a system by entry name."""

    orders: dict[str, int]
    lags: dict[str, int]
