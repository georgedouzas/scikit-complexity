"""Define the market state spec of the Brock-Hommes market."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.specs import BaseSpec
from ....core.types import Array


@dataclass(frozen=True)
class _MarketStateSpec(BaseSpec):
    """Last two prices and forecasts of the market, which its institution keeps in the storage of the system.

    Args:
        history:
            The last two deviations of the price, or the initial one.

        forecasts:
            The forecasts of the heuristics at the last step, or `None` before the first.
    """

    history: tuple[float, ...]
    forecasts: Array | None = None
