"""Define the pool spec of the El Farol patrons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....specs import BaseSpec
from ....types import Array


@dataclass(frozen=True)
class _PoolSpec(BaseSpec):
    """Forecasts and errors of the pool of predictors in one week, shared by the patrons through the storage.

    Args:
        time:
            The week the pool was updated at.

        seeded:
            The attendances of the weeks before the run, which seed the history.

        forecasts:
            The forecast of every predictor.

        errors:
            The discounted error of every predictor.
    """

    time: float
    seeded: Array
    forecasts: Array
    errors: Array
