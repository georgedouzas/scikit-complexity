"""Define the iterative matching spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class IterativeMatchingSpec(BaseSpec):
    """Iterative matching that adjusts a clearing value.

    The value moves until the requested and provided quantities balance. Starting from the last clearing value, or the
    median of the limits, the value moves by `p <- p (1 + adjustment (D - S) / (D + S))` until the relative excess of
    requests falls below `tolerance`, where `D` is the quantity requested by requesters whose limit is at least `p` and
    `S` the quantity of providers whose limit is at most `p`. The members then match at that value, and the long side is
    rationed in random order (Hamill and Gilbert 2016 ch. 5).

    Args:
        adjustment:
            The speed of the value adjustment.

        iterations:
            The largest number of adjustments per clearing.

        tolerance:
            The relative excess of requests at which the adjustment stops.

    Examples:
        >>> from skcomplex.core.specs import IterativeMatchingSpec
        >>> IterativeMatchingSpec(adjustment=0.05).iterations
        500
    """

    adjustment: float = 0.1
    iterations: int = 500
    tolerance: float = 1e-3
