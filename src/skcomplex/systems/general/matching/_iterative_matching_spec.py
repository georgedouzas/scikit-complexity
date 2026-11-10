"""Define the spec of an iterative matching that adjusts a clearing value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.specs import BaseSpec


@dataclass(frozen=True)
class IterativeMatchingSpec(BaseSpec):
    """A spec of an iterative matching that moves a value until supply meets demand.

    Starting from the last clearing value, or the median of the limits, the value moves by `p <- p (1 + adjustment (D -
    S) / (D + S))` until the relative excess of requests falls below `tolerance`. Here `D` is the quantity requested at
    a limit of at least `p` and `S` the quantity offered at a limit of at most `p`. The members then match at that
    value, and the long side is rationed in random order, as in Hamill and Gilbert (2016, chapter 5).

    Unlike `BatchMatchingSpec`, which reads the value off sorted offers, it searches for it by small steps. Choose it
    when a value adjusts to balance supply and demand, as in a tâtonnement of Walras.

    It appears in many fields. Prices adjust in models of general equilibrium, wages adjust in labour markets, and
    interest rates adjust in models of credit.

    Args:
        adjustment:
            The speed of the value adjustment.

        iterations:
            The largest number of adjustments per clearing.

        tolerance:
            The relative excess of requests at which the adjustment stops.

    Examples:
        >>> from skcomplex.systems.general.matching import IterativeMatchingSpec
        >>> IterativeMatchingSpec(adjustment=0.05).iterations
        500
    """

    adjustment: float = 0.1
    iterations: int = 500
    tolerance: float = 1e-3
