"""Define the settings of an evolution strategy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class EvolutionStrategySpec(BaseSpec):
    """Settings of an evolution strategy.

    Args:
        offspring:
            The number of offspring per generation, or `None` for pymoo's default.

    Examples:
        >>> from skcomplex.core.specs import EvolutionStrategySpec
        >>> EvolutionStrategySpec(offspring=200).offspring
        200
    """

    offspring: int | None = None
