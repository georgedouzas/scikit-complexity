"""Define the settings of differential evolution."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class DifferentialEvolutionSpec(BaseSpec):
    """Settings of differential evolution.

    Args:
        variant:
            The scheme of pymoo's differential evolution, such as `'DE/best/1/bin'`, its default.

        weight:
            The differential weight of the mutation, by default that of pymoo.

        crossover:
            The crossover probability, by default that of pymoo.

    Examples:
        >>> from skcomplex.specs import DifferentialEvolutionSpec
        >>> DifferentialEvolutionSpec().check().variant
        'DE/best/1/bin'
    """

    variant: str = 'DE/best/1/bin'
    weight: float = 0.5
    crossover: float = 0.2
