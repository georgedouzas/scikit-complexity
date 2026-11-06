"""Ricker growth of a population in discrete generations, `n' = r n exp(-n)`, overcompensating at high density."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.specs import ParameterSpec, RoleSpec


class RickerGrowthInteraction(Interaction):
    """Ricker growth of a population in discrete generations, `n' = r n exp(-n)`, overcompensating at high density.

    The size `n` is in units of the density at which the offspring of a generation peak, so a crowded generation
    leaves fewer offspring than a sparser one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The growth factor at low density.

    Examples:
        >>> from skcomplex.systems.biology.ecology import RickerGrowthInteraction
        >>> RickerGrowthInteraction('growth', target='fish', r=12.0).r
        12.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next generation `r n exp(-n)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n = s.target.n
        return [UpdateLaw(n, s.r * n * exp(-n))]
