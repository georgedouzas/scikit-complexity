"""Contacts of the Maki-Thompson rumour system in a well-mixed population of size `N`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.specs import ParameterSpec, RoleSpec


class RumourContactsInteraction(Interaction):
    """Contacts of the Maki-Thompson rumour system in a well-mixed population of size `N`.

    A spreader who meets an ignorant tells the rumour at rate `spread * X * Y / N`, and a spreader who meets another
    spreader or a stifler stops spreading at rate `stifle * Y * (Y - 1 + Z) / N`.

    Args:
        label:
            The label of the interaction.

        ignorants:
            The path of the ignorants `X`.

        spreaders:
            The path of the spreaders `Y`.

        stiflers:
            The path of the stiflers `Z`.

        spread:
            The contact rate that spreads the rumour.

        stifle:
            The contact rate that stifles a spreader.

    Examples:
        >>> from skcomplex.systems.social.spreading import RumourContactsInteraction
        >>> RumourContactsInteraction('contacts', ignorants='X', spreaders='Y', stiflers='Z').spread
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        ignorants: Annotated[str, RoleSpec()],
        spreaders: Annotated[str, RoleSpec()],
        stiflers: Annotated[str, RoleSpec()],
        spread: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        stifle: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.ignorants = ignorants
        self.spreaders = spreaders
        self.stiflers = stiflers
        self.spread = spread
        self.stifle = stifle

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the telling and stifling jumps.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        ignorants, spreaders, stiflers = s.ignorants.n, s.spreaders.n, s.stiflers.n
        total = ignorants + spreaders + stiflers
        return [
            JumpLaw(s.spread * ignorants * spreaders / total, {ignorants: -1, spreaders: 1}),
            JumpLaw(s.stifle * spreaders * (spreaders - 1 + stiflers) / total, {spreaders: -1, stiflers: 1}),
        ]
