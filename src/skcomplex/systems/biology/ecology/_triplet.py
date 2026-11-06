"""Third-order interactions of the species of a group, `x_i sum_(j<k) a_ijk x_j x_k`, through a coupling tensor `a`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import contraction
from ....core.specs import ParameterSpec, RoleSpec


class TripletInteraction(RateInteraction):
    """Interactions of every species with the pairs of other species, `dx_i/dt += x_i sum_(j<k) a_ijk x_j x_k`.

    The coefficient `a_ijk` is the effect of the combination of the species `j` and `k` on the growth of the species
    `i` (Sidhom and Galla 2024, Eq. 2). The tensor holds it for both orders of the pair, `a_ijk = a_ikj`, so the sum
    over the sets `j < k` is half the contraction of the tensor with the abundances, and it holds 0 where indices
    coincide.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of species.

        a:
            The coupling tensor of order 3, with one index per species, or a number for every entry.

        state:
            The name of the state that holds the abundance of a species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import TripletInteraction
        >>> TripletInteraction('triplets', target='species', a=0.0, state='n').state
        'n'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[object, ParameterSpec(shape=('target', 'target', 'target'))] = 0.0,
        state: str = 'x',
    ) -> None:
        super().__init__(label, target=target, state=state, rate=self._triplets)
        self.a = a

    def _triplets(self, namespace: SimpleNamespace) -> object:
        """Return the growth of a species from the pairs of other species."""
        abundance = getattr(namespace.target, self.state)
        return abundance * contraction(namespace.a, abundance, abundance) / 2
