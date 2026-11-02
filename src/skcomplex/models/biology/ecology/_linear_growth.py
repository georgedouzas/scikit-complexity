"""Linear growth of every species by the abundances of the species that influence it, normalised to one."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import neighbors, total, where
from ....specs import ParameterSpec, RoleSpec


class LinearGrowthInteraction(Interaction):
    """Linear growth of every species by the abundances of the species that influence it, normalised to one.

    Every species grows by `rate` times the sum of the abundances of the species it reads weighted by the strengths `w`
    of the influences, its abundance stays nonnegative, and the abundances are normalised to sum to one.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species on the nodes.

        influences:
            The path of the variable group on the edges.

        rate:
            The growth per step per unit of influence.

    Examples:
        >>> from skcomplex.models.biology.ecology import LinearGrowthInteraction
        >>> LinearGrowthInteraction('growth', species='species', influences='influences').rate
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        species: Annotated[str, RoleSpec()],
        influences: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label)
        self.species = species
        self.influences = influences
        self.rate = rate

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the linear growth and the normalisation of the abundances.

        Args:
            namespace:
                The namespace with `namespace.species` and `namespace.influences`.

        Returns:
            laws:
                The update of the abundance.
        """
        species = namespace.species
        grown = species.x + namespace.rate * neighbors(species.other.x, weight=namespace.influences.w)
        kept = where(grown > 0, grown, 0)
        return [UpdateLaw(species.x, kept / total(kept))]
