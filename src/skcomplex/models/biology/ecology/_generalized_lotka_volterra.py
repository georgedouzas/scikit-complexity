"""Growth of the species of a group, `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`, with an interaction matrix `a`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.interactions import RateInteraction
from ....core.laws.utils import total
from ....specs import RoleSpec


class GeneralizedLotkaVolterraInteraction(RateInteraction):
    """Growth of the species of a group, `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`, with an interaction matrix `a`.

    The rate of a species is its intrinsic growth rate `b`, a parameter of the species, plus the total over the group
    weighted by row `i` of the matrix, times its abundance. The diagonal entries are the interactions of a species with
    itself, negative for self-limitation.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of species.

        weights:
            The interaction matrix, where `a_ij` is the effect of species `j` on the growth of species `i`.

        state:
            The name of the state that holds the abundance of a species.

    Examples:
        >>> from skcomplex.models.biology.ecology import GeneralizedLotkaVolterraInteraction
        >>> matrix = ((-1.0, 0.5), (0.5, -1.0))
        >>> GeneralizedLotkaVolterraInteraction('glv', target='species', weights=matrix).check().weights
        ((-1.0, 0.5), (0.5, -1.0))
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        weights: Sequence[Sequence[float]] = ((-1.0,),),
        state: str = 'x',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.weights = weights

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the rate `x_i (b_i + sum_j a_ij x_j)` of a species.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the group.

        Returns:
            rate:
                The rate of the abundance.
        """
        species = namespace.target
        x = getattr(species, self.state)
        rate: Expression = x * (species.b + total(x, weights=np.asarray(self.weights, dtype=float)))
        return rate
