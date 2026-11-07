"""Lateral inhibition of the neural fate of cells by their neighbours, through a kernel of the distance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import total
from ....core.specs import RoleSpec


class LateralInhibitionInteraction(Interaction):
    """Lateral inhibition of the neural fate of cells by their neighbours, through a kernel of the distance.

    Following Eq. (2) of Rand et al. (2021), `da_i/dt = sigma(a_i - h_i) - a_i` with the inhibition `h_i = sum_{j != i}
    K_ij d(a_j)`, where `sigma(u) = 1 / (1 + exp(-u / width))` and `d(a) = sigma(a - 1 / 2)`. A cell under an
    inhibition below 0.37 can only turn neural, above 0.63 only epidermal, and in between it is bistable.

    Args:
        label:
            The label of the interaction.

        cells:
            The path of the group of cells.

        kernel:
            The inhibition `K_ij` of every cell by every other, with a zero diagonal.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import LateralInhibitionInteraction
        >>> LateralInhibitionInteraction('inhibition', cells='cells', kernel=((0.0, 1.0), (1.0, 0.0))).kernel
        ((0.0, 1.0), (1.0, 0.0))
    """

    def __init__(
        self,
        label: str,
        *,
        cells: Annotated[str, RoleSpec()],
        kernel: Sequence[Sequence[float]] = ((0.0,),),
    ) -> None:
        super().__init__(label)
        self.cells = cells
        self.kernel = kernel

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation of every cell towards the fate its inhibition allows.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        cell = s.cells
        inhibition = total(self._sigmoid(cell.a - 1 / 2, cell.width), weights=np.asarray(self.kernel, dtype=float))
        return [RateLaw(cell.a, self._sigmoid(cell.a - inhibition, cell.width) - cell.a)]

    def _sigmoid(self, u: Expression, width: Expression) -> Expression:
        """Return the logistic sigmoid `1 / (1 + exp(-u / width))`."""
        return 1 / (1 + exp(-u / width))
