"""Gompertz growth under a carrying capacity that the vessels on the tumour surface raise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import log, oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, RoleSpec


class VascularSupportInteraction(Interaction):
    """Gompertz growth under a carrying capacity that the vessels on the tumour surface raise.

    The tumour follows `dn/dt = r n log(K / n)` and the capacity `dK/dt = d n^(2/3)`.

    Args:
        label:
            The label of the interaction.

        tumour:
            The path of the tumour.

        vasculature:
            The path of the vasculature.

        r:
            The rate at which the growth slows.

        d:
            The rate at which the surface of the tumour raises the capacity.

    Examples:
        >>> from skcomplex.models.biology.oncology import VascularSupportInteraction
        >>> VascularSupportInteraction('support', tumour='tumour', vasculature='vessels').d
        0.829
    """

    def __init__(
        self,
        label: str,
        *,
        tumour: Annotated[str, RoleSpec()],
        vasculature: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.63,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.829,
    ) -> None:
        super().__init__(label)
        self.tumour = tumour
        self.vasculature = vasculature
        self.r = r
        self.d = d

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Gompertz growth of the tumour and the growth of the capacity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n, K = s.tumour.n, s.vasculature.K
        return [RateLaw(n, s.r * n * log(K / n)), RateLaw(K, s.d * n ** (2 / 3))]
