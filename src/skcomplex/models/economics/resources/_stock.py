"""Renewable stock that grows logistically, `n' = r n (1 - n / K)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class StockElement(Element):
    """Renewable stock that grows logistically, `n' = r n (1 - n / K)`.

    Args:
        label:
            The label of the stock.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial stock.

        r:
            The intrinsic growth rate.

        K:
            The carrying capacity.

    Examples:
        >>> from skcomplex.models.economics.resources import StockElement
        >>> StockElement('stock', n=500.0).n
        500.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        r: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.r = r
        self.K = K

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the logistic growth of the stock.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.n, s.r * s.n * (1 - s.n / s.K))]
