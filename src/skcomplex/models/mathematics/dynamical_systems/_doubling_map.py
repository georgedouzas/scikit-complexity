"""Doubling map `x' = 2 x (mod 1)` on the grid of the fractions of denominator `q`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import floor, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import mod
from ....specs import ParameterSpec, StateSpec, VisualSpec

DOUBLING_DENOMINATOR = 1125899906840747.0


class DoublingMapElement(Element):
    """Doubling map `x' = 2 x (mod 1)` on the grid of the fractions of denominator `q`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        q:
            The denominator of the grid.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import DoublingMapElement
        >>> DoublingMapElement('point', q=1024.0).q
        1024.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.4,
        q: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = DOUBLING_DENOMINATOR,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.q = q

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next value, rounded to the grid.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [UpdateLaw(s.x, mod(2 * floor(s.q * s.x + 1 / 2), s.q) / s.q)]
