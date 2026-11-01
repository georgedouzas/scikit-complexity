"""Skew tent map `x' = x / a` below the peak `a` and `(1 - x) / (1 - a)` above it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import where
from ....specs import ParameterSpec, StateSpec, VisualSpec


class SkewTentMapElement(Element):
    """Skew tent map `x' = x / a` below the peak `a` and `(1 - x) / (1 - a)` above it.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        a:
            The position of the peak, between 0 and 1.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import SkewTentMapElement
        >>> SkewTentMapElement('point', a=0.5).a
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.4,
        a: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [UpdateLaw(s.x, where(s.x < s.a, s.x / s.a, (1 - s.x) / (1 - s.a)))]
