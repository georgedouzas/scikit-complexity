"""Ikeda delay equation `x' = a sin x(t - tau)`, which oscillates about its stable rest point once `a tau > π / 2`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import delayed
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class IkedaElement(Element):
    """Ikeda delay equation `x' = a sin x(t - tau)`, which oscillates about its stable rest point once `a tau > π / 2`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, also its history.

        a:
            The gain of the delayed feedback.

        tau:
            The delay.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import IkedaElement
        >>> IkedaElement('ikeda', tau=2.0).tau
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 2.0,
        a: Annotated[float, ParameterSpec()] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the delayed feedback of the state.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.a * sin(delayed(s.x, s.tau)))]
