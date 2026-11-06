"""Concentration `x` of circulating blood cells, made at a delayed Hill rate and lost at a constant rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import delayed, hill
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MackeyGlassElement(Element):
    """Concentration `x` of circulating blood cells, made at a delayed Hill rate and lost at a constant rate.

    The concentration follows `x' = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration, also its history.

        beta:
            The production rate.

        gamma:
            The loss rate.

        n:
            The Hill coefficient of the production.

        tau:
            The delay of the production.

    Examples:
        >>> from skcomplex.systems.biology.physiology import MackeyGlassElement
        >>> MackeyGlassElement('blood', tau=6.0).tau
        6.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta: Annotated[float, ParameterSpec()] = 0.2,
        gamma: Annotated[float, ParameterSpec()] = 0.1,
        n: Annotated[float, ParameterSpec()] = 10.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 17.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the delayed production and the loss of the cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        history = delayed(s.x, s.tau)
        return [RateLaw(s.x, s.beta * history * (1 - hill(history, 1, s.n)) - s.gamma * s.x)]
