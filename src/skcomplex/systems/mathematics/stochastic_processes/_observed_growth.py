"""Quantity that grows exponentially and is observed and restarted at exponentially distributed times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, JumpLaw, RateLaw
from ....core.components.laws.utils import draw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ObservedGrowthElement(Element):
    """Quantity that grows exponentially and is observed and restarted at exponentially distributed times.

    The quantity follows `x' = mu x` from `x0`, and after a waiting time drawn from the exponential distribution of
    rate `lam` it is observed and starts again from `x0`.

    Args:
        label:
            The label of the element or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        x0:
            The value every growth starts from.

        mu:
            The growth rate.

        lam:
            The rate of the observation times.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ObservedGrowthElement
        >>> ObservedGrowthElement('growth', lam=2.0).lam
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        x0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x0 = x0
        self.mu = mu
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and the restart at the observation times.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            RateLaw(s.x, s.mu * s.x),
            JumpLaw(waiting=draw('wait', 'expon', scale=1 / s.lam), reset={s.x: s.x0}, label='observations'),
        ]
