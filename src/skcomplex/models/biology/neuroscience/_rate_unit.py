"""Neural unit of a competitive network, whose activity grows with its input and saturates, `tau x' = x (I - x)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RateUnitElement(Element):
    """Neural unit of a competitive network, whose activity grows with its input and saturates, `tau x' = x (I - x)`.

    Args:
        label:
            The label of the unit.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial activity.

        I:
            The input.

        tau:
            The time constant.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import RateUnitElement
        >>> RateUnitElement('unit', I=1.2).I
        1.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        I: Annotated[float, ParameterSpec()] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.I = I
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the self-limited growth of the activity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.x, s.x * (s.I - s.x) / s.tau)]
