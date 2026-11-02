"""Define linear relaxation, the decay of a state at a rate proportional to itself."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RelaxationElement(Element):
    """Linear relaxation `x' = -lam x`, whose fractional subclasses replace the derivative by a Caputo derivative.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        lam:
            The relaxation rate.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import RelaxationElement
        >>> RelaxationElement('relaxation', lam=2.0).lam
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        lam: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the decay of the value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, -s.lam * s.x)]
