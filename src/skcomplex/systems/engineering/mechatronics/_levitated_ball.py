"""Conductive ball levitated in a vacuum, whose velocity relaxes between the impulses of the levitating field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LevitatedBallElement(Element):
    """Conductive ball levitated in a vacuum, whose velocity relaxes between the impulses of the levitating field.

    Between impulses the velocity follows `v' = -b v`, and every impulse of the field changes it by `m` or `-m`.

    Args:
        label:
            The label of the ball.

        visual:
            How the ball is drawn, or `None` for `default_visual_`.

        v:
            The initial velocity.

        b:
            The damping rate.

        m:
            The strength of every impulse.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import LevitatedBallElement
        >>> LevitatedBallElement('ball').m
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m: Annotated[float, ParameterSpec()] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.b = b
        self.m = m

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation of the velocity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.v, -s.b * s.v)]
