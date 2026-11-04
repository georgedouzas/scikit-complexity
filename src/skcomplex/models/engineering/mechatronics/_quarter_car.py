"""Quarter of a car: a body on a suspension above a wheel on a tire that follows a road step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import step
from ....specs import ParameterSpec, StateSpec, VisualSpec

POSITIVE = RealSet.open(0, oo)


class QuarterCarElement(Element):
    """Body of displacement `y1` and wheel of displacement `y2`, with a tire between the wheel and a road step.

    The element holds the tire, whose force on the wheel is `kt (road - y2)`, and a suspension interaction joins the
    body and the wheel.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        y1:
            The initial displacement of the body.

        y1_d1:
            The initial velocity of the body.

        y2:
            The initial displacement of the wheel.

        y2_d1:
            The initial velocity of the wheel.

        m1:
            The mass of the body.

        m2:
            The mass of the wheel.

        k:
            The stiffness of the suspension.

        b:
            The damping of the suspension.

        kt:
            The stiffness of the tire.

        height:
            The height of the road step.

        at:
            The time of the road step.

    Examples:
        >>> from skcomplex.models.engineering.mechatronics import QuarterCarElement
        >>> QuarterCarElement('car', height=0.05).height
        0.05
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y1: Annotated[object, StateSpec(order=2)] = 0.0,
        y1_d1: object = 0.0,
        y2: Annotated[object, StateSpec(order=2)] = 0.0,
        y2_d1: object = 0.0,
        m1: Annotated[object, ParameterSpec(domain=POSITIVE)] = 250.0,
        m2: Annotated[object, ParameterSpec(domain=POSITIVE)] = 30.0,
        k: Annotated[object, ParameterSpec(domain=POSITIVE)] = 16000.0,
        b: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1000.0,
        kt: Annotated[object, ParameterSpec(domain=POSITIVE)] = 160000.0,
        height: Annotated[object, ParameterSpec()] = 0.05,
        at: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y1 = y1
        self.y1_d1 = y1_d1
        self.y2 = y2
        self.y2_d1 = y2_d1
        self.m1 = m1
        self.m2 = m2
        self.k = k
        self.b = b
        self.kt = kt
        self.height = height
        self.at = at

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the acceleration of the wheel by the tire, the suspension apart."""
        return [RateLaw(s.y2, s.kt * (step(s.t, s.at, s.height) - s.y2) / s.m2)]
