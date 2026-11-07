"""Receptor-effector negative feedback loop that holds a product at a set point an outside signal steers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import delayed, pulse
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FeedbackLoopElement(Element):
    """Receptor-effector negative feedback loop that holds a product at a set point an outside signal steers.

    The product `x` is made by the effector at the rate `v e` and consumed at the rate `d x`. The receptor reads the
    product after the transmission delay `tau` and inhibits the effector with the Hill function `1 / (1 + (x / K)^n)`,
    where the sensitivity `K (1 + steering)` rises while a steering signal acts, from `start` for `duration`. The
    effector follows the receptor with the response time `T`, `de/dt = (1 / (1 + (x(t - tau) / K)^n) - e) / T`.

    Args:
        label:
            The label of the loop.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial product.

        e:
            The initial effector activity.

        v:
            The largest rate of synthesis.

        d:
            The consumption rate of the product.

        K:
            The sensitivity of the receptor, the product at which it inhibits the effector by half.

        n:
            The steepness of the receptor.

        T:
            The response time of the effector.

        tau:
            The transmission delay between the receptor and the effector.

        steering:
            The relative shift of the sensitivity by the steering signal.

        start:
            The time the steering signal starts.

        duration:
            The duration of the steering signal.

    Examples:
        >>> from skcomplex.systems.biology.physiology import FeedbackLoopElement
        >>> FeedbackLoopElement('loop', tau=2.0).tau
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        e: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        v: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        steering: Annotated[float, ParameterSpec()] = 0.0,
        start: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.e = e
        self.v = v
        self.d = d
        self.K = K
        self.n = n
        self.T = T
        self.tau = tau
        self.steering = steering
        self.start = start
        self.duration = duration

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the synthesis and consumption of the product and the delayed response of the effector.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        sensitivity = s.K * (1 + s.steering * pulse(s.t, self.start, self.duration))
        receptor = 1 / (1 + (delayed(s.x, s.tau) / sensitivity) ** s.n)
        return [RateLaw(s.x, s.v * s.e - s.d * s.x), RateLaw(s.e, (receptor - s.e) / s.T)]
