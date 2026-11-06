"""Eastern Pacific of the delayed oscillator of El Niño, its sea surface temperature anomaly."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import delayed
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class DelayedOscillatorElement(Element):
    """Eastern Pacific of the delayed oscillator of El Niño, its sea surface temperature anomaly.

    The anomaly follows `dT/dt = T - T^3 - alpha T(t - delta)`. It grows by local feedback and is damped by the
    delayed return of oceanic waves reflected at the western boundary.

    Args:
        label:
            The label of the ocean.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        T:
            The initial temperature anomaly, also its history.

        alpha:
            The strength of the delayed negative feedback.

        delta:
            The delay of the wave transit.

    Examples:
        >>> from skcomplex.systems.physics.climate import DelayedOscillatorElement
        >>> DelayedOscillatorElement('pacific').delta
        6.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec()] = 0.1,
        alpha: Annotated[float, ParameterSpec()] = 0.75,
        delta: Annotated[float, ParameterSpec()] = 6.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.alpha = alpha
        self.delta = delta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the local growth and saturation of the anomaly and its delayed negative feedback.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the anomaly.
        """
        return [RateLaw(s.T, s.T - s.T**3 - s.alpha * delayed(s.T, s.delta))]
