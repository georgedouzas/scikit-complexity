"""Climate of glacial cycles as a Van der Pol relaxation oscillator, a slow ice volume and a fast climate state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, pi

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class VanDerPolIceAgesElement(Element):
    """Climate of glacial cycles as a Van der Pol relaxation oscillator, a slow ice volume and a fast climate state.

    With time in thousands of years, `tau x' = -(y + beta + gamma F(t))` and `tau y' = -alpha (y^3/3 - y - x)`, where
    `F(t) = cos(2 pi t/23) + cos(2 pi t/41)` stands for precession and obliquity.

    Args:
        label:
            The label of the climate.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial ice volume.

        y:
            The initial climate state.

        alpha:
            The time-scale separation of the fast climate state.

        beta:
            The asymmetry of the cycle, between build-up and termination.

        gamma:
            The strength of the orbital forcing.

        tau:
            The time scale, in thousands of years.

    Examples:
        >>> from skcomplex.models.physics.climate import VanDerPolIceAgesElement
        >>> VanDerPolIceAgesElement('climate').alpha
        30.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 30.0,
        beta: Annotated[float, ParameterSpec()] = 0.75,
        gamma: Annotated[float, ParameterSpec()] = 0.0,
        tau: Annotated[float, ParameterSpec()] = 35.09,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def _get_orbital_forcing(self, s: SimpleNamespace) -> Expression:
        """Return the orbital forcing, the sum of the cycles of precession and obliquity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            forcing:
                The orbital forcing.
        """
        return cos(2 * pi * s.t / 23) + cos(2 * pi * s.t / 41)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the slow ice volume and the fast climate of the relaxation oscillator under orbital forcing.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the ice volume and the climate state.
        """
        forcing = self._get_orbital_forcing(s)
        return [
            RateLaw(s.x, -(s.y + s.beta + s.gamma * forcing) / s.tau),
            RateLaw(s.y, -s.alpha * (s.y**3 / 3 - s.y - s.x) / s.tau),
        ]
