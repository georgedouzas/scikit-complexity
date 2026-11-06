"""Phase difference of a forced oscillator, `phi' = detuning + epsilon sin(phi)`, the Adler equation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import sin

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class AdlerPhaseElement(Element):
    """Phase difference of a forced oscillator, `phi' = detuning + epsilon sin(phi)`, the Adler equation.

    The phase locks at a fixed point when `|detuning| <= epsilon`, and slips with the period
    `2 pi / sqrt(detuning^2 - epsilon^2)` otherwise.

    Args:
        label:
            The label of the phase difference.

        visual:
            How the phase difference is drawn, or `None` for `default_visual_`.

        phi:
            The initial phase difference.

        detuning:
            The detuning of the natural frequency from the forcing.

        epsilon:
            The strength of the forcing.

    Attributes:
        default_visual_:
            The visual of the phase difference when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import AdlerPhaseElement
        >>> AdlerPhaseElement('difference').detuning
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec()] = 0.0,
        detuning: Annotated[float, ParameterSpec()] = 0.5,
        epsilon: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.detuning = detuning
        self.epsilon = epsilon

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the phase difference.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of `phi`.
        """
        return [RateLaw(namespace.phi, namespace.detuning + namespace.epsilon * sin(namespace.phi))]
