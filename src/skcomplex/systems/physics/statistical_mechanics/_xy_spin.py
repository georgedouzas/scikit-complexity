"""Planar spin of the XY system, an angle `theta`, with its coupling `J` and its temperature."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class XYSpinElement(Element):
    """Planar spin of the XY system, an angle `theta`, with its coupling `J` and its temperature.

    An `XYSamplingInteraction` rotates the angles by Monte Carlo sweeps at the temperature of the spins.

    Args:
        label:
            The label of the spin or the lattice group.

        visual:
            How the spin is drawn, or `None` for `default_visual_`.

        theta:
            The initial angle.

        J:
            The coupling between neighbouring spins.

        temperature:
            The temperature in units of the coupling.

    Attributes:
        default_visual_:
            The visual of the spin when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import XYSpinElement
        >>> XYSpinElement('spin').temperature
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.J = J
        self.temperature = temperature
