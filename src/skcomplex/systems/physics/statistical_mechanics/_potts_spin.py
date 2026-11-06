"""Spin of the q-state Potts system, in a state from 0 to `q - 1`, with its coupling `J` and its temperature."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class PottsSpinElement(Element):
    """Spin of the q-state Potts system, in a state from 0 to `q - 1`, with its coupling `J` and its temperature.

    A `PottsSamplingInteraction` redraws the states by Monte Carlo sweeps at the temperature of the spins.

    Args:
        label:
            The label of the spin or the lattice group.

        visual:
            How the spin is drawn, or `None` for `default_visual_`.

        color:
            The initial state.

        J:
            The coupling between neighbouring spins.

        temperature:
            The temperature in units of the coupling.

    Attributes:
        default_visual_:
            The visual of the spin when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import PottsSpinElement
        >>> PottsSpinElement('spin').temperature
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        color: Annotated[float, StateSpec(domain=NN)] = 0.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.color = color
        self.J = J
        self.temperature = temperature
