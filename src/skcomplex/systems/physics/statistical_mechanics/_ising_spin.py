"""Ising spin, up at 1 or down at -1, with its coupling `J`, its field `h`, and its temperature."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class IsingSpinElement(Element):
    """Ising spin, up at 1 or down at -1, with its coupling `J`, its field `h`, and its temperature.

    An `IsingSamplingInteraction` flips the spins by Monte Carlo sweeps at the temperature of the spins.

    Args:
        label:
            The label of the spin or the lattice group.

        visual:
            How the spin is drawn, or `None` for `default_visual_`.

        spin:
            The initial spin, 1 or -1.

        J:
            The coupling between neighbouring spins.

        h:
            The external field.

        temperature:
            The temperature in units of the coupling.

    Attributes:
        default_visual_:
            The visual of the spin when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import IsingSpinElement
        >>> IsingSpinElement('spin').temperature
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.spin = spin
        self.J = J
        self.h = h
        self.temperature = temperature
