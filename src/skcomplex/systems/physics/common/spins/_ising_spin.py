"""Ising spins, up at 1 or down at -1.

Every spin has its coupling `J`, its field `h`, and its temperature.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace


class IsingSpinElement(Element):
    """Ising spins, up at 1 or down at -1.

    Every spin has its coupling `J`, its field `h`, and its temperature.

    An `IsingSamplingInteraction` flips the spins by Monte Carlo sweeps at the temperature of the spins.

    Args:
        label:
            The label of the spin or the lattice of spins.

        spin:
            The initial spin, 1 or -1, or one spin per site.

        J:
            The coupling between neighbouring spins.

        h:
            The external field.

        temperature:
            The temperature in units of the coupling.

        space:
            The lattice or network of the spins, or `None` for one spin.

    Examples:
        >>> from skcomplex.systems.physics.common.spins import IsingSpinElement
        >>> IsingSpinElement('spin').temperature
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        spin: object = 1.0,
        J: float = 1.0,
        h: float = 0.0,
        temperature: Annotated[float, RealSet.open(0, oo)] = 2.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.spin = spin
        self.J = J
        self.h = h
        self.temperature = temperature
