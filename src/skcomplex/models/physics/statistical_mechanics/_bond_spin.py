"""Ising spin on a square lattice with quenched couplings to its four neighbours, of local energy `-s sum J_k s_k`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class BondSpinElement(Element):
    """Ising spin on a square lattice with quenched couplings to its four neighbours, of local energy `-s sum J_k s_k`.

    The couplings are constant states of every site, so a bond between two sites is stored at both of them with the same
    value. A `BondSamplingInteraction` flips the spins by Metropolis Monte Carlo at the temperature `T`, with the energy
    `-sum_<ij> J_ij s_i s_j - h sum_i s_i`.

    Args:
        label:
            The label of the spin or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        spin:
            The initial spin, 1 or -1.

        J_up:
            The coupling to the upper neighbour.

        J_left:
            The coupling to the left neighbour.

        J_right:
            The coupling to the right neighbour.

        J_down:
            The coupling to the lower neighbour.

        h:
            The external field.

        temperature:
            The temperature.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import BondSpinElement
        >>> BondSpinElement('spin', J_left=-1.0).J_left
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J_up: Annotated[float, StateSpec()] = 1.0,
        J_left: Annotated[float, StateSpec()] = 1.0,
        J_right: Annotated[float, StateSpec()] = 1.0,
        J_down: Annotated[float, StateSpec()] = 1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.spin = spin
        self.J_up = J_up
        self.J_left = J_left
        self.J_right = J_right
        self.J_down = J_down
        self.h = h
        self.temperature = temperature
