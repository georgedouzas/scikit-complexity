"""Spin of the Mattis system, a bond spin that also holds its quenched pattern `xi`, with couplings `J xi_i xi_j`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ

from ....core.specs import StateSpec, VisualSpec
from ._bond_spin import BondSpinElement


class MattisSpinElement(BondSpinElement):
    """Spin of the Mattis system, a bond spin that also holds its quenched pattern `xi`, with couplings `J xi_i xi_j`.

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

        pattern:
            The quenched pattern `xi`, 1 or -1.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import MattisSpinElement
        >>> MattisSpinElement('spin', pattern=-1.0).pattern
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: float = 1.0,
        J_up: float = 1.0,
        J_left: float = 1.0,
        J_right: float = 1.0,
        J_down: float = 1.0,
        h: float = 0.0,
        temperature: float = 1.0,
        pattern: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
    ) -> None:
        super().__init__(
            label,
            visual=visual,
            spin=spin,
            J_up=J_up,
            J_left=J_left,
            J_right=J_right,
            J_down=J_down,
            h=h,
            temperature=temperature,
        )
        self.pattern = pattern
