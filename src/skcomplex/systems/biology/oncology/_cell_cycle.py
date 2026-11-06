"""Tumour cells in three phases of the cell cycle under a killing and a blocking agent."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CellCycleElement(Element):
    """Tumour cells in three phases of the cell cycle under a killing and a blocking agent.

    The compartments hold the cells in G0 and G1 `A1`, in S `A2`, and in G2 and M `A3`, and follow
    `dA1/dt = -l1 A1 + 2 (1 - u1) l3 A3`, `dA2/dt = l1 A1 - (1 - u2) l2 A2`, and `dA3/dt = (1 - u2) l2 A2 - l3 A3`.
    The killing agent kills the fraction `u1` of the cells that divide, and the blocking agent slows the transit
    through S by the fraction `u2`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `cell_cycle_chemotherapy` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A1:
            The initial cells in G0 and G1.

        A2:
            The initial cells in S.

        A3:
            The initial cells in G2 and M.

        l1:
            The rate of exit from G0 and G1.

        l2:
            The rate of exit from S.

        l3:
            The rate of exit from G2 and M.

        u1:
            The fraction of dividing cells the killing agent kills.

        u2:
            The fraction by which the blocking agent slows the transit through S.

    Examples:
        >>> from skcomplex.systems.biology.oncology import CellCycleElement
        >>> CellCycleElement('tumour').l2
        0.395
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        A2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 3.8,
        A3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        l1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.197,
        l2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.395,
        l3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.107,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.u1 = u1
        self.u2 = u2
