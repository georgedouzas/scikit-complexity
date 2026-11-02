"""Tumour cells in a dormant phase and two cycling phases under a killing and a recruiting agent."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class DormantCellCycleElement(Element):
    """Tumour cells in a dormant phase and two cycling phases under a killing and a recruiting agent.

    The compartments hold the dormant cells in G0 `A0`, the cells in G1 and S `A1`, and the cells in G2 and M `A2`.
    A fraction `b0` of the daughters enters G0 and `b1` enters G1, so `dA0/dt = -l0 (1 + v) A0 + 2 b0 (1 - u) l2 A2`,
    `dA1/dt = l0 (1 + v) A0 - l1 A1 + 2 b1 (1 - u) l2 A2`, and `dA2/dt = l1 A1 - l2 A2`. The killing agent kills the
    fraction `u` of the cells that divide, and the recruiting agent speeds the exit from G0 by the factor `1 + v`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `cell_cycle_chemotherapy` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A0:
            The initial dormant cells.

        A1:
            The initial cells in G1 and S.

        A2:
            The initial cells in G2 and M.

        l0:
            The rate of exit from G0.

        l1:
            The rate of exit from G1 and S.

        l2:
            The rate of exit from G2 and M.

        b0:
            The fraction of daughters that enters G0.

        b1:
            The fraction of daughters that enters G1.

        u:
            The fraction of dividing cells the killing agent kills.

        v:
            The relative increase of the exit from G0 the recruiting agent causes.

    Examples:
        >>> from skcomplex.models.biology.oncology import DormantCellCycleElement
        >>> DormantCellCycleElement('tumour').b0
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A0: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 6.5,
        A1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        A2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        l0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        l1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        l2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        b0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        v: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.b0 = b0
        self.b1 = b1
        self.u = u
        self.v = v
