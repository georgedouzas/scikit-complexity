"""Two competing tumour clones and an immune response under chemotherapy and a dendritic-cell vaccine."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ClonalTumourElement(Element):
    """Two competing tumour clones and an immune response under chemotherapy and a dendritic-cell vaccine.

    Each clone grows logistically, competes with the other clone and with the immune cells `E`, and dies at the
    chemotherapy rate `u_chemo1` or `u_chemo2` and at the rate `b UIm E` of the vaccine-recruited immune cells. The
    immune cells grow logistically to `theta` and are recruited by both clones. Cells are counted, and time is in days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `clonal_chemoimmunotherapy` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A1:
            The initial number of cells of the first clone.

        A2:
            The initial number of cells of the second clone.

        E:
            The initial number of immune cells.

        r1:
            The growth rate of the first clone.

        r2:
            The growth rate of the second clone.

        b1:
            The reciprocal carrying capacity of the first clone.

        b2:
            The reciprocal carrying capacity of the second clone.

        c1:
            The competition of the second clone on the first.

        c2:
            The competition of the immune cells on the first clone.

        c3:
            The competition of the first clone on the second.

        c4:
            The competition of the immune cells on the second clone.

        r3:
            The growth rate of the immune cells.

        theta:
            The immune threshold.

        s1:
            The rate at which the first clone recruits immune cells.

        s2:
            The rate at which the second clone recruits immune cells.

        u_chemo1:
            The chemotherapy kill rate of the first clone.

        u_chemo2:
            The chemotherapy kill rate of the second clone.

        u_vaccine:
            The level of the immunotherapeutic agent.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ClonalTumourElement
        >>> ClonalTumourElement('tumour').theta
        300000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 90000.0,
        A2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 40000.0,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 500.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.11,
        b1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2e-08,
        b2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2e-08,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.5,
        c3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        c4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.5,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        theta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 300000.0,
        s1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0018,
        s2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0018,
        u_chemo1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        u_chemo2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        u_vaccine: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A1 = A1
        self.A2 = A2
        self.E = E
        self.r1 = r1
        self.r2 = r2
        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.r3 = r3
        self.theta = theta
        self.s1 = s1
        self.s2 = s2
        self.u_chemo1 = u_chemo1
        self.u_chemo2 = u_chemo2
        self.u_vaccine = u_vaccine
