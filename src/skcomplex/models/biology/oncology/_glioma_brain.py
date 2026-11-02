"""Glia, neurons, and a glioma under temozolomide and an oncolytic virus."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class GliomaBrainElement(Element):
    """Glia, neurons, and a glioma under temozolomide and an oncolytic virus.

    Normal glia `G` and the glioma compete. Neurons `N` die in proportion to the loss of glia,
    `psi dG/dt H(-dG/dt) N`. The drug `U` kills glia, neurons, and glioma with saturating kills. The virus `V` infects
    only glioma cells, and the infected cells `AV` burst. Cells and virions are counted, the drug is in mg per m², and
    time is in days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `glioma_chemovirotherapy` read its parameters.

    Args:
        label:
            The label of the brain.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        G:
            The initial number of glial cells.

        N:
            The initial number of neurons.

        AU:
            The initial number of uninfected glioma cells.

        AV:
            The initial number of infected glioma cells.

        V:
            The initial number of virions.

        U:
            The initial amount of drug.

        r1:
            The growth rate of glia.

        r2:
            The growth rate of the glioma.

        b:
            The reciprocal carrying capacity of glia and glioma.

        c1:
            The competition of the glioma on glia.

        c2:
            The competition of glia on the glioma.

        a1:
            The drug kill rate of glia.

        a2:
            The drug kill rate of neurons.

        a3:
            The drug kill rate of the glioma.

        g1:
            The half-saturation of the drug kill of glia.

        g2:
            The half-saturation of the drug kill of neurons.

        g3:
            The half-saturation of the drug kill of the glioma.

        psi:
            The loss of neurons per glial cell lost.

        alpha:
            The infection rate.

        a4:
            The rate at which the virus kills infected cells.

        beta_v:
            The burst size.

        d_u1:
            The decay rate of the virus.

        d_u2:
            The clearance rate of the drug.

        u1:
            The infusion rate of the drug, in mg per m² per day.

    Examples:
        >>> from skcomplex.models.biology.oncology import GliomaBrainElement
        >>> GliomaBrainElement('brain').u1
        400.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        G: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5940000000000.0,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5940000000000.0,
        AU: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 60000000000.0,
        AV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000000000.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0068,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.012,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.67e-13,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0769e-15,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0769e-16,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 280800.0,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 280800.0,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 280800000.0,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6000000000000.0,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6000000000000.0,
        g3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6000000000000.0,
        psi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.418e-13,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.2e-10,
        a4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta_v: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        d_u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 24.0,
        d_u2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 400.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.G = G
        self.N = N
        self.AU = AU
        self.AV = AV
        self.V = V
        self.U = U
        self.r1 = r1
        self.r2 = r2
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.psi = psi
        self.alpha = alpha
        self.a4 = a4
        self.beta_v = beta_v
        self.d_u1 = d_u1
        self.d_u2 = d_u2
        self.u1 = u1
