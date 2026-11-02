"""Tumour cells, an oncolytic adenovirus, activated and naive T cells, and injected dendritic cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class OncolyticVirusDendriticElement(Element):
    """Tumour cells, an oncolytic adenovirus, activated and naive T cells, and injected dendritic cells.

    The uninfected cells grow at the rate `r1` and the virus infects them at the rate `alpha AU V / Atot`, where
    `Atot = AU + AV + E + EN`. Activated T cells kill both kinds of tumour cells at the rate
    `(a0 + a1 AV) E / Atot`. Infected cells lyse at the rate `a2` and release `r2` virions each. The infected cells,
    the naive T cells, and the dendritic cells activate T cells.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `oncolytic_virus_dendritic` read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AU:
            The initial uninfected tumour volume.

        AV:
            The initial infected tumour volume.

        V:
            The initial number of virions.

        E:
            The initial number of activated T cells.

        EN:
            The initial number of naive T cells.

        D:
            The initial number of dendritic cells.

        r1:
            The growth rate of uninfected cells.

        alpha:
            The infection rate.

        a0:
            The kill rate of T cells that does not depend on the virus.

        a1:
            The kill rate of T cells per infected cell.

        a2:
            The lysis rate of infected cells.

        r2:
            The number of virions released per lysed cell.

        du1:
            The decay rate of virions.

        a3:
            The activation rate of T cells by infected cells.

        m:
            The activation rate of naive T cells.

        a4:
            The activation rate of T cells by dendritic cells.

        d1:
            The death rate of activated T cells.

        a5:
            The recruitment rate of naive T cells by infected cells.

        d2:
            The death rate of naive T cells.

        du2:
            The decay rate of dendritic cells.

        uv:
            The infusion rate of virions.

        ud:
            The injection rate of dendritic cells.

    Examples:
        >>> from skcomplex.models.biology.oncology import OncolyticVirusDendriticElement
        >>> OncolyticVirusDendriticElement('site').r2
        3500.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AU: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 60.0,
        AV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        EN: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3198,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00100854,
        a0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.954e-07,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3500.0,
        du1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.3,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.6984,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        a4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.6754,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.35,
        a5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.22,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.35,
        du2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.35,
        uv: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2500000000.0,
        ud: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AU = AU
        self.AV = AV
        self.V = V
        self.E = E
        self.EN = EN
        self.D = D
        self.r1 = r1
        self.alpha = alpha
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.r2 = r2
        self.du1 = du1
        self.a3 = a3
        self.m = m
        self.a4 = a4
        self.d1 = d1
        self.a5 = a5
        self.d2 = d2
        self.du2 = du2
        self.uv = uv
        self.ud = ud
