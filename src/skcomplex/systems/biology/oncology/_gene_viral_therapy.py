"""Oncology systems: gene therapy of the immune response and oncolytic virotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.reactions import ReactionInteraction
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class OncolyticVirusElement(Element):
    """Uninfected and infected tumour cells that share a carrying capacity.

    The infection is by mass action.

    In units of the carrying capacity and of the growth time of uninfected cells, the uninfected cells follow
    `dAU/dt = AU (1 - AU - AV) - alpha AU AV` and the infected cells `dAV/dt = r AV (1 - AU - AV) + alpha AU AV - d AV`.

    The element holds the states and parameters and has no laws of its own: the interactions of `OncolyticVirusSystem`
    read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AU:
            The initial uninfected tumour cells.

        AV:
            The initial infected tumour cells.

        alpha:
            The infection rate.

        r:
            The growth rate of infected cells.

        d:
            The death rate of infected cells.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OncolyticVirusElement
        >>> OncolyticVirusElement('tumour').alpha
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AU: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        AV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AU = AU
        self.AV = AV
        self.alpha = alpha
        self.r = r
        self.d = d


class OncolyticVirusDendriticElement(Element):
    """Tumour cells, an oncolytic adenovirus, T cells, and dendritic cells.

    The T cells are activated and naive, and the dendritic cells are injected.

    The uninfected cells grow at the rate `r1` and the virus infects them at the rate `alpha AU V / Atot`, where
    `Atot = AU + AV + E + EN`. Activated T cells kill both kinds of tumour cells at the rate
    `(a0 + a1 AV) E / Atot`. Infected cells lyse at the rate `a2` and release `r2` virions each. The infected cells,
    the naive T cells, and the dendritic cells activate T cells.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `OncolyticVirusDendriticSystem` read its parameters.

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
        >>> from skcomplex.systems.biology.oncology import OncolyticVirusDendriticElement
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


class SirnaTgfBetaElement(Element):
    """Effector cells, tumour cells, IL-2, TGF-beta, and siRNA.

    The siRNA silences the production of TGF-beta.

    The system extends Kirschner and Panetta. TGF-beta lowers the recruitment and the IL-2 driven proliferation of
    effector cells and the production of IL-2, and raises the growth of the tumour. The siRNA, infused at the rate
    `u`, divides the production of TGF-beta by `1 + psi G^gamma3`.

    The element holds the states and parameters and has no laws of its own: the interactions of `SirnaTGFBetaSystem`
    read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The initial number of effector cells.

        A:
            The initial number of tumour cells.

        C1:
            The initial concentration of IL-2.

        C2:
            The initial concentration of TGF-beta.

        G:
            The initial amount of siRNA.

        alpha:
            The antigenicity of the tumour.

        gamma1:
            The inhibition of the recruitment of effector cells by TGF-beta.

        d1:
            The death rate of effector cells.

        g1:
            The half-saturation constant of the proliferation of effector cells by IL-2.

        rho1:
            The proliferation rate of effector cells without TGF-beta.

        rmax:
            The largest inhibition of that proliferation by TGF-beta.

        g2:
            The half-saturation constant of that inhibition.

        r1:
            The growth rate of tumour cells.

        b:
            The reciprocal carrying capacity of tumour cells.

        a1:
            The kill rate of tumour cells by effector cells.

        g3:
            The half-saturation constant of the kill.

        rho2:
            The largest growth of tumour cells that TGF-beta adds.

        g4:
            The half-saturation constant of that growth.

        rho3:
            The production rate of IL-2.

        g5:
            The half-saturation constant of the production of IL-2.

        gamma2:
            The inhibition of the production of IL-2 by TGF-beta.

        d2:
            The decay rate of IL-2.

        rho4:
            The largest production rate of TGF-beta.

        mu:
            The number of tumour cells at which the production of TGF-beta rises.

        psi:
            The fraction of the siRNA bound to the mRNA.

        gamma3:
            The exponent of the suppression of the mRNA by the siRNA.

        d3:
            The decay rate of TGF-beta.

        u:
            The infusion rate of siRNA.

        du:
            The decay rate of siRNA.

    Examples:
        >>> from skcomplex.systems.biology.oncology import SirnaTgfBetaElement
        >>> SirnaTgfBetaElement('site').rho4
        300000000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        C1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        G: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.035,
        gamma1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20000000.0,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1245,
        rmax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1121,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2000000.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-09,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        g3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100000.0,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.27,
        g4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20000000.0,
        rho3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        g5: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        gamma2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        rho4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 300000000.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000000.0,
        psi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.66,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.A = A
        self.C1 = C1
        self.C2 = C2
        self.G = G
        self.alpha = alpha
        self.gamma1 = gamma1
        self.d1 = d1
        self.g1 = g1
        self.rho1 = rho1
        self.rmax = rmax
        self.g2 = g2
        self.r1 = r1
        self.b = b
        self.a1 = a1
        self.g3 = g3
        self.rho2 = rho2
        self.g4 = g4
        self.rho3 = rho3
        self.g5 = g5
        self.gamma2 = gamma2
        self.d2 = d2
        self.rho4 = rho4
        self.mu = mu
        self.psi = psi
        self.gamma3 = gamma3
        self.d3 = d3
        self.u = u
        self.du = du


class TcrGeneTherapyElement(Element):
    """Effector cells and tumour cells under a gene therapy.

    The therapy makes the T cells express a tumour receptor.

    The effector cells follow `dE/dt = s + alpha A - d1 E + r1 E / (E + g1)` and the tumour cells
    `dA/dt = r2 A (1 - b A) - a E A / (A + g2)`, a predator-prey system of the immune attack.

    The element holds the states and parameters and has no laws of its own: the interactions of `TCRGeneTherapySystem`
    read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The initial number of effector cells.

        A:
            The initial number of tumour cells.

        s:
            The influx rate of effector cells.

        alpha:
            The antigenicity of the tumour.

        d1:
            The death rate of effector cells.

        r1:
            The proliferation rate of effector cells.

        g1:
            The half-saturation constant of the proliferation of effector cells.

        r2:
            The growth rate of tumour cells.

        b:
            The reciprocal carrying capacity of tumour cells.

        a:
            The kill rate that the therapy gives the effector cells.

        g2:
            The half-saturation constant of the kill.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TcrGeneTherapyElement
        >>> TcrGeneTherapyElement('site').r2
        0.18
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1245,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.001,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-09,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.A = A
        self.s = s
        self.alpha = alpha
        self.d1 = d1
        self.r1 = r1
        self.g1 = g1
        self.r2 = r2
        self.b = b
        self.a = a
        self.g2 = g2


class TCRGeneTherapySystem(BaseSystem):
    """The predator-prey system of a gene therapy against a tumour.

    The therapy lets the T cells recognise the tumour.

    The effector cells follow `dE/dt = s + alpha A - d1 E + r1 E / (E + g1)` and the tumour cells `dA/dt = r2 A (1 - b
    A) - a E A / (A + g2)`. The book takes the time-varying influx, antigenicity, growth, and kill as constants. Cells
    are counts and time is in days. The tumour and the effector cells oscillate in damped cycles and settle on a dormant
    tumour far below its carrying capacity. The system has the effector cells `site.E` and the tumour cells `site.A`.

    Args:
        E:
            The initial number of effector cells.

        A:
            The initial number of tumour cells.

        s:
            The influx rate of effector cells, per day.

        alpha:
            The antigenicity of the tumour, per day.

        d1:
            The death rate of effector cells, per day.

        r1:
            The proliferation rate of effector cells, per day.

        g1:
            The half-saturation constant of the proliferation of effector cells.

        r2:
            The growth rate of tumour cells, per day.

        b:
            The reciprocal carrying capacity of tumour cells, per cell.

        a:
            The kill rate that the therapy gives the effector cells, per day.

        g2:
            The half-saturation constant of the kill, in cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TCRGeneTherapySystem
        >>> TCRGeneTherapySystem().prepare().system_.elements[0].g2
        100000.0
    """

    field = 'biology/oncology'
    topics = ('gene therapy', 'immunotherapy', 'ode', 'predator-prey')
    references = (BOOK, 'Tsygvintsev, Marino, and Kirschner 2013')
    until = 1500.0
    dt = 5.0

    def __init__(
        self,
        E: float = 1e3,
        A: float = 1e3,
        *,
        s: float = 1.0,
        alpha: float = 0.05,
        d1: float = 0.03,
        r1: float = 0.1245,
        g1: float = 1e-3,
        r2: float = 0.18,
        b: float = 1e-9,
        a: float = 1.0,
        g2: float = 1e5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.E = E
        self.A = A
        self.s = s
        self.alpha = alpha
        self.d1 = d1
        self.r1 = r1
        self.g1 = g1
        self.r2 = r2
        self.b = b
        self.a = a
        self.g2 = g2

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the predator-prey system of a gene therapy.

        The therapy lets the T cells recognise the tumour.
        """
        s = self.s
        site = TcrGeneTherapyElement(
            'site',
            E=self.E,
            A=self.A,
            s=s,
            alpha=self.alpha,
            d1=self.d1,
            r1=self.r1,
            g1=self.g1,
            r2=self.r2,
            b=self.b,
            a=self.a,
            g2=self.g2,
        )
        interactions = [
            DosingInteraction('influx', target='site', state='E', dose=lambda s: s.target.s),
            RateInteraction('stimulation', target='site', state='E', rate=lambda s: s.target.alpha * s.target.A),
            ReactionInteraction(
                'effector_death',
                target='site',
                reactants={'E': 1},
                propensity=lambda s: s.target.d1 * s.target.E,
            ),
            MichaelisMentenInteraction(
                'self_proliferation',
                target='site',
                products={'E': 1},
                substrate='E',
                vmax=lambda s: s.target.r1,
                km=lambda s: s.target.g1,
            ),
            RateInteraction(
                'tumour_growth',
                target='site',
                state='A',
                rate=lambda s: s.target.r2 * s.target.A * (1 - s.target.b * s.target.A),
            ),
            MichaelisMentenInteraction(
                'kill',
                target='site',
                reactants={'A': 1},
                substrate='A',
                vmax=lambda s: s.target.a,
                km=lambda s: s.target.g2,
                modifiers={'E': 1},
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)


class SirnaTGFBetaSystem(BaseSystem):
    """The Kirschner-Panetta system with TGF-beta and siRNA.

    TGF-beta lets the tumour evade the immune system.

    TGF-beta lowers the recruitment of effector cells by `1 + gamma1 C2` and the production of IL-2, and raises the
    growth of the tumour. Without TGF-beta, `rho4 = 0`, the effector cells hold the tumour near `1e4` cells. With the
    book's production of TGF-beta the tumour escapes to its carrying capacity, and a large enough infusion of siRNA
    clears it. Cells are counts, the cytokines are in pg/ml, and time is in days. The other parameters are those of the
    book's table. Three decisions fill gaps in the book: the decay of IL-2 is `d2 C1`, where the book prints `d2 E`. The
    production of TGF-beta is `rho4 A^2 / (mu^2 + (1 + psi G^gamma3) A^2)`. The book gives no values for `psi`,
    `gamma3`, or `u`, so `psi = gamma3 = 1` and `G` is in units where the infusion `u` is the dose. The system has the
    effector cells `site.E`, the tumour cells `site.A`, IL-2 `site.C1`, TGF-beta `site.C2`, and the siRNA `site.G`.

    Args:
        E:
            The initial number of effector cells.

        A:
            The initial number of tumour cells.

        alpha:
            The antigenicity of the tumour, per day.

        rho4:
            The largest production rate of TGF-beta, in pg/ml per day.

        psi:
            The fraction of the siRNA bound to the mRNA.

        gamma3:
            The exponent of the suppression of the mRNA by the siRNA.

        u:
            The infusion rate of siRNA, per day.

        du:
            The decay rate of siRNA, per day.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import SirnaTGFBetaSystem
        >>> SirnaTGFBetaSystem(u=1e6).prepare().system_.elements[0].u
        1000000.0
    """

    field = 'biology/oncology'
    topics = ('gene therapy', 'immunotherapy', 'cytokines', 'ode')
    references = (BOOK, 'Arciero, Jackson, and Kirschner 2004')
    until = 200.0
    dt = 0.5

    def __init__(
        self,
        E: float = 1e3,
        A: float = 1e3,
        *,
        alpha: float = 0.035,
        rho4: float = 3e8,
        psi: float = 1.0,
        gamma3: float = 1.0,
        u: float = 0.0,
        du: float = 0.66,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.E = E
        self.A = A
        self.alpha = alpha
        self.rho4 = rho4
        self.psi = psi
        self.gamma3 = gamma3
        self.u = u
        self.du = du

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the Kirschner-Panetta system with TGF-beta."""
        site = SirnaTgfBetaElement(
            'site',
            E=self.E,
            A=self.A,
            alpha=self.alpha,
            rho4=self.rho4,
            psi=self.psi,
            gamma3=self.gamma3,
            u=self.u,
            du=self.du,
        )
        interactions = [
            RateInteraction(
                'stimulation',
                target='site',
                state='E',
                rate=lambda s: s.target.alpha * s.target.A / (1 + s.target.gamma1 * s.target.C2),
            ),
            ReactionInteraction(
                'effector_death',
                target='site',
                reactants={'E': 1},
                propensity=lambda s: s.target.d1 * s.target.E,
            ),
            RateInteraction(
                'effector_proliferation',
                target='site',
                state='E',
                rate=lambda s: s.target.E
                * s.target.C1
                / (s.target.g1 + s.target.C1)
                * (s.target.rho1 - s.target.rmax * s.target.C2 / (s.target.g2 + s.target.C2)),
            ),
            RateInteraction(
                'tumour_growth',
                target='site',
                state='A',
                rate=lambda s: s.target.r1 * s.target.A * (1 - s.target.b * s.target.A),
            ),
            MichaelisMentenInteraction(
                'kill',
                target='site',
                reactants={'A': 1},
                substrate='A',
                vmax=lambda s: s.target.a1,
                km=lambda s: s.target.g3,
                modifiers={'E': 1},
            ),
            MichaelisMentenInteraction(
                'stimulated_growth',
                target='site',
                products={'A': 1},
                substrate='C2',
                vmax=lambda s: s.target.rho2,
                km=lambda s: s.target.g4,
                modifiers={'A': 1},
            ),
            RateInteraction(
                'il2_production',
                target='site',
                state='C1',
                rate=lambda s: s.target.rho3
                * s.target.E
                * s.target.A
                / ((s.target.g5 + s.target.A) * (1 + s.target.gamma2 * s.target.C2)),
            ),
            ReactionInteraction(
                'il2_loss',
                target='site',
                reactants={'C1': 1},
                propensity=lambda s: s.target.d2 * s.target.C1,
            ),
            RateInteraction(
                'tgf_beta_production',
                target='site',
                state='C2',
                rate=lambda s: s.target.rho4
                * s.target.A**2
                / (s.target.mu**2 + (1 + s.target.psi * s.target.G**s.target.gamma3) * s.target.A**2),
            ),
            ReactionInteraction(
                'tgf_beta_loss',
                target='site',
                reactants={'C2': 1},
                propensity=lambda s: s.target.d3 * s.target.C2,
            ),
            DosingInteraction('sirna_dosing', target='site', state='G', dose=lambda s: s.target.u),
            ReactionInteraction(
                'sirna_loss',
                target='site',
                reactants={'G': 1},
                propensity=lambda s: s.target.du * s.target.G,
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)


class OncolyticVirusSystem(BaseSystem):
    """Uninfected and infected tumour cells that share a carrying capacity.

    The infection is by mass action.

    The uninfected cells follow `dAU/dt = AU (1 - AU - AV) - alpha AU AV` and the infected cells `dAV/dt = r AV (1 - AU
    - AV) + alpha AU AV - d AV`, in units of the carrying capacity and of the growth time of uninfected cells. The
    defaults lie in the ranges of the book's table. For `alpha > d` and `1 + alpha > r` the cells settle on the
    coexistence point `AV = (1 - d / alpha) / (1 + alpha - r)`, `AU = d / alpha - r AV`, so the virus reduces the tumour
    without eradicating it. The system has the uninfected cells `tumour.AU` and the infected cells `tumour.AV`.

    Args:
        AU:
            The initial uninfected cells.

        AV:
            The initial infected cells.

        alpha:
            The infection rate.

        r:
            The growth rate of infected cells.

        d:
            The death rate of infected cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OncolyticVirusSystem
        >>> OncolyticVirusSystem(alpha=1.5).prepare().system_.elements[0].alpha
        1.5
    """

    field = 'biology/oncology'
    topics = ('virotherapy', 'ode', 'infection', 'competition')
    references = (BOOK, 'Novozhilov et al. 2006')
    until = 60.0
    dt = 0.2

    def __init__(
        self,
        AU: float = 0.1,
        AV: float = 1e-4,
        *,
        alpha: float = 1.0,
        r: float = 0.5,
        d: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.AU = AU
        self.AV = AV
        self.alpha = alpha
        self.r = r
        self.d = d

    def _build_system(self) -> CompositeSystem:
        """Return the parts of uninfected and infected tumour cells.

        The cells share a carrying capacity.
        """
        tumour = OncolyticVirusElement('tumour', AU=self.AU, AV=self.AV, alpha=self.alpha, r=self.r, d=self.d)
        interactions = [
            RateInteraction(
                'uninfected_growth',
                target='tumour',
                state='AU',
                rate=lambda s: s.target.AU * (1 - s.target.AU - s.target.AV),
            ),
            RateInteraction(
                'infected_growth',
                target='tumour',
                state='AV',
                rate=lambda s: s.target.r * s.target.AV * (1 - s.target.AU - s.target.AV),
            ),
            ReactionInteraction(
                'infection',
                target='tumour',
                reactants={'AU': 1},
                products={'AV': 1},
                modifiers={'AV': 1},
                propensity=lambda s: s.target.alpha * s.target.AU * s.target.AV,
            ),
            ReactionInteraction(
                'lysis',
                target='tumour',
                reactants={'AV': 1},
                propensity=lambda s: s.target.d * s.target.AV,
            ),
        ]
        return CompositeSystem(self.label, elements=[tumour], interactions=interactions)


class OncolyticVirusDendriticSystem(BaseSystem):
    """A tumour treated with an oncolytic adenovirus and with dendritic cells.

    The adenovirus expresses immune signals.

    The virus infects the uninfected cells at the rate `alpha AU V / Atot`, where `Atot = AU + AV + E + EN`, the
    infected cells lyse and release virions, and the T cells that the infected and the dendritic cells activate kill
    both kinds of tumour cells. Volumes are in mm³, virions and cells are counts, and time is in days. The virus and the
    dendritic cells are given at the constant rates of the book. The book gives no values for the kill rate `a0`
    independent of the virus and the activation rate `m` of naive T cells, so both default to 0. The virus at the book's
    rate clears the tumour within days, while the dendritic cells, by raising the number of T cells in `Atot`, dilute
    the infection and let the tumour regrow. The system has the tumour volumes `site.AU` and `site.AV`, the virions
    `site.V`, the activated and naive T cells `site.E` and `site.EN`, and the dendritic cells `site.D`.

    Args:
        AU:
            The initial uninfected tumour volume, in mm³.

        uv:
            The infusion rate of virions, per day.

        ud:
            The injection rate of dendritic cells, per day.

        a0:
            The kill rate of T cells that does not depend on the virus, per day.

        m:
            The activation rate of naive T cells, per day.

        a5:
            The recruitment rate of naive T cells by infected cells, per day.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OncolyticVirusDendriticSystem
        >>> OncolyticVirusDendriticSystem(ud=0.0).prepare().system_.elements[0].ud
        0.0
    """

    field = 'biology/oncology'
    topics = ('virotherapy', 'immunotherapy', 'dendritic cells', 'ode')
    references = (BOOK, 'Wares et al. 2015')
    until = 30.0
    dt = 0.1
    solver = ScipyODESolver(method='LSODA')

    def __init__(
        self,
        AU: float = 60.0,
        *,
        uv: float = 2.5e9,
        ud: float = 1e6,
        a0: float = 0.0,
        m: float = 0.0,
        a5: float = 0.22,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.AU = AU
        self.uv = uv
        self.ud = ud
        self.a0 = a0
        self.m = m
        self.a5 = a5

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a tumour treated with an oncolytic adenovirus."""
        site = OncolyticVirusDendriticElement(
            'site',
            AU=self.AU,
            uv=self.uv,
            ud=self.ud,
            a0=self.a0,
            m=self.m,
            a5=self.a5,
        )
        interactions = [
            RateInteraction('tumour_growth', target='site', state='AU', rate=lambda s: s.target.r1 * s.target.AU),
            ReactionInteraction(
                'infection',
                target='site',
                reactants={'AU': 1},
                products={'AV': 1},
                modifiers={'V': 1},
                propensity=lambda s: s.target.alpha
                * s.target.AU
                * s.target.V
                / (s.target.AU + s.target.AV + s.target.E + s.target.EN),
            ),
            ReactionInteraction(
                'uninfected_kill',
                target='site',
                reactants={'AU': 1},
                modifiers={'E': 1},
                propensity=lambda s: (s.target.a0 + s.target.a1 * s.target.AV)
                * s.target.E
                / (s.target.AU + s.target.AV + s.target.E + s.target.EN)
                * s.target.AU,
            ),
            ReactionInteraction(
                'infected_kill',
                target='site',
                reactants={'AV': 1},
                modifiers={'E': 1},
                propensity=lambda s: (s.target.a0 + s.target.a1 * s.target.AV)
                * s.target.E
                / (s.target.AU + s.target.AV + s.target.E + s.target.EN)
                * s.target.AV,
            ),
            ReactionInteraction(
                'lysis',
                target='site',
                reactants={'AV': 1},
                propensity=lambda s: s.target.a2 * s.target.AV,
            ),
            RateInteraction(
                'virus_release',
                target='site',
                state='V',
                rate=lambda s: s.target.r2 * s.target.a2 * s.target.AV,
            ),
            DosingInteraction('virus_injection', target='site', state='V', dose=lambda s: s.target.uv),
            ReactionInteraction(
                'virus_clearance',
                target='site',
                reactants={'V': 1},
                propensity=lambda s: s.target.du1 * s.target.V,
            ),
            RateInteraction('effector_recruitment', target='site', state='E', rate=lambda s: s.target.a3 * s.target.AV),
            RateInteraction('naive_activation', target='site', state='E', rate=lambda s: s.target.m * s.target.EN),
            RateInteraction('dendritic_activation', target='site', state='E', rate=lambda s: s.target.a4 * s.target.D),
            ReactionInteraction(
                'effector_death',
                target='site',
                reactants={'E': 1},
                propensity=lambda s: s.target.d1 * s.target.E,
            ),
            RateInteraction('naive_recruitment', target='site', state='EN', rate=lambda s: s.target.a5 * s.target.AV),
            ReactionInteraction(
                'naive_death',
                target='site',
                reactants={'EN': 1},
                propensity=lambda s: s.target.d2 * s.target.EN,
            ),
            DosingInteraction('dendritic_injection', target='site', state='D', dose=lambda s: s.target.ud),
            ReactionInteraction(
                'dendritic_clearance',
                target='site',
                reactants={'D': 1},
                propensity=lambda s: s.target.du2 * s.target.D,
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)
