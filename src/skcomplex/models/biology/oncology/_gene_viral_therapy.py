"""Oncology models: gene therapy of the immune response and oncolytic virotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import ScipyODESolver
from ....specs import ParameterSpec, StateSpec, VisualSpec

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class TcrGeneTherapyElement(Element):
    """Effector cells and tumour cells under a gene therapy that makes the T cells express a tumour receptor.

    The effector cells follow `dE/dt = s + alpha A - d1 E + r1 E / (E + g1)` and the tumour cells
    `dA/dt = r2 A (1 - b A) - a E A / (A + g2)`, a predator-prey model of the immune attack.

    Args:
        label:
            The label of the tumour site.

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
        >>> from skcomplex.models.biology.oncology import TcrGeneTherapyElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the recruitment of effector cells and the logistic growth and immune kill of tumour cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        E, A = s.E, s.A
        return [
            RateLaw(E, s.s + s.alpha * A - s.d1 * E + s.r1 * E / (E + s.g1)),
            RateLaw(A, s.r2 * A * (1 - s.b * A) - s.a * E * A / (A + s.g2)),
        ]


class SirnaTgfBetaElement(Element):
    """Effector cells, tumour cells, IL-2, TGF-beta, and the siRNA that silences the production of TGF-beta.

    The model extends Kirschner and Panetta. TGF-beta lowers the recruitment and the IL-2 driven proliferation of
    effector cells and the production of IL-2, and raises the growth of the tumour. The siRNA, infused at the rate
    `u`, divides the production of TGF-beta by `1 + psi G^gamma3`.

    Args:
        label:
            The label of the tumour site.

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
        >>> from skcomplex.models.biology.oncology import SirnaTgfBetaElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the balances of the effector cells, the tumour, the two cytokines, and the siRNA.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        E, A, C1, C2, G = s.E, s.A, s.C1, s.C2, s.G
        proliferation = E * C1 / (s.g1 + C1) * (s.rho1 - s.rmax * C2 / (s.g2 + C2))
        silencing = 1 + s.psi * G**s.gamma3
        return [
            RateLaw(E, s.alpha * A / (1 + s.gamma1 * C2) - s.d1 * E + proliferation),
            RateLaw(A, s.r1 * A * (1 - s.b * A) - s.a1 * E * A / (s.g3 + A) + s.rho2 * C2 * A / (s.g4 + C2)),
            RateLaw(C1, s.rho3 * E * A / ((s.g5 + A) * (1 + s.gamma2 * C2)) - s.d2 * C1),
            RateLaw(C2, s.rho4 * A**2 / (s.mu**2 + silencing * A**2) - s.d3 * C2),
            RateLaw(G, s.u - s.du * G),
        ]


class OncolyticVirusElement(Element):
    """Uninfected and infected tumour cells that share a carrying capacity, with infection by mass action.

    In units of the carrying capacity and of the growth time of uninfected cells, the uninfected cells follow
    `dAU/dt = AU (1 - AU - AV) - alpha AU AV` and the infected cells `dAV/dt = r AV (1 - AU - AV) + alpha AU AV - d AV`.

    Args:
        label:
            The label of the tumour.

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
        >>> from skcomplex.models.biology.oncology import OncolyticVirusElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the shared logistic growth and the infection of the tumour cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        AU, AV = s.AU, s.AV
        room = 1 - AU - AV
        infection = s.alpha * AU * AV
        return [RateLaw(AU, AU * room - infection), RateLaw(AV, s.r * AV * room + infection - s.d * AV)]


class OncolyticVirusDendriticElement(Element):
    """Tumour cells, an oncolytic adenovirus, activated and naive T cells, and injected dendritic cells.

    The uninfected cells grow at the rate `r1` and the virus infects them at the rate `alpha AU V / Atot`, where
    `Atot = AU + AV + E + EN`. Activated T cells kill both kinds of tumour cells at the rate
    `(a0 + a1 AV) E / Atot`. Infected cells lyse at the rate `a2` and release `r2` virions each. The infected cells,
    the naive T cells, and the dendritic cells activate T cells.

    Args:
        label:
            The label of the tumour site.

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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infection, the immune kill, the virus, and the three immune populations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        AU, AV, V, E, EN, D = s.AU, s.AV, s.V, s.E, s.EN, s.D
        total = AU + AV + E + EN
        infection = s.alpha * AU * V / total
        kill = (s.a0 + s.a1 * AV) * E / total
        return [
            RateLaw(AU, s.r1 * AU - infection - kill * AU),
            RateLaw(AV, infection - s.a2 * AV - kill * AV),
            RateLaw(V, s.uv + s.r2 * s.a2 * AV - s.du1 * V),
            RateLaw(E, s.a3 * AV + s.m * EN + s.a4 * D - s.d1 * E),
            RateLaw(EN, s.a5 * AV - s.d2 * EN),
            RateLaw(D, s.ud - s.du2 * D),
        ]


@register_model(
    field='biology/oncology',
    tags=('gene therapy', 'immunotherapy', 'ode', 'predator-prey'),
    references=(BOOK, 'Tsygvintsev, Marino, and Kirschner 2013'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def tcr_gene_therapy(
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
) -> System:
    """Return the predator-prey model of a gene therapy that lets the T cells recognise the tumour.

    The effector cells follow `dE/dt = s + alpha A - d1 E + r1 E / (E + g1)` and the tumour cells
    `dA/dt = r2 A (1 - b A) - a E A / (A + g2)`. The book takes the time-varying influx, antigenicity, growth, and
    kill as constants. Cells are counts and time is in days. The tumour and the effector cells oscillate in damped
    cycles and settle on a dormant tumour far below its carrying capacity.

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

    Returns:
        system:
            The model, with the effector cells `site.E` and the tumour cells `site.A`.

    Examples:
        >>> from skcomplex.models.biology.oncology import tcr_gene_therapy
        >>> tcr_gene_therapy().elements[0].g2
        100000.0
    """
    site = TcrGeneTherapyElement('site', E=E, A=A, s=s, alpha=alpha, d1=d1, r1=r1, g1=g1, r2=r2, b=b, a=a, g2=g2)
    return System(elements=[site])


@register_model(
    field='biology/oncology',
    tags=('gene therapy', 'immunotherapy', 'cytokines', 'ode'),
    references=(BOOK, 'Arciero, Jackson, and Kirschner 2004'),
    defaults={'until': 200.0, 'dt': 0.5},
)
def sirna_tgf_beta(
    E: float = 1e3,
    A: float = 1e3,
    *,
    alpha: float = 0.035,
    rho4: float = 3e8,
    psi: float = 1.0,
    gamma3: float = 1.0,
    u: float = 0.0,
    du: float = 0.66,
) -> System:
    """Return the Kirschner-Panetta model with TGF-beta, which lets the tumour evade the immune system, and siRNA.

    TGF-beta lowers the recruitment of effector cells by `1 + gamma1 C2` and the production of IL-2, and raises the
    growth of the tumour. Without TGF-beta, `rho4 = 0`, the effector cells hold the tumour near `1e4` cells. With the
    book's production of TGF-beta the tumour escapes to its carrying capacity, and a large enough infusion of siRNA
    clears it. Cells are counts, the cytokines are in pg/ml, and time is in days. The other parameters are those of
    the book's table. Three decisions fill gaps in the book: the decay of IL-2 is `d2 C1`, where the book prints
    `d2 E`; the production of TGF-beta is `rho4 A^2 / (mu^2 + (1 + psi G^gamma3) A^2)`; and the book gives no values
    for `psi`, `gamma3`, or `u`, so `psi = gamma3 = 1` and `G` is in units where the infusion `u` is the dose.

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

    Returns:
        system:
            The model, with the effector cells `site.E`, the tumour cells `site.A`, IL-2 `site.C1`, TGF-beta
            `site.C2`, and the siRNA `site.G`.

    Examples:
        >>> from skcomplex.models.biology.oncology import sirna_tgf_beta
        >>> sirna_tgf_beta(u=1e6).elements[0].u
        1000000.0
    """
    site = SirnaTgfBetaElement('site', E=E, A=A, alpha=alpha, rho4=rho4, psi=psi, gamma3=gamma3, u=u, du=du)
    return System(elements=[site])


@register_model(
    field='biology/oncology',
    tags=('virotherapy', 'ode', 'infection', 'competition'),
    references=(BOOK, 'Novozhilov et al. 2006'),
    defaults={'until': 60.0, 'dt': 0.2},
)
def oncolytic_virus(AU: float = 0.1, AV: float = 1e-4, *, alpha: float = 1.0, r: float = 0.5, d: float = 0.5) -> System:
    """Return uninfected and infected tumour cells that share a carrying capacity, with infection by mass action.

    The uninfected cells follow `dAU/dt = AU (1 - AU - AV) - alpha AU AV` and the infected cells
    `dAV/dt = r AV (1 - AU - AV) + alpha AU AV - d AV`, in units of the carrying capacity and of the growth time of
    uninfected cells. The defaults lie in the ranges of the book's table. For `alpha > d` and `1 + alpha > r` the
    cells settle on the coexistence point `AV = (1 - d / alpha) / (1 + alpha - r)`, `AU = d / alpha - r AV`, so the
    virus reduces the tumour without eradicating it.

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

    Returns:
        system:
            The model, with the uninfected cells `tumour.AU` and the infected cells `tumour.AV`.

    Examples:
        >>> from skcomplex.models.biology.oncology import oncolytic_virus
        >>> oncolytic_virus(alpha=1.5).elements[0].alpha
        1.5
    """
    return System(elements=[OncolyticVirusElement('tumour', AU=AU, AV=AV, alpha=alpha, r=r, d=d)])


@register_model(
    field='biology/oncology',
    tags=('virotherapy', 'immunotherapy', 'dendritic cells', 'ode'),
    references=(BOOK, 'Wares et al. 2015'),
    defaults={'until': 30.0, 'dt': 0.1, 'solver': ScipyODESolver(method='LSODA')},
)
def oncolytic_virus_dendritic(
    AU: float = 60.0,
    *,
    uv: float = 2.5e9,
    ud: float = 1e6,
    a0: float = 0.0,
    m: float = 0.0,
    a5: float = 0.22,
) -> System:
    """Return a tumour treated with an oncolytic adenovirus that expresses immune signals and with dendritic cells.

    The virus infects the uninfected cells at the rate `alpha AU V / Atot`, where `Atot = AU + AV + E + EN`, the
    infected cells lyse and release virions, and the T cells that the infected and the dendritic cells activate kill
    both kinds of tumour cells. Volumes are in mm³, virions and cells are counts, and time is in days. The virus and
    the dendritic cells are given at the constant rates of the book. The book gives no values for the kill rate
    `a0` independent of the virus and the activation rate `m` of naive T cells, so both default to 0. The virus at
    the book's rate clears the tumour within days, while the dendritic cells, by raising the number of T cells in
    `Atot`, dilute the infection and let the tumour regrow.

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

    Returns:
        system:
            The model, with the tumour volumes `site.AU` and `site.AV`, the virions `site.V`, the activated and
            naive T cells `site.E` and `site.EN`, and the dendritic cells `site.D`.

    Examples:
        >>> from skcomplex.models.biology.oncology import oncolytic_virus_dendritic
        >>> oncolytic_virus_dendritic(ud=0.0).elements[0].ud
        0.0
    """
    return System(elements=[OncolyticVirusDendriticElement('site', AU=AU, uv=uv, ud=ud, a0=a0, m=m, a5=a5)])
