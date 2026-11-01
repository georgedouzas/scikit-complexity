"""Oncology models: gene therapy of the immune response and oncolytic virotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.interactions import RateInteraction, ReactionInteraction
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ..._catalog import register_model
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction
from ._oncolytic_virus import OncolyticVirusElement
from ._oncolytic_virus_dendritic import OncolyticVirusDendriticElement
from ._sirna_tgf_beta import SirnaTgfBetaElement
from ._tcr_gene_therapy import TcrGeneTherapyElement

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


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
    return System(elements=[site], interactions=interactions)


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
    return System(elements=[site], interactions=interactions)


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
    tumour = OncolyticVirusElement('tumour', AU=AU, AV=AV, alpha=alpha, r=r, d=d)
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
    return System(elements=[tumour], interactions=interactions)


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
    site = OncolyticVirusDendriticElement('site', AU=AU, uv=uv, ud=ud, a0=a0, m=m, a5=a5)
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
    return System(elements=[site], interactions=interactions)
