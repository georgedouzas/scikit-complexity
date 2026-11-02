"""Oncology models: tumour-immune interactions under adoptive therapy, vaccines, cytokines, and checkpoints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, log

from ....core.interactions import RateInteraction, ReactionInteraction
from ....core.laws.utils import delayed, hill, where
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ..._catalog import register_model
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction
from ._checkpoint_tumour import CheckpointTumourElement
from ._delayed_vaccinated_tumour import DelayedVaccinatedTumourElement
from ._kirschner_panetta_tumour import KirschnerPanettaTumourElement
from ._suppressed_tumour import SuppressedTumourElement
from ._vaccinated_tumour import VaccinatedTumourElement

BOOK_REFERENCE = 'Padmanabhan, Meskin, and Al Moustafa 2021'


STIFF_SOLVER = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'ode', 'adoptive therapy', 'IL-2', 'oscillations'),
    references=(BOOK_REFERENCE, 'Kirschner and Panetta 1998'),
    defaults={'until': 1000.0, 'dt': 1.0, 'solver': STIFF_SOLVER},
)
def kirschner_panetta(
    E: float = 1.0,
    A: float = 1.0,
    C: float = 1.0,
    *,
    alpha_a: float = 0.02,
    r1: float = 0.18,
    c1: float = 1.0,
    a1: float = 500.0,
    a2: float = 7e7,
    u1: float = 0.0,
    u2: float = 0.0,
) -> System:
    """Return the Kirschner-Panetta model of effector cells, tumour cells, and IL-2 under adoptive therapy and IL-2.

    Untreated, a tumour of antigenicity in the oscillatory range, such as the default 0.02, and the effector cells
    oscillate in long cycles in which the tumour nearly vanishes and regrows to about `4 10^5`. The adoptive therapy
    adds `a1 u1` effector cells per day and the IL-2 injection `a2 u2` IL-2 per day. With `u1` and `u2` above 1, the
    book's treatment factors `a1 = 500` and `a2 = 7 10^7` are exceeded and the tumour is cleared. The populations are
    in volume units, the inputs in volume per day, and time in days, and the initial state is the book's.

    Args:
        E:
            The initial effector cells.

        A:
            The initial tumour cells.

        C:
            The initial IL-2 concentration.

        alpha_a:
            The antigenicity of the tumour, in the book's range 0 to 0.05.

        r1:
            The growth rate of the tumour.

        c1:
            The kill rate of tumour cells by effector cells.

        a1:
            The effector cells per unit of adoptive therapy.

        a2:
            The IL-2 per unit of IL-2 injection.

        u1:
            The adoptive therapy input.

        u2:
            The IL-2 injection input.

    Returns:
        system:
            The model, with the effector cells `site.E`, the tumour cells `site.A`, and the IL-2 `site.C`.

    Examples:
        >>> from skcomplex.models.biology.oncology import kirschner_panetta
        >>> kirschner_panetta(u1=1.0).elements[0].u1
        1.0
    """
    site = KirschnerPanettaTumourElement(
        'site',
        E=E,
        A=A,
        C=C,
        alpha_a=alpha_a,
        r1=r1,
        c1=c1,
        a1=a1,
        a2=a2,
        u1=u1,
        u2=u2,
    )
    interactions = [
        RateInteraction('stimulation', target='site', state='E', rate=lambda s: s.target.alpha_a * s.target.A),
        ReactionInteraction(
            'effector_death',
            target='site',
            reactants={'E': 1},
            propensity=lambda s: s.target.d1 * s.target.E,
        ),
        DosingInteraction('adoptive_therapy', target='site', state='E', dose=lambda s: s.target.a1 * s.target.u1),
        MichaelisMentenInteraction(
            'effector_proliferation',
            target='site',
            products={'E': 1},
            substrate='C',
            vmax=lambda s: s.target.rho1,
            km=lambda s: s.target.alpha1,
            modifiers={'E': 1},
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
            vmax=lambda s: s.target.c1,
            km=lambda s: s.target.alpha2,
            modifiers={'E': 1},
        ),
        DosingInteraction('il2_injection', target='site', state='C', dose=lambda s: s.target.a2 * s.target.u2),
        MichaelisMentenInteraction(
            'il2_production',
            target='site',
            products={'C': 1},
            substrate='A',
            vmax=lambda s: s.target.rho2,
            km=lambda s: s.target.alpha3,
            modifiers={'E': 1},
        ),
        ReactionInteraction(
            'il2_loss',
            target='site',
            reactants={'C': 1},
            propensity=lambda s: s.target.du * s.target.C,
        ),
    ]
    return System(elements=[site], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'ode', 'vaccine', 'dendritic cells', 'dosing'),
    references=(BOOK_REFERENCE, 'Castiglione and Piccoli 2006'),
    defaults={'until': 2880.0, 'dt': 2.0, 'solver': STIFF_SOLVER},
)
def dendritic_cell_vaccine(
    A: float = 0.1,
    *,
    rho2: float = 10.0,
    r4: float = 1e-2,
    dose: float = 0.5,
    period: float = 168.0,
    width: float = 1.0,
    start: float = 0.0,
    doses: int = 8,
) -> System:
    """Return helper and cytotoxic T cells, a tumour, dendritic cells, and IL-2 under a dendritic-cell vaccine.

    Vaccinated dendritic cells drive the proliferation of helper cells, which secrete IL-2, and IL-2 with the antigen
    drives the proliferation of cytotoxic cells, which kill the tumour. The book gives no initial state or schedule.
    The T cells start at their resting level `s / d = 0.02`, the dendritic cells and IL-2 at 0, and the vaccine is a
    one-hour injection of `dose` per hour every week, 8 times. Without vaccine the tumour settles at
    `1 - c1 s2 / (d2 r3) = 0.9`. The populations are in cells per mm³ scaled by the carrying capacity, IL-2 in
    pg/mm³, and time in hours.

    Args:
        A:
            The initial tumour cells.

        rho2:
            The proliferation constant of cytotoxic T cells.

        r4:
            The IL-2 secretion rate.

        dose:
            The injection rate of dendritic cells during a dose, 0 for no vaccine.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

    Returns:
        system:
            The model, with the helper cells `site.EH`, the cytotoxic cells `site.E`, the tumour `site.A`, the
            dendritic cells `site.ED`, and the IL-2 `site.C`.

    Examples:
        >>> from skcomplex.models.biology.oncology import dendritic_cell_vaccine
        >>> dendritic_cell_vaccine(dose=0.0).elements[0].dose
        0.0
    """
    site = VaccinatedTumourElement(
        'site',
        A=A,
        rho2=rho2,
        r4=r4,
        dose=dose,
        period=period,
        width=width,
        start=start,
        doses=doses,
    )
    interactions = [
        DosingInteraction('helper_influx', target='site', state='EH', dose=lambda s: s.target.s1),
        RateInteraction(
            'helper_proliferation',
            target='site',
            state='EH',
            rate=lambda s: s.target.rho1 * s.target.ED * s.target.r1 * s.target.EH * (1 - s.target.b1 * s.target.EH),
        ),
        ReactionInteraction(
            'helper_death',
            target='site',
            reactants={'EH': 1},
            propensity=lambda s: s.target.d1 * s.target.EH,
        ),
        DosingInteraction('cytotoxic_influx', target='site', state='E', dose=lambda s: s.target.s2),
        RateInteraction(
            'cytotoxic_proliferation',
            target='site',
            state='E',
            rate=lambda s: s.target.rho2
            * s.target.C
            * (s.target.A + s.target.ED)
            * s.target.r2
            * s.target.E
            * (1 - s.target.b2 * s.target.E),
        ),
        ReactionInteraction(
            'cytotoxic_death',
            target='site',
            reactants={'E': 1},
            propensity=lambda s: s.target.d2 * s.target.E,
        ),
        RateInteraction(
            'tumour_growth',
            target='site',
            state='A',
            rate=lambda s: s.target.r3 * s.target.A * (1 - s.target.b3 * s.target.A),
        ),
        ReactionInteraction(
            'kill',
            target='site',
            reactants={'A': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.c1 * s.target.A * s.target.E,
        ),
        DosingInteraction(
            'vaccine',
            target='site',
            state='ED',
            dose=lambda s: s.target.dose,
            period=period,
            width=width,
            start=start,
            count=doses,
        ),
        ReactionInteraction(
            'dendritic_kill',
            target='site',
            reactants={'ED': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.c2 * s.target.E * s.target.ED,
        ),
        RateInteraction(
            'il2_production',
            target='site',
            state='C',
            rate=lambda s: s.target.r4 * s.target.EH * s.target.ED,
        ),
        ReactionInteraction(
            'il2_consumption',
            target='site',
            reactants={'C': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.c3 * s.target.E * s.target.C,
        ),
        ReactionInteraction(
            'il2_loss',
            target='site',
            reactants={'C': 1},
            propensity=lambda s: s.target.d3 * s.target.C,
        ),
    ]
    return System(elements=[site], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'ode', 'checkpoint inhibitor', 'PD-1', 'dosing'),
    references=(BOOK_REFERENCE, 'Nikolopoulou et al. 2018'),
    defaults={'until': 150.0, 'dt': 0.5, 'solver': STIFF_SOLVER},
)
def anti_pd1_checkpoint(
    E: float = 6e-3,
    A: float = 0.3968,
    *,
    u_ap: float = 1e-10,
    sigma: float = 5.0,
    ENC: float = 1e-4,
    r3: float = 0.18,
    b: float = 1 / 0.8,
    c_l: float = 1.0,
) -> System:
    """Return activated T cells, a tumour, and an anti-PD-1 drug that frees the T cells from checkpoint inhibition.

    The PD-1-PD-L1 complex inhibits the activation and proliferation of T cells, and the drug lowers the free PD-1 on
    T cells. The cytokines are at their steady states, as in the book. The book gives neither the naive T cells `ENC`
    nor the association ratio `sigma`, which are set so that the T cells settle below the level that controls the
    tumour. The ranged parameters take the low ends of the book's ranges, and the drug is the book's continuous
    infusion of `10^-10` per day. A larger infusion frees more T cells and lowers the tumour. The book's free PD-1
    `(rho_r - du1 UAP) E` turns negative once the drug exceeds `rho_r / du1`, and it is held at 0 there. With all PD-1
    blocked nothing limits the T cells, which grow without bound and clear the tumour. Cells and the drug are in g/cm³
    and time in days.

    Args:
        E:
            The initial activated T cells.

        A:
            The initial tumour cells.

        u_ap:
            The infusion rate of the drug, 0 for no treatment.

        sigma:
            The ratio of association to dissociation of PD-1 and PD-L1.

        ENC:
            The naive T cells.

        r3:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        c_l:
            The expression of PD-L1 on tumour cells relative to T cells.

    Returns:
        system:
            The model, with the T cells `site.E`, the tumour `site.A`, and the drug `site.UAP`.

    Examples:
        >>> from skcomplex.models.biology.oncology import anti_pd1_checkpoint
        >>> anti_pd1_checkpoint(u_ap=0.0).elements[0].u_ap
        0.0
    """
    site = CheckpointTumourElement('site', E=E, A=A, u_ap=u_ap, sigma=sigma, ENC=ENC, r3=r3, b=b, c_l=c_l)

    def free_receptor(s: SimpleNamespace) -> Expression:
        """Return the free PD-1 on the T cells, which the drug lowers."""
        expression = s.target.rho_r - s.target.du1 * s.target.UAP
        receptor: Expression = where(expression > 0, expression, 0) * s.target.E
        return receptor

    interactions = [
        RateInteraction(
            'activation',
            target='site',
            state='E',
            rate=lambda s: (
                s.target.r1 * s.target.ENC * hill(s.target.C1_ss, s.target.C1_50)
                + s.target.r2 * s.target.E * hill(s.target.C2_ss, s.target.C2_50)
            )
            / (
                1
                + s.target.sigma
                * free_receptor(s)
                * s.target.sigma_l
                * (s.target.E + s.target.c_l * s.target.A)
                / s.target.psi_pp
            ),
        ),
        ReactionInteraction(
            'effector_death',
            target='site',
            reactants={'E': 1},
            propensity=lambda s: s.target.d1 * s.target.E,
        ),
        RateInteraction(
            'tumour_growth',
            target='site',
            state='A',
            rate=lambda s: s.target.r3 * s.target.A * (1 - s.target.b * s.target.A),
        ),
        ReactionInteraction(
            'kill',
            target='site',
            reactants={'A': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.a_rho * s.target.A * s.target.E,
        ),
        DosingInteraction('infusion', target='site', state='UAP', dose=lambda s: s.target.u_ap),
        ReactionInteraction(
            'binding',
            target='site',
            reactants={'UAP': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.a_rl * free_receptor(s) * s.target.UAP,
        ),
        ReactionInteraction(
            'clearance',
            target='site',
            reactants={'UAP': 1},
            propensity=lambda s: s.target.du2 * s.target.UAP,
        ),
    ]
    return System(elements=[site], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'delay', 'vaccine', 'dendritic cells', 'regulatory T cells'),
    references=(BOOK_REFERENCE, 'Qomlaqi et al. 2017'),
    defaults={'until': 30.0, 'dt': 0.1, 'solver': STIFF_SOLVER},
)
def delayed_dc_vaccine(
    *,
    dose: float = 1e6,
    period: float = 7.0,
    width: float = 1.0,
    start: float = 0.0,
    doses: int = 4,
    tau1: float = 1.26,
    tau2: float = 0.45,
    tau3: float = 0.47,
) -> System:
    """Return a tumour and eight innate and adaptive immune populations, with division delays, under a vaccine.

    Natural killer cells divide in `tau1`, CD8+ cells mature after seven divisions `7 tau2`, and naive helper cells
    become type 1 helper cells after two divisions `2 tau3`. Regulatory T cells, driven by the tumour and IL-2, inhibit
    natural killer, CD8+, and helper cells. The book's growth term `m1 ED E` of mature CD8+ cells has no saturation and
    makes them diverge within a day, so it saturates in the dendritic cells as `m1 ED E / (g3 + ED)`, like every other
    term the dendritic cells drive. The book gives no schedule, and the vaccine is a one-day injection of `dose`
    dendritic cells per day every week, 4 times. The parameters and initial state are the book's, and with them the
    regulatory T cells hold the adaptive response down, so the tumour grows toward its capacity `b`. Cells are counts,
    IL-2 is in IU/L, and time in days.

    Args:
        dose:
            The injection rate of dendritic cells during a dose, 0 for no vaccine.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

        tau1:
            The division time of natural killer cells.

        tau2:
            The division time of CD8+ cells.

        tau3:
            The division time of helper cells.

    Returns:
        system:
            The model, with the tumour `site.A`, the natural killer cells `site.ENK`, the naive and mature CD8+ cells
            `site.ENC` and `site.E`, the dendritic cells `site.ED`, the IL-2 `site.C`, the regulatory cells `site.ER`,
            and the naive and type 1 helper cells `site.ENH` and `site.E1H`.

    Examples:
        >>> from skcomplex.models.biology.oncology import delayed_dc_vaccine
        >>> delayed_dc_vaccine(tau2=0.5).elements[0].tau2
        0.5
    """
    site = DelayedVaccinatedTumourElement(
        'site',
        dose=dose,
        period=period,
        width=width,
        start=start,
        doses=doses,
        tau1=tau1,
        tau2=tau2,
        tau3=tau3,
    )
    interactions = [
        RateInteraction(
            'tumour_growth',
            target='site',
            state='A',
            rate=lambda s: s.target.r1 * s.target.A * log(s.target.b / s.target.A),
        ),
        ReactionInteraction(
            'cytotoxic_kill',
            target='site',
            reactants={'A': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.a1
            * (log(s.target.E) / log(s.target.A)) ** 5
            / ((log(s.target.E) / log(s.target.A)) ** 5 + s.target.g1)
            * s.target.A,
        ),
        ReactionInteraction(
            'nk_kill',
            target='site',
            reactants={'A': 1},
            modifiers={'ENK': 1},
            propensity=lambda s: s.target.a2 * s.target.ENK * s.target.A,
        ),
        DosingInteraction('nk_influx', target='site', state='ENK', dose=lambda s: s.target.s1),
        ReactionInteraction(
            'nk_death',
            target='site',
            reactants={'ENK': 1},
            propensity=lambda s: s.target.d1 * s.target.ENK,
        ),
        RateInteraction(
            'nk_proliferation',
            target='site',
            state='ENK',
            rate=lambda s: s.target.rho1
            * delayed(s.target.ENK, s.target.tau1)
            * delayed(s.target.C, s.target.tau1)
            / (s.target.g2 + delayed(s.target.C, s.target.tau1)),
        ),
        ReactionInteraction(
            'nk_suppression',
            target='site',
            reactants={'ENK': 1},
            modifiers={'ER': 1},
            propensity=lambda s: s.target.psi1 * s.target.ENK * s.target.ER,
        ),
        DosingInteraction('naive_cytotoxic_influx', target='site', state='ENC', dose=lambda s: s.target.s2),
        ReactionInteraction(
            'naive_cytotoxic_death',
            target='site',
            reactants={'ENC': 1},
            propensity=lambda s: s.target.d2 * s.target.ENC,
        ),
        MichaelisMentenInteraction(
            'cytotoxic_priming',
            target='site',
            reactants={'ENC': 1},
            modifiers={'ENC': 1},
            substrate='ED',
            vmax=lambda s: s.target.rho2,
            km=lambda s: s.target.g3,
        ),
        ReactionInteraction(
            'cytotoxic_death',
            target='site',
            reactants={'E': 1},
            propensity=lambda s: s.target.d3 * s.target.E,
        ),
        RateInteraction(
            'primed_expansion',
            target='site',
            state='E',
            rate=lambda s: s.target.rho3
            * delayed(s.target.ENC, 7 * s.target.tau2, 7 * tau2)
            * delayed(s.target.ED, 7 * s.target.tau2, 7 * tau2)
            / (s.target.g3 + delayed(s.target.ED, 7 * s.target.tau2, 7 * tau2)),
        ),
        RateInteraction(
            'dendritic_expansion',
            target='site',
            state='E',
            rate=lambda s: s.target.m1
            * delayed(s.target.ED, s.target.tau2)
            * delayed(s.target.E, s.target.tau2)
            / (s.target.g3 + delayed(s.target.ED, s.target.tau2)),
        ),
        RateInteraction(
            'il2_expansion',
            target='site',
            state='E',
            rate=lambda s: s.target.rho4
            * delayed(s.target.E, s.target.tau2)
            * delayed(s.target.C, s.target.tau2) ** 2
            / (s.target.g4 + delayed(s.target.C, s.target.tau2) ** 2),
        ),
        ReactionInteraction(
            'cytotoxic_suppression',
            target='site',
            reactants={'E': 1},
            modifiers={'ER': 1},
            propensity=lambda s: s.target.psi2 * s.target.E * s.target.ER,
        ),
        ReactionInteraction(
            'dendritic_death',
            target='site',
            reactants={'ED': 1},
            propensity=lambda s: s.target.d4 * s.target.ED,
        ),
        MichaelisMentenInteraction(
            'antigen_presentation',
            target='site',
            products={'ED': 1},
            substrate='A',
            vmax=lambda s: s.target.kappa1,
            km=lambda s: s.target.g5,
        ),
        DosingInteraction(
            'vaccine',
            target='site',
            state='ED',
            dose=lambda s: s.target.dose,
            period=period,
            width=width,
            start=start,
            count=doses,
        ),
        DosingInteraction('il2_influx', target='site', state='C', dose=lambda s: s.target.s3),
        ReactionInteraction(
            'il2_loss',
            target='site',
            reactants={'C': 1},
            propensity=lambda s: s.target.d5 * s.target.C,
        ),
        RateInteraction(
            'cytotoxic_secretion',
            target='site',
            state='C',
            rate=lambda s: s.target.rho5 * s.target.E**2 * s.target.C / (s.target.g6 + s.target.E**2),
        ),
        MichaelisMentenInteraction(
            'helper_secretion',
            target='site',
            products={'C': 1},
            modifiers={'C': 1},
            substrate='E1H',
            vmax=lambda s: s.target.rho6,
            km=lambda s: s.target.g7,
        ),
        DosingInteraction('regulatory_influx', target='site', state='ER', dose=lambda s: s.target.s4),
        ReactionInteraction(
            'regulatory_death',
            target='site',
            reactants={'ER': 1},
            propensity=lambda s: s.target.d6 * s.target.ER,
        ),
        RateInteraction(
            'antigen_expansion',
            target='site',
            state='ER',
            rate=lambda s: s.target.rho7
            * delayed(s.target.ER, s.target.tau3)
            * delayed(s.target.A, s.target.tau3) ** 2
            / (s.target.g8 + delayed(s.target.A, s.target.tau3) ** 2),
        ),
        RateInteraction(
            'il2_regulatory_expansion',
            target='site',
            state='ER',
            rate=lambda s: s.target.rho8
            * delayed(s.target.ER, s.target.tau3)
            * delayed(s.target.C, s.target.tau3)
            / (s.target.g9 + delayed(s.target.C, s.target.tau3)),
        ),
        DosingInteraction('naive_helper_influx', target='site', state='ENH', dose=lambda s: s.target.s5),
        ReactionInteraction(
            'naive_helper_death',
            target='site',
            reactants={'ENH': 1},
            propensity=lambda s: s.target.d7 * s.target.ENH,
        ),
        MichaelisMentenInteraction(
            'helper_priming',
            target='site',
            reactants={'ENH': 1},
            modifiers={'ENH': 1},
            substrate='ED',
            vmax=lambda s: s.target.rho9,
            km=lambda s: s.target.g3,
        ),
        ReactionInteraction(
            'helper_death',
            target='site',
            reactants={'E1H': 1},
            propensity=lambda s: s.target.d8 * s.target.E1H,
        ),
        RateInteraction(
            'primed_helper_expansion',
            target='site',
            state='E1H',
            rate=lambda s: s.target.m2
            * delayed(s.target.ED, 2 * s.target.tau3, 2 * tau3)
            * delayed(s.target.ENH, 2 * s.target.tau3, 2 * tau3)
            / (s.target.g3 + delayed(s.target.ED, 2 * s.target.tau3, 2 * tau3)),
        ),
        ReactionInteraction(
            'helper_suppression',
            target='site',
            reactants={'E1H': 1},
            modifiers={'ER': 1},
            propensity=lambda s: s.target.psi3 * s.target.ER * s.target.E1H,
        ),
        RateInteraction(
            'helper_expansion',
            target='site',
            state='E1H',
            rate=lambda s: s.target.rho10
            * delayed(s.target.ED, s.target.tau3)
            * delayed(s.target.E, s.target.tau3)
            / (s.target.g3 + delayed(s.target.ED, s.target.tau3)),
        ),
    ]
    return System(elements=[site], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'ode', 'immunosuppression', 'cytokines', 'antigenicity'),
    references=(BOOK_REFERENCE, 'Robertson-Tessi, El-Kareh, and Goriely 2012'),
    defaults={'until': 200.0, 'dt': 0.5, 'solver': STIFF_SOLVER},
)
def tumour_immune_suppression(
    *,
    alpha_a: float = 1e-3,
    r1: float = 0.333,
    r2: float = 333.0,
    EMC: float = 1e4,
    EMH: float = 1e4,
    EMR: float = 1e4,
) -> System:
    """Return a tumour against dendritic, cytotoxic, helper, and regulatory T cells with IL-2, IL-10, and TGF-beta.

    A tumour of one cell grows exponentially, then as a power of its size, and only its accessible fraction is killed.
    Its antigen matures dendritic cells, which recruit T cells from the memory pools, and the regulatory cells with
    IL-10 and TGF-beta suppress the response. A weakly antigenic tumour escapes, a moderately antigenic one oscillates,
    and a strongly antigenic one is held to a few cells. The book's growth rows are garbled, and the exponential rate
    is read as 0.333 per day and the power coefficient as 333 cells^(1/2) per day, within the book's range 100 to
    1000, so the growth turns from exponential to power at `10^6` cells. The book gives no memory pools, which are
    taken as constant at `10^4` cells each. The other parameters are the book's, the initial immune cells are
    `10^-5`, cells are counts, cytokines in ng/ml, and time in days.

    Args:
        alpha_a:
            The antigenicity of the tumour, in the book's range `10^-5` to `10^5`.

        r1:
            The exponential growth rate of the tumour.

        r2:
            The power growth coefficient of the tumour.

        EMC:
            The memory cytotoxic T cells.

        EMH:
            The memory helper T cells.

        EMR:
            The memory regulatory T cells.

    Returns:
        system:
            The model, with the tumour `site.A`, the dendritic cells `site.EUD` and `site.ED`, the T cells `site.EPC`,
            `site.E`, `site.EPH`, `site.EH`, `site.EPR`, and `site.ER`, and the cytokines `site.C1`, `site.C2`, and
            `site.C3`.

    Examples:
        >>> from skcomplex.models.biology.oncology import tumour_immune_suppression
        >>> tumour_immune_suppression(alpha_a=1.0).elements[0].alpha_a
        1.0
    """
    site = SuppressedTumourElement('site', alpha_a=alpha_a, r1=r1, r2=r2, EMC=EMC, EMH=EMH, EMR=EMR)

    def accessible(s: SimpleNamespace) -> Expression:
        """Return the fraction of the tumour accessible to the cytotoxic T cells."""
        fraction: Expression = s.target.A / (1 + (s.target.A ** (1 - s.target.q) / s.target.g3) ** s.target.p) ** (
            1 / s.target.p
        )
        return fraction

    def maturation(s: SimpleNamespace) -> Expression:
        """Return the maturation by IL-2 of proliferating T cells, which TGF-beta suppresses."""
        factor: Expression = hill(s.target.C1, s.target.EC1_50) / (1 + s.target.C3 / s.target.EC3_50)
        return factor

    interactions = [
        RateInteraction(
            'tumour_growth',
            target='site',
            state='A',
            rate=lambda s: s.target.A
            / ((1 / s.target.r1) ** s.target.p + (s.target.A ** (1 - s.target.n) / s.target.r2) ** s.target.p)
            ** (1 / s.target.p),
        ),
        ReactionInteraction(
            'kill',
            target='site',
            reactants={'A': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.a1
            * accessible(s)
            * s.target.E
            / (s.target.E + accessible(s) / s.target.g1 + s.target.ER / s.target.g2)
            / (1 + s.target.C3 / s.target.C3_50),
        ),
        RateInteraction(
            'antigen_uptake',
            target='site',
            state='EUD',
            rate=lambda s: s.target.alpha_a
            * accessible(s)
            / ((1 + s.target.C2 / s.target.C2_50) * (1 + s.target.ER / s.target.ER_50)),
        ),
        ReactionInteraction(
            'licensing',
            target='site',
            reactants={'EUD': 1},
            products={'ED': 1},
            propensity=lambda s: s.target.lambda_a * s.target.EUD / (1 + s.target.EUD / s.target.EMH),
        ),
        ReactionInteraction(
            'immature_death',
            target='site',
            reactants={'EUD': 1},
            propensity=lambda s: s.target.d1 * s.target.EUD,
        ),
        ReactionInteraction(
            'licensed_death',
            target='site',
            reactants={'ED': 1},
            propensity=lambda s: s.target.d2 * s.target.ED,
        ),
        RateInteraction(
            'cytotoxic_priming',
            target='site',
            state='EPC',
            rate=lambda s: s.target.a2
            * s.target.EMC
            * s.target.ED
            / (s.target.ED + s.target.g4 * (s.target.EMC + s.target.EMH + s.target.EMR)),
        ),
        ReactionInteraction(
            'cytotoxic_precursor_death',
            target='site',
            reactants={'EPC': 1},
            propensity=lambda s: s.target.d3 * s.target.EPC,
        ),
        RateInteraction(
            'cytotoxic_maturation',
            target='site',
            state='E',
            rate=lambda s: s.target.a3 * s.target.EPC * maturation(s),
        ),
        ReactionInteraction(
            'cytotoxic_death',
            target='site',
            reactants={'E': 1},
            propensity=lambda s: s.target.d4 * s.target.E,
        ),
        RateInteraction(
            'helper_priming',
            target='site',
            state='EPH',
            rate=lambda s: s.target.a4
            * s.target.EMH
            * (s.target.ED + s.target.EUD)
            / (s.target.ED + s.target.EUD + s.target.g4 * (s.target.EMC + s.target.EMH + s.target.EMR)),
        ),
        ReactionInteraction(
            'helper_precursor_death',
            target='site',
            reactants={'EPH': 1},
            propensity=lambda s: s.target.d5 * s.target.EPH,
        ),
        RateInteraction(
            'helper_maturation',
            target='site',
            state='EH',
            rate=lambda s: s.target.a5 * s.target.EPH * maturation(s),
        ),
        ReactionInteraction(
            'conversion',
            target='site',
            reactants={'EH': 1},
            products={'ER': 1},
            modifiers={'C3': 1},
            propensity=lambda s: s.target.a6 * s.target.EH * s.target.C3 / (s.target.g5 + s.target.C3),
        ),
        ReactionInteraction(
            'helper_death',
            target='site',
            reactants={'EH': 1},
            propensity=lambda s: s.target.d6 * s.target.EH,
        ),
        RateInteraction(
            'regulatory_priming',
            target='site',
            state='EPR',
            rate=lambda s: s.target.a7
            * s.target.EMR
            * s.target.ED
            / (s.target.ED + s.target.g4 * (s.target.EMC + s.target.EMH + s.target.EMR)),
        ),
        ReactionInteraction(
            'regulatory_precursor_death',
            target='site',
            reactants={'EPR': 1},
            propensity=lambda s: s.target.d7 * s.target.EPR,
        ),
        RateInteraction(
            'regulatory_maturation',
            target='site',
            state='ER',
            rate=lambda s: s.target.a8 * s.target.EPR * hill(s.target.C1, s.target.EC1_50),
        ),
        ReactionInteraction(
            'regulatory_death',
            target='site',
            reactants={'ER': 1},
            propensity=lambda s: s.target.d8 * s.target.ER,
        ),
        RateInteraction(
            'il2_secretion',
            target='site',
            state='C1',
            rate=lambda s: s.target.a9
            * s.target.EPH
            / ((1 + s.target.C3 / s.target.CC3_50) * (1 + s.target.C2 / s.target.CC2_50)),
        ),
        ReactionInteraction(
            'il2_loss',
            target='site',
            reactants={'C1': 1},
            propensity=lambda s: s.target.C1 / s.target.tau1,
        ),
        RateInteraction(
            'regulatory_il10',
            target='site',
            state='C2',
            rate=lambda s: s.target.r3 * s.target.ER,
        ),
        RateInteraction(
            'tumour_il10',
            target='site',
            state='C2',
            rate=lambda s: s.target.r4 * s.target.A,
        ),
        ReactionInteraction(
            'il10_loss',
            target='site',
            reactants={'C2': 1},
            propensity=lambda s: s.target.C2 / s.target.tau2,
        ),
        RateInteraction(
            'regulatory_tgf_beta',
            target='site',
            state='C3',
            rate=lambda s: s.target.r5 * s.target.ER,
        ),
        RateInteraction(
            'tumour_tgf_beta',
            target='site',
            state='C3',
            rate=lambda s: s.target.r6 * s.target.A,
        ),
        ReactionInteraction(
            'tgf_beta_loss',
            target='site',
            reactants={'C3': 1},
            propensity=lambda s: s.target.C3 / s.target.tau3,
        ),
    ]
    return System(elements=[site], interactions=interactions)
