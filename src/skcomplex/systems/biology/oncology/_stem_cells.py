"""Oncology systems: normal, malignant, and leukaemic stem cells and the lineages they feed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Literal

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, exp, log

from ....core.components.interactions import RateInteraction, ReactionInteraction
from ....core.components.laws.utils import delayed, pulse
from ....core.components.system import System
from ....core.engine.solvers import ScipyODESolver
from ..._catalog import register_system
from ._bone_marrow_blood import BoneMarrowBloodElement
from ._leukaemia_stem_cells import LeukaemiaStemCellsElement
from ._stem_cell_differentiation import StemCellDifferentiationElement

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


def _normal_renewal(target: SimpleNamespace) -> Expression:
    """Return the renewal rate of normal progenitors, raised by the feedback of the differentiated cells."""
    feedback = target.beta0 * (1 - target.ND / (target.ND + target.kappa0))
    renewal: Expression = (target.r1 + feedback) * (1 - target.NPr / target.NPmax)
    return renewal


def _therapy_deaths(s: SimpleNamespace, therapy: str, start: float, duration: float) -> tuple[Expression, Expression]:
    """Return the death rate of normal progenitors and the factor of the other death rates under a therapy."""
    target = s.target
    if therapy == 'continuous':
        return target.d2 * target.uc * (1 + target.delta * _normal_renewal(target)), target.uc
    if therapy == 'pulsed':
        on = pulse(s.t, start, duration)
        return target.d2 * target.up * (1 + on * target.delta * _normal_renewal(target)), target.up * (1 + 3 * on)
    return target.d2, SR(1)


@register_system(
    field='biology/oncology',
    tags=('stem cells', 'cancer stem cells', 'ode', 'feedback', 'random'),
    references=(BOOK, 'Calmelet et al. 2010'),
    defaults={'until': 2000.0, 'dt': 10.0},
)
def stem_cell_differentiation(
    *,
    m0: float = 0.005,
    beta0: float = 5.0,
    delta: float = 1.0,
    therapy: Literal['none', 'pulsed', 'continuous'] = 'continuous',
    start: float = 100.0,
    duration: float = 100.0,
    seed: int = 0,
) -> System:
    """Return the eight-compartment system of normal and malignant stem-cell lineages with feedback and therapy.

    Each lineage runs from quiescent stem cells through progenitors and basal cells to differentiated cells, and
    normal progenitors mutate into malignant ones. The mutation rate `m1 = m0 (1 + w)` carries the random factor `w`,
    which is drawn once per run from the uniform distribution on `[0, 1)` with the seed. The parameters and the
    initial cells are those of the book's table, with `beta0` at the top of its range. The book gives no value for
    `delta`, so it defaults to 1, and no times for the pulse of a pulsed therapy. The book prints the death rates
    off the pulse as the death rates times the dose, and the system keeps that. Time is in the units of the source.
    Under the continuous therapy the lineages settle on a steady state in which the mutation feeds a small malignant
    population.

    Args:
        m0:
            The mutation rate without the random factor.

        beta0:
            The feedback constant.

        delta:
            The weight of the renewal in the death the therapy causes to normal progenitors.

        therapy:
            The mode of therapy, `'none'`, `'pulsed'`, or `'continuous'`.

        start:
            The time the pulse of a pulsed therapy starts.

        duration:
            The duration of the pulse of a pulsed therapy.

        seed:
            The seed of the random factor of the mutation rate.

    Returns:
        system:
            The system, with the normal cells `tissue.NSq`, `tissue.NPr`, `tissue.NBs`, and `tissue.ND` and the
            malignant cells `tissue.ASq`, `tissue.APr`, `tissue.ABs`, and `tissue.AD`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import stem_cell_differentiation
        >>> 0.005 <= stem_cell_differentiation(seed=1).elements[0].m1 < 0.01
        True
    """
    m1 = m0 * (1 + np.random.default_rng(seed).uniform())
    tissue = StemCellDifferentiationElement(
        'tissue',
        m1=m1,
        beta0=beta0,
        delta=delta,
        therapy=therapy,
        start=start,
        duration=duration,
    )
    interactions = [
        ReactionInteraction(
            'normal_quiescent_death',
            target='tissue',
            reactants={'NSq': 1},
            propensity=lambda s: s.target.d1 * s.target.NSq,
        ),
        ReactionInteraction(
            'normal_activation',
            target='tissue',
            reactants={'NSq': 1},
            products={'NPr': 1},
            propensity=lambda s: s.target.k12 * s.target.NSq,
        ),
        ReactionInteraction(
            'normal_return',
            target='tissue',
            reactants={'NPr': 1},
            products={'NSq': 1},
            propensity=lambda s: s.target.k21 * (1 - s.target.NPr / (s.target.NPr + s.target.NPinf)) * s.target.NPr,
        ),
        RateInteraction(
            'normal_renewal',
            target='tissue',
            state='NPr',
            rate=lambda s: _normal_renewal(s.target) * s.target.NPr,
        ),
        RateInteraction(
            'normal_progenitor_death',
            target='tissue',
            state='NPr',
            rate=lambda s: -_therapy_deaths(s, therapy, start, duration)[0] * s.target.NPr,
        ),
        ReactionInteraction(
            'mutation',
            target='tissue',
            reactants={'NPr': 1},
            products={'APr': 1},
            propensity=lambda s: s.target.m1 * s.target.NPr,
        ),
        ReactionInteraction(
            'normal_commitment',
            target='tissue',
            reactants={'NPr': 1},
            products={'NBs': 1},
            propensity=lambda s: s.target.k23 * s.target.NPr,
        ),
        RateInteraction(
            'normal_basal_death',
            target='tissue',
            state='NBs',
            rate=lambda s: -s.target.d3 * _therapy_deaths(s, therapy, start, duration)[1] * s.target.NBs,
        ),
        ReactionInteraction(
            'normal_differentiation',
            target='tissue',
            reactants={'NBs': 1},
            products={'ND': 1},
            propensity=lambda s: s.target.k34 * s.target.NBs,
        ),
        RateInteraction(
            'normal_basal_growth',
            target='tissue',
            state='NBs',
            rate=lambda s: s.target.r2 * (1 - s.target.NBs / s.target.Nbmax) * s.target.NBs,
        ),
        ReactionInteraction(
            'normal_death',
            target='tissue',
            reactants={'ND': 1},
            propensity=lambda s: s.target.d4 * s.target.ND,
        ),
        ReactionInteraction(
            'malignant_quiescent_death',
            target='tissue',
            reactants={'ASq': 1},
            propensity=lambda s: s.target.d5 * s.target.ASq,
        ),
        ReactionInteraction(
            'malignant_activation',
            target='tissue',
            reactants={'ASq': 1},
            products={'APr': 1},
            propensity=lambda s: s.target.k56 * s.target.ASq,
        ),
        ReactionInteraction(
            'malignant_return',
            target='tissue',
            reactants={'APr': 1},
            products={'ASq': 1},
            propensity=lambda s: s.target.k65 * (1 - s.target.APr / (s.target.APr + s.target.APinf)) * s.target.APr,
        ),
        RateInteraction(
            'malignant_renewal',
            target='tissue',
            state='APr',
            rate=lambda s: s.target.r3 * (1 - s.target.APr / s.target.APmax) * s.target.APr,
        ),
        RateInteraction(
            'malignant_progenitor_death',
            target='tissue',
            state='APr',
            rate=lambda s: -s.target.d6 * _therapy_deaths(s, therapy, start, duration)[1] * s.target.APr,
        ),
        ReactionInteraction(
            'malignant_commitment',
            target='tissue',
            reactants={'APr': 1},
            products={'ABs': 1},
            propensity=lambda s: s.target.k67 * s.target.APr,
        ),
        RateInteraction(
            'malignant_basal_death',
            target='tissue',
            state='ABs',
            rate=lambda s: -s.target.d7 * _therapy_deaths(s, therapy, start, duration)[1] * s.target.ABs,
        ),
        ReactionInteraction(
            'malignant_differentiation',
            target='tissue',
            reactants={'ABs': 1},
            products={'AD': 1},
            propensity=lambda s: s.target.k78 * s.target.ABs,
        ),
        RateInteraction(
            'malignant_basal_growth',
            target='tissue',
            state='ABs',
            rate=lambda s: s.target.r4 * (1 - s.target.ABs / s.target.Abmax) * s.target.ABs,
        ),
        ReactionInteraction(
            'malignant_death',
            target='tissue',
            reactants={'AD': 1},
            propensity=lambda s: s.target.d8 * s.target.AD,
        ),
    ]
    return System(elements=[tissue], interactions=interactions)


@register_system(
    field='biology/oncology',
    tags=('leukaemia', 'stem cells', 'delay', 'bone marrow'),
    references=(BOOK, 'Afenya, Ouifki, and Mundle 2019'),
    defaults={'until': 3000.0, 'dt': 10.0, 'solver': ScipyODESolver(method='BDF')},
)
def bone_marrow_blood_leukaemia(
    *,
    m1: float = 0.0013,
    m3: float = 1e-7,
    tau1: float = 96.0,
    tau2: float = 576.0,
) -> System:
    """Return the bone marrow and blood system of leukaemia, in which blasts crowd out the normal cells.

    Normal marrow cells renew, mutate into blasts and cancer stem cells, and mature into blood cells after the delay
    `tau1`. Blasts grow in the marrow, move to the blood, and inhibit the normal cells, and cancer stem cells convert
    into blasts after the delay `tau2`. Cells are counts, time is in hours, and the history before the start is the
    initial state. The book's table gives ranges, and the defaults are values in them: the largest growth rate of
    normal marrow cells and carrying capacity of blasts, the smallest delays, and the release rate `m2` that keeps
    the normal blood cells near their initial number. The book prints the arrival of marrow cells in the blood with
    a minus sign, and the system uses the plus sign of the release it balances. The blasts take over the marrow and
    the normal blood cells fall by orders of magnitude.

    Args:
        m1:
            The mutation rate of normal marrow cells into blasts, per hour.

        m3:
            The mutation rate of normal marrow cells into cancer stem cells, per hour.

        tau1:
            The maturation time of normal cells, in hours.

        tau2:
            The conversion time of cancer stem cells into blasts, in hours.

    Returns:
        system:
            The system, with the normal marrow and blood cells `patient.NBM` and `patient.NPB`, the blasts
            `patient.ABM` and `patient.APB`, and the cancer stem cells `patient.AS`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import bone_marrow_blood_leukaemia
        >>> bone_marrow_blood_leukaemia(tau1=144.0).elements[0].tau1
        144.0
    """
    patient = BoneMarrowBloodElement('patient', m1=m1, m3=m3, tau1=tau1, tau2=tau2)
    interactions = [
        RateInteraction(
            'marrow_growth',
            target='patient',
            state='NBM',
            rate=lambda s: s.target.r1
            * s.target.NBM
            * log((s.target.b1 + s.target.b2) / (s.target.theta1 + s.target.NBM)),
        ),
        ReactionInteraction(
            'mutation',
            target='patient',
            reactants={'NBM': 1},
            products={'ABM': 1},
            propensity=lambda s: s.target.m1 * s.target.NBM,
        ),
        ReactionInteraction(
            'stem_mutation',
            target='patient',
            reactants={'NBM': 1},
            products={'AS': 1},
            propensity=lambda s: s.target.m3 * s.target.NBM,
        ),
        RateInteraction(
            'release',
            target='patient',
            state='NBM',
            rate=lambda s: -s.target.m2
            * s.target.NBM
            * exp(-s.target.r2 * s.target.APB)
            / (s.target.theta2 + s.target.NPB),
        ),
        RateInteraction(
            'arrival',
            target='patient',
            state='NPB',
            rate=lambda s: s.target.m2
            * delayed(s.target.NBM, s.target.tau1)
            * exp(-s.target.alpha1 * s.target.tau1 - s.target.r2 * delayed(s.target.APB, s.target.tau1))
            / (s.target.theta2 + delayed(s.target.NPB, s.target.tau1)),
        ),
        ReactionInteraction(
            'marrow_competition',
            target='patient',
            reactants={'NBM': 1},
            modifiers={'ABM': 1},
            propensity=lambda s: s.target.c1 * s.target.NBM * s.target.ABM,
        ),
        ReactionInteraction(
            'blood_competition',
            target='patient',
            reactants={'NPB': 1},
            modifiers={'APB': 1},
            propensity=lambda s: s.target.c2 * s.target.NPB * s.target.APB,
        ),
        ReactionInteraction(
            'blood_death',
            target='patient',
            reactants={'NPB': 1},
            propensity=lambda s: s.target.d1 * s.target.NPB,
        ),
        RateInteraction(
            'leukaemic_growth',
            target='patient',
            state='ABM',
            rate=lambda s: s.target.r3 * s.target.ABM * log(s.target.b3 / (s.target.theta3 + s.target.ABM)),
        ),
        ReactionInteraction(
            'leukaemic_release',
            target='patient',
            reactants={'ABM': 1},
            products={'APB': 1},
            propensity=lambda s: s.target.m4 * s.target.ABM,
        ),
        RateInteraction(
            'conversion_arrival',
            target='patient',
            state='ABM',
            rate=lambda s: s.target.m5
            * exp(-s.target.alpha2 * s.target.tau2 - s.target.r4 * delayed(s.target.APB, s.target.tau2))
            * delayed(s.target.AS, s.target.tau2),
        ),
        ReactionInteraction(
            'leukaemic_death',
            target='patient',
            reactants={'APB': 1},
            propensity=lambda s: s.target.d2 * s.target.APB,
        ),
        RateInteraction(
            'stem_growth',
            target='patient',
            state='AS',
            rate=lambda s: s.target.r5 * s.target.AS * (1 - s.target.b4 * s.target.AS),
        ),
        RateInteraction(
            'conversion',
            target='patient',
            state='AS',
            rate=lambda s: -s.target.m5 * exp(-s.target.r4 * s.target.APB) * s.target.AS,
        ),
    ]
    return System(elements=[patient], interactions=interactions)


@register_system(
    field='biology/oncology',
    tags=('leukaemia', 'stem cells', 'ode', 'feedback', 'competition'),
    references=(BOOK, 'MacLean, Lo Celso, and Stumpf 2013'),
    defaults={'until': 500.0, 'dt': 1.0},
)
def leukaemia_stem_cells(
    NS: float = 100.0,
    AS: float = 1.0,
    *,
    r1: float = 0.6,
    m1: float = 0.2,
    d1: float = 0.1,
    r4: float = 0.7,
    m3: float = 0.2,
    d4: float = 0.1,
) -> System:
    """Return normal and leukaemic stem cells that compete for one niche through the feedback of their progeny.

    Both kinds of stem cells renew at the rate `r e^-psi1 - m e^-psi1 e^-psi3`, where `psi1` counts the stem cells
    in the niche and `psi3` their progeny, so the kind with the larger net renewal excludes the other. With the
    defaults the leukaemic stem cells win and take over, and with `r4 = 0.5` the leukaemia is controlled. The book
    gives ranges for the rates, and the defaults are values in them with the feedback gains `0.01` of the book.
    The book gives no initial cells, so a niche of 100 normal stem cells receives one leukaemic stem cell.

    Args:
        NS:
            The initial normal stem cells.

        AS:
            The initial leukaemic stem cells.

        r1:
            The symmetric renewal rate of normal stem cells.

        m1:
            The symmetric differentiation rate of normal stem cells.

        d1:
            The death rate of normal stem cells.

        r4:
            The symmetric renewal rate of leukaemic stem cells.

        m3:
            The symmetric differentiation rate of leukaemic stem cells.

        d4:
            The death rate of leukaemic stem cells.

    Returns:
        system:
            The system, with the normal cells `niche.NS`, `niche.NPr`, and `niche.ND` and the leukaemic cells
            `niche.AS` and `niche.AD`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import leukaemia_stem_cells
        >>> leukaemia_stem_cells(r4=0.5).elements[0].r4
        0.5
    """
    niche = LeukaemiaStemCellsElement('niche', NS=NS, AS=AS, r1=r1, m1=m1, d1=d1, r4=r4, m3=m3, d4=d4)
    interactions = [
        RateInteraction(
            'normal_renewal',
            target='niche',
            state='NS',
            rate=lambda s: s.target.r1 * s.target.NS * exp(-s.target.beta1 * (s.target.NS + s.target.AS)),
        ),
        ReactionInteraction(
            'normal_differentiation',
            target='niche',
            reactants={'NS': 1},
            products={'NPr': 2},
            propensity=lambda s: s.target.m1
            * exp(-s.target.beta3 * (s.target.NPr + s.target.ND + s.target.AD))
            * s.target.NS
            * exp(-s.target.beta1 * (s.target.NS + s.target.AS)),
        ),
        ReactionInteraction(
            'normal_death',
            target='niche',
            reactants={'NS': 1},
            propensity=lambda s: s.target.d1 * s.target.NS,
        ),
        RateInteraction(
            'normal_division',
            target='niche',
            state='NPr',
            rate=lambda s: s.target.r2
            * exp(-s.target.beta2 * (s.target.NPr + s.target.ND + s.target.AD))
            * s.target.NS
            * exp(-s.target.beta1 * (s.target.NS + s.target.AS)),
        ),
        RateInteraction('progenitor_growth', target='niche', state='NPr', rate=lambda s: s.target.r3 * s.target.NPr),
        RateInteraction('progenitor_loss', target='niche', state='NPr', rate=lambda s: -s.target.d2 * s.target.NPr),
        RateInteraction(
            'maturation',
            target='niche',
            state='ND',
            rate=lambda s: s.target.m2 * s.target.d2 * s.target.NPr,
        ),
        ReactionInteraction(
            'differentiated_death',
            target='niche',
            reactants={'ND': 1},
            propensity=lambda s: s.target.d3 * s.target.ND,
        ),
        RateInteraction(
            'leukaemic_renewal',
            target='niche',
            state='AS',
            rate=lambda s: s.target.r4 * s.target.AS * exp(-s.target.beta1 * (s.target.NS + s.target.AS)),
        ),
        ReactionInteraction(
            'leukaemic_differentiation',
            target='niche',
            reactants={'AS': 1},
            products={'AD': 2},
            propensity=lambda s: s.target.m3
            * exp(-s.target.beta3 * (s.target.NPr + s.target.ND + s.target.AD))
            * s.target.AS
            * exp(-s.target.beta1 * (s.target.NS + s.target.AS)),
        ),
        ReactionInteraction(
            'leukaemic_death',
            target='niche',
            reactants={'AS': 1},
            propensity=lambda s: s.target.d4 * s.target.AS,
        ),
        RateInteraction(
            'leukaemic_division',
            target='niche',
            state='AD',
            rate=lambda s: s.target.r5
            * exp(-s.target.beta2 * (s.target.NPr + s.target.ND + s.target.AD))
            * s.target.AS
            * exp(-s.target.beta1 * (s.target.NS + s.target.AS)),
        ),
        ReactionInteraction(
            'leukaemic_progeny_death',
            target='niche',
            reactants={'AD': 1},
            propensity=lambda s: s.target.d5 * s.target.AD,
        ),
    ]
    return System(elements=[niche], interactions=interactions)
