"""Oncology systems: chemotherapy combined with immunotherapy, viruses, antibodies, and antiangiogenic drugs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, exp, log, sin

from ....core.components.interactions import RateInteraction, ReactionInteraction
from ....core.components.laws.utils import where
from ....core.components.system import System
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction
from ._angiogenesis_immunotherapy_tumour import AngiogenesisImmunotherapyTumourElement
from ._chemoimmunotherapy_tumour import ChemoimmunotherapyTumourElement
from ._chemovirotherapy_tumour import ChemovirotherapyTumourElement
from ._clonal_tumour import ClonalTumourElement
from ._glioma_brain import GliomaBrainElement
from ._her2_tumour import Her2TumourElement

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


def _toxicity(concentration: Expression, scale: Expression) -> Expression:
    """Return a toxicity linear in a low concentration and exponential in a high one, `exp(c / scale) - 1`."""
    return exp(concentration / scale) - 1


def _total_toxicity(s: SimpleNamespace) -> Expression:
    """Return the mean toxicity of the antiangiogenic and the immunotherapeutic drug."""
    tumour = s.tumour
    return (_toxicity(tumour.c_an, tumour.tox_an) + _toxicity(tumour.c_im, tumour.tox_im)) / 2


def _effector_kill(target: SimpleNamespace) -> Expression:
    """Return the ratio-dependent kill rate `D = a1 (E^l) / (g1 A^l + E^l)` of tumour cells by effector cells."""
    ratio = abs(target.E) ** target.l
    kill: Expression = target.a1 * ratio / (target.g1 * abs(target.A) ** target.l + ratio)
    return kill


def _infusion_shape(s: SimpleNamespace, infusion: str) -> Expression:
    """Return the shape in time of a constant, exponentially decaying, or periodic infusion."""
    shapes = {'constant': 1, 'exponential': exp(-s.target.d_u * s.t), 'periodic': sin(s.target.d_u * s.t) ** 2}
    shape: Expression = shapes[infusion]
    return shape


def _glia_rate(target: SimpleNamespace) -> Expression:
    """Return the net rate of the glia, whose decline kills neurons."""
    G = target.G
    rate: Expression = (
        target.r1 * G * (1 - target.b * G) - target.c1 * G * target.AU - target.a1 * G * target.U / (target.g1 + G)
    )
    return rate


def _elimination(target: SimpleNamespace, drug: str) -> Expression:
    """Return the elimination rate `k10 = alpha beta / k21` of a drug from its central compartment."""
    alpha, beta, k21 = (getattr(target, f'{name}_{drug}') for name in ('alpha', 'beta', 'k21'))
    elimination: Expression = alpha * beta / k21
    return elimination


def _drug_effect(target: SimpleNamespace, drug: str) -> Expression:
    """Return the Emax effect of a drug, whose half-effect concentration the resistance shifts."""
    c, R, u50 = (getattr(target, f'{name}_{drug}') for name in ('c', 'R', 'u50'))
    effect: Expression = c**target.g / (((1 + target.delta_r * R) * u50) ** target.g + c**target.g)
    return effect


@register_system(
    field='biology/oncology',
    tags=('chemotherapy', 'immunotherapy', 'combination therapy', 'ode', 'dosing'),
    references=(BOOK, 'de Pillis, Gu, and Radunskaya 2006'),
    defaults={'until': 60.0, 'dt': 0.2, 'solver': ScipyODESolver(method='LSODA', rtol=1e-6, atol=1e-6)},
)
def chemoimmunotherapy(
    A: float = 1e6,
    N: float = 5e4,
    E: float = 100.0,
    C: float = 1.1e7,
    *,
    u_til: float = 1e7,
    u_il2: float = 1e7,
    u_chemo: float = 2.0,
    start: float = 7.0,
    period: float = 7.0,
    width: float = 1.0,
    doses: int = 3,
) -> System:
    """Return a tumour with NK cells, CD8 T cells, and lymphocytes under chemotherapy, IL-2, and TIL injection.

    The tumour, implanted in a mouse, grows toward a high equilibrium that the immune cells cannot hold back. The
    doses of chemotherapy shrink it, and the injected tumour-infiltrating lymphocytes clear what the drug leaves. The
    parameters are the table of the book, except the decay rates of the drugs, which the book's table swaps. The
    chemotherapeutic drug decays at 0.9 per day and IL-2 at 10 per day, as in de Pillis et al. 2006. The schedule of
    three weekly one-day doses from day 7 is a choice, since the book gives none. Cells are counted, and time is in
    days.

    Args:
        A:
            The initial number of tumour cells.

        N:
            The initial number of NK cells.

        E:
            The initial number of CD8 T cells.

        C:
            The initial number of circulating lymphocytes.

        u_til:
            The rate of TIL injection during a dose, in cells per day.

        u_il2:
            The rate of IL-2 injection during a dose.

        u_chemo:
            The rate of chemotherapy infusion during a dose.

        start:
            The time of the first dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        doses:
            The number of doses.

    Returns:
        system:
            The system, with the states `tumour.A`, `tumour.N`, `tumour.E`, `tumour.C`, `tumour.I`, and `tumour.M`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import chemoimmunotherapy
        >>> chemoimmunotherapy(u_til=0.0).elements[0].u_til
        0.0
    """
    tumour = ChemoimmunotherapyTumourElement(
        'tumour',
        A=A,
        N=N,
        E=E,
        C=C,
        u_til=u_til,
        u_il2=u_il2,
        u_chemo=u_chemo,
        start=start,
        period=period,
        width=width,
        doses=doses,
    )
    interactions = [
        RateInteraction(
            'tumour_growth',
            target='tumour',
            state='A',
            rate=lambda s: s.target.r1 * s.target.A * (1 - s.target.b1 * s.target.A),
        ),
        ReactionInteraction(
            'nk_kill',
            target='tumour',
            reactants={'A': 1},
            modifiers={'N': 1},
            propensity=lambda s: s.target.c1 * s.target.N * s.target.A,
        ),
        ReactionInteraction(
            'effector_kill',
            target='tumour',
            reactants={'A': 1},
            propensity=lambda s: _effector_kill(s.target) * s.target.A,
        ),
        ReactionInteraction(
            'chemo_tumour_kill',
            target='tumour',
            reactants={'A': 1},
            propensity=lambda s: s.target.a2 * (1 - exp(-s.target.M)) * s.target.A,
        ),
        RateInteraction('nk_production', target='tumour', state='N', rate=lambda s: s.target.m1 * s.target.C),
        ReactionInteraction(
            'nk_death',
            target='tumour',
            reactants={'N': 1},
            propensity=lambda s: s.target.d1 * s.target.N,
        ),
        RateInteraction(
            'nk_recruitment',
            target='tumour',
            state='N',
            rate=lambda s: s.target.r2 * s.target.A**2 / (s.target.g2 + s.target.A**2) * s.target.N,
        ),
        ReactionInteraction(
            'nk_inactivation',
            target='tumour',
            reactants={'N': 1},
            modifiers={'A': 1},
            propensity=lambda s: s.target.psi1 * s.target.N * s.target.A,
        ),
        ReactionInteraction(
            'chemo_nk_kill',
            target='tumour',
            reactants={'N': 1},
            propensity=lambda s: s.target.a3 * (1 - exp(-s.target.M)) * s.target.N,
        ),
        RateInteraction(
            'effector_recruitment',
            target='tumour',
            state='E',
            rate=lambda s: s.target.r3
            * (_effector_kill(s.target) * s.target.A) ** 2
            / (s.target.g3 + (_effector_kill(s.target) * s.target.A) ** 2)
            * s.target.E,
        ),
        ReactionInteraction(
            'effector_death',
            target='tumour',
            reactants={'E': 1},
            propensity=lambda s: s.target.d2 * s.target.E,
        ),
        ReactionInteraction(
            'effector_inactivation',
            target='tumour',
            reactants={'E': 1},
            modifiers={'A': 1},
            propensity=lambda s: s.target.psi2 * s.target.E * s.target.A,
        ),
        ReactionInteraction(
            'nk_regulation',
            target='tumour',
            reactants={'E': 1},
            propensity=lambda s: s.target.psi3 * s.target.N * s.target.E**2,
        ),
        ReactionInteraction(
            'chemo_effector_kill',
            target='tumour',
            reactants={'E': 1},
            propensity=lambda s: s.target.a4 * (1 - exp(-s.target.M)) * s.target.E,
        ),
        RateInteraction(
            'effector_activation',
            target='tumour',
            state='E',
            rate=lambda s: (s.target.r4 * s.target.N + s.target.r5 * s.target.C) * s.target.A,
        ),
        MichaelisMentenInteraction(
            'il2_proliferation',
            target='tumour',
            products={'E': 1},
            substrate='I',
            vmax=lambda s: s.target.r6,
            km=lambda s: s.target.g4,
            modifiers={'E': 1},
        ),
        DosingInteraction(
            'til_infusion',
            target='tumour',
            state='E',
            dose=lambda s: s.target.u_til,
            period=tumour.period,
            width=tumour.width,
            start=tumour.start,
            count=tumour.doses,
        ),
        RateInteraction('lymphocyte_production', target='tumour', state='C', rate=lambda s: s.target.s1),
        ReactionInteraction(
            'lymphocyte_death',
            target='tumour',
            reactants={'C': 1},
            propensity=lambda s: s.target.d3 * s.target.C,
        ),
        ReactionInteraction(
            'chemo_lymphocyte_kill',
            target='tumour',
            reactants={'C': 1},
            propensity=lambda s: s.target.a5 * (1 - exp(-s.target.M)) * s.target.C,
        ),
        ReactionInteraction(
            'il2_clearance',
            target='tumour',
            reactants={'I': 1},
            propensity=lambda s: s.target.d_u1 * s.target.I,
        ),
        DosingInteraction(
            'il2_infusion',
            target='tumour',
            state='I',
            dose=lambda s: s.target.u_il2,
            period=tumour.period,
            width=tumour.width,
            start=tumour.start,
            count=tumour.doses,
        ),
        ReactionInteraction(
            'chemo_clearance',
            target='tumour',
            reactants={'M': 1},
            propensity=lambda s: s.target.d_u2 * s.target.M,
        ),
        DosingInteraction(
            'chemo_infusion',
            target='tumour',
            state='M',
            dose=lambda s: s.target.u_chemo,
            period=tumour.period,
            width=tumour.width,
            start=tumour.start,
            count=tumour.doses,
        ),
    ]
    return System(elements=[tumour], interactions=interactions)


@register_system(
    field='biology/oncology',
    tags=('chemotherapy', 'immunotherapy', 'combination therapy', 'heterogeneity', 'resistance', 'ode'),
    references=(BOOK, 'Piretto, Delitala, and Ferraro 2018'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def clonal_chemoimmunotherapy(
    A1: float = 9e4,
    A2: float = 4e4,
    E: float = 500.0,
    *,
    c1: float = 0.1,
    c3: float = 0.1,
    u_chemo1: float = 0.2,
    u_chemo2: float = 0.02,
    u_vaccine: float = 20.0,
) -> System:
    """Return two competing tumour clones, sensitive and resistant to chemotherapy, under a drug and a vaccine.

    The chemotherapy kills the sensitive first clone ten times faster than the resistant second clone. Alone, it
    removes the sensitive clone and frees the resistant clone from competition, so the tumour relapses resistant. With
    the dendritic-cell vaccine, the recruited immune cells clear both clones. The book does not system the drugs, so
    their effects are constant rates, and their values are a choice. The competition rates take a value in the book's
    range of 0.01 to 1. Cells are counted, and time is in days.

    Args:
        A1:
            The initial number of cells of the sensitive clone.

        A2:
            The initial number of cells of the resistant clone.

        E:
            The initial number of immune cells.

        c1:
            The competition of the second clone on the first.

        c3:
            The competition of the first clone on the second.

        u_chemo1:
            The chemotherapy kill rate of the first clone.

        u_chemo2:
            The chemotherapy kill rate of the second clone.

        u_vaccine:
            The level of the immunotherapeutic agent.

    Returns:
        system:
            The system, with the clones `tumour.A1` and `tumour.A2` and the immune cells `tumour.E`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import clonal_chemoimmunotherapy
        >>> clonal_chemoimmunotherapy(u_vaccine=0.0).elements[0].u_vaccine
        0.0
    """
    tumour = ClonalTumourElement(
        'tumour',
        A1=A1,
        A2=A2,
        E=E,
        c1=c1,
        c3=c3,
        u_chemo1=u_chemo1,
        u_chemo2=u_chemo2,
        u_vaccine=u_vaccine,
    )
    interactions = [
        RateInteraction(
            'first_growth',
            target='tumour',
            state='A1',
            rate=lambda s: s.target.r1 * s.target.A1 * (1 - s.target.b1 * s.target.A1),
        ),
        ReactionInteraction(
            'first_competition',
            target='tumour',
            reactants={'A1': 1},
            modifiers={'A2': 1},
            propensity=lambda s: s.target.c1 * s.target.A2 * s.target.b1 * s.target.A1,
        ),
        ReactionInteraction(
            'first_immune_kill',
            target='tumour',
            reactants={'A1': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.c2 * s.target.E * s.target.b1 * s.target.A1,
        ),
        ReactionInteraction(
            'first_chemotherapy',
            target='tumour',
            reactants={'A1': 1},
            propensity=lambda s: s.target.u_chemo1 * s.target.A1,
        ),
        ReactionInteraction(
            'first_vaccine_kill',
            target='tumour',
            reactants={'A1': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.b1 * s.target.u_vaccine * s.target.A1 * s.target.E,
        ),
        RateInteraction(
            'second_growth',
            target='tumour',
            state='A2',
            rate=lambda s: s.target.r2 * s.target.A2 * (1 - s.target.b2 * s.target.A2),
        ),
        ReactionInteraction(
            'second_competition',
            target='tumour',
            reactants={'A2': 1},
            modifiers={'A1': 1},
            propensity=lambda s: s.target.c3 * s.target.A1 * s.target.b2 * s.target.A2,
        ),
        ReactionInteraction(
            'second_immune_kill',
            target='tumour',
            reactants={'A2': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.c4 * s.target.E * s.target.b2 * s.target.A2,
        ),
        ReactionInteraction(
            'second_chemotherapy',
            target='tumour',
            reactants={'A2': 1},
            propensity=lambda s: s.target.u_chemo2 * s.target.A2,
        ),
        ReactionInteraction(
            'second_vaccine_kill',
            target='tumour',
            reactants={'A2': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.b2 * s.target.u_vaccine * s.target.A2 * s.target.E,
        ),
        RateInteraction(
            'immune_growth',
            target='tumour',
            state='E',
            rate=lambda s: s.target.r3 * s.target.E * (1 - s.target.E / s.target.theta),
        ),
        RateInteraction(
            'immune_stimulation',
            target='tumour',
            state='E',
            rate=lambda s: (s.target.s1 * s.target.A1 + s.target.s2 * s.target.A2) * s.target.E / s.target.theta,
        ),
    ]
    return System(elements=[tumour], interactions=interactions)


@register_system(
    field='biology/oncology',
    tags=('chemotherapy', 'virotherapy', 'combination therapy', 'oncolytic virus', 'ode'),
    references=(BOOK, 'Malinzi 2019'),
    defaults={'until': 32.0, 'dt': 0.1},
)
def chemovirotherapy(
    AU: float = 1.0,
    V: float = 0.1,
    U: float = 0.1,
    *,
    u1: float = 10.0,
    infusion: str = 'constant',
) -> System:
    """Return uninfected and infected tumour cells, an oncolytic virus, and a chemotherapeutic drug.

    The virus infects the growing tumour, and the drug kills both uninfected and infected cells. Over the default run
    of 64 days, before the virus bursts through the tumour, the combination leaves the smallest tumour. Later the virus
    alone clears the tumour slightly faster, because the drug also kills the infected cells that make the virus. The
    infusion rate is a choice, since the book gives none. One unit of cells is a million cells, and one unit of time is
    two days.

    Args:
        AU:
            The initial number of uninfected cells.

        V:
            The initial number of virions, 0 for chemotherapy alone.

        U:
            The initial amount of drug.

        u1:
            The infusion rate of the drug, 0 for virotherapy alone.

        infusion:
            The form of the infusion, `'constant'`, `'exponential'`, or `'periodic'`.

    Returns:
        system:
            The system, with the cells `tumour.AU` and `tumour.AV`, the virus `tumour.V`, and the drug `tumour.U`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import chemovirotherapy
        >>> chemovirotherapy(infusion='exponential').elements[0].infusion
        'exponential'
    """
    tumour = ChemovirotherapyTumourElement('tumour', AU=AU, V=V, U=U, u1=u1, infusion=infusion)
    interactions = [
        RateInteraction(
            'tumour_growth',
            target='tumour',
            state='AU',
            rate=lambda s: s.target.r1 * s.target.AU * (1 - s.target.b * (s.target.AU + s.target.AV)),
        ),
        ReactionInteraction(
            'infection',
            target='tumour',
            reactants={'AU': 1, 'V': 1},
            products={'AV': 1},
            propensity=lambda s: s.target.alpha * s.target.AU * s.target.V,
        ),
        ReactionInteraction(
            'drug_kill',
            target='tumour',
            reactants={'AU': 1},
            modifiers={'U': 1},
            propensity=lambda s: s.target.a1 * s.target.AU * s.target.U,
        ),
        ReactionInteraction(
            'lysis',
            target='tumour',
            reactants={'AV': 1},
            propensity=lambda s: s.target.a2 * s.target.AV,
        ),
        ReactionInteraction(
            'infected_drug_kill',
            target='tumour',
            reactants={'AV': 1},
            modifiers={'U': 1},
            propensity=lambda s: s.target.a3 * s.target.AV * s.target.U,
        ),
        RateInteraction(
            'burst',
            target='tumour',
            state='V',
            rate=lambda s: s.target.beta_v * s.target.a2 * s.target.AV,
        ),
        ReactionInteraction(
            'virus_clearance',
            target='tumour',
            reactants={'V': 1},
            propensity=lambda s: s.target.d1 * s.target.V,
        ),
        ReactionInteraction(
            'drug_clearance',
            target='tumour',
            reactants={'U': 1},
            propensity=lambda s: s.target.d_u * s.target.U,
        ),
        RateInteraction(
            'infusion',
            target='tumour',
            state='U',
            rate=lambda s: s.target.u1 * _infusion_shape(s, infusion),
        ),
    ]
    return System(elements=[tumour], interactions=interactions)


@register_system(
    field='biology/oncology',
    tags=('chemotherapy', 'virotherapy', 'combination therapy', 'glioma', 'neurons', 'ode'),
    references=(BOOK, 'Urenda-Cázares, Gallegos, and Macías-Díaz 2020', 'Iarosz et al. 2015'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def glioma_chemovirotherapy(
    AU: float = 6e10,
    V: float = 1e9,
    *,
    u1: float = 400.0,
) -> System:
    """Return glia, neurons, and a glioma under a constant temozolomide infusion and an oncolytic virus.

    Untreated, the glioma grows and crowds out the glia, and the neurons die with the glia. The virus or the drug clears
    the glioma, and the neurons survive. The drug also kills some neurons. The book gives no initial virus or drug, so
    the run starts with no drug and with a billion virions. Cells and virions are counted, the drug is in mg per m², and
    time is in days.

    Args:
        AU:
            The initial number of glioma cells.

        V:
            The initial number of virions, 0 for chemotherapy alone.

        u1:
            The infusion rate of temozolomide, in mg per m² per day, 0 for virotherapy alone.

    Returns:
        system:
            The system, with the glia `brain.G`, the neurons `brain.N`, the glioma `brain.AU` and `brain.AV`, the virus
            `brain.V`, and the drug `brain.U`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import glioma_chemovirotherapy
        >>> glioma_chemovirotherapy(u1=0.0).elements[0].u1
        0.0
    """
    brain = GliomaBrainElement('brain', AU=AU, V=V, u1=u1)
    interactions = [
        RateInteraction(
            'glia_growth',
            target='brain',
            state='G',
            rate=lambda s: s.target.r1 * s.target.G * (1 - s.target.b * s.target.G),
        ),
        ReactionInteraction(
            'glia_competition',
            target='brain',
            reactants={'G': 1},
            modifiers={'AU': 1},
            propensity=lambda s: s.target.c1 * s.target.G * s.target.AU,
        ),
        MichaelisMentenInteraction(
            'glia_drug_kill',
            target='brain',
            reactants={'G': 1},
            substrate='G',
            vmax=lambda s: s.target.a1,
            km=lambda s: s.target.g1,
            modifiers={'U': 1},
        ),
        ReactionInteraction(
            'neuron_loss',
            target='brain',
            reactants={'N': 1},
            propensity=lambda s: -s.target.psi * where(_glia_rate(s.target) < 0, _glia_rate(s.target), 0) * s.target.N,
        ),
        MichaelisMentenInteraction(
            'neuron_drug_kill',
            target='brain',
            reactants={'N': 1},
            substrate='N',
            vmax=lambda s: s.target.a2,
            km=lambda s: s.target.g2,
            modifiers={'U': 1},
        ),
        RateInteraction(
            'glioma_growth',
            target='brain',
            state='AU',
            rate=lambda s: s.target.r2 * s.target.AU * (1 - s.target.b * (s.target.AU + s.target.AV)),
        ),
        ReactionInteraction(
            'glioma_competition',
            target='brain',
            reactants={'AU': 1},
            modifiers={'G': 1},
            propensity=lambda s: s.target.c2 * s.target.G * s.target.AU,
        ),
        MichaelisMentenInteraction(
            'glioma_drug_kill',
            target='brain',
            reactants={'AU': 1},
            substrate='AU',
            vmax=lambda s: s.target.a3,
            km=lambda s: s.target.g3,
            modifiers={'U': 1},
        ),
        ReactionInteraction(
            'infection',
            target='brain',
            reactants={'AU': 1, 'V': 1},
            products={'AV': 1},
            propensity=lambda s: s.target.alpha * s.target.AU * s.target.V,
        ),
        ReactionInteraction(
            'lysis',
            target='brain',
            reactants={'AV': 1},
            propensity=lambda s: s.target.a4 * s.target.AV,
        ),
        RateInteraction('burst', target='brain', state='V', rate=lambda s: s.target.beta_v * s.target.a4 * s.target.AV),
        ReactionInteraction(
            'virus_clearance',
            target='brain',
            reactants={'V': 1},
            propensity=lambda s: s.target.d_u1 * s.target.V,
        ),
        ReactionInteraction(
            'drug_clearance',
            target='brain',
            reactants={'U': 1},
            propensity=lambda s: s.target.d_u2 * s.target.U,
        ),
        DosingInteraction('infusion', target='brain', state='U', dose=lambda s: s.target.u1),
    ]
    return System(elements=[brain], interactions=interactions)


@register_system(
    field='biology/oncology',
    tags=('chemotherapy', 'targeted therapy', 'combination therapy', 'HER2', 'antibody', 'ode'),
    references=(BOOK, 'Jarrett et al. 2019'),
    defaults={'until': 10.0, 'dt': 0.02, 'solver': ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)},
)
def her2_targeted_therapy(
    A: float = 0.035,
    *,
    antibody: float = 0.2,
    paclitaxel: float = 2.0,
    t_antibody: float = 1.0,
    t_a: float = 2.0,
    t_p: float = 1.0,
    sigma: float = 1.0,
) -> System:
    """Return HER2+ breast cancer cells under trastuzumab and paclitaxel, with antibody binding and drug uptake.

    The free antibody and the free paclitaxel are drivers held at their levels from their doses on. The antibody binds
    the receptors, which are in excess, so the bound antibody rises to the free antibody, and from the time `t_a` on it
    lowers the growth rate to `r (1 - a1 Ab)`. The paclitaxel enters the cells and briefly lowers the carrying capacity.
    The cytotoxicity falls from one at the dose, so `t_p` is the dose time. The book's units cannot hold 35,000 cells
    with a capacity of 0.67, so cells are counted in millions and the antibody in millions of molecules. The dose levels
    and times are a choice. Time is in days.

    Args:
        A:
            The initial number of cells, in millions.

        antibody:
            The level of free antibody from its dose on.

        paclitaxel:
            The level of free paclitaxel from its dose on, in pmol.

        t_antibody:
            The time of the antibody dose.

        t_a:
            The time from which the bound antibody inhibits proliferation.

        t_p:
            The time of the paclitaxel dose.

        sigma:
            The synergy of the two drugs, 1 for none.

    Returns:
        system:
            The system, with the cells `tumour.A`, the bound antibody `tumour.Ab`, and the internalized paclitaxel
            `tumour.Pin`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import her2_targeted_therapy
        >>> her2_targeted_therapy(sigma=1.5).elements[0].sigma
        1.5
    """
    tumour = Her2TumourElement(
        'tumour',
        A=A,
        antibody=antibody,
        paclitaxel=paclitaxel,
        t_antibody=t_antibody,
        t_a=t_a,
        t_p=t_p,
        sigma=sigma,
    )
    return System(elements=[tumour])


@register_system(
    field='biology/oncology',
    tags=('antiangiogenic therapy', 'immunotherapy', 'combination therapy', 'pharmacokinetics', 'resistance', 'ode'),
    references=(BOOK, 'Teles and Lemos 2019', 'Hahnfeldt et al. 1999'),
    defaults={'until': 150.0, 'dt': 0.25},
    observables=lambda system: [ObservableSpec('toxicity', _total_toxicity)],
)
def angiogenesis_immunotherapy(
    A: float = 5000.0,
    Vb: float = 625.0,
    E: float = 0.1,
    *,
    dose_an: float = 15.0,
    dose_im: float = 20.0,
    a2: float = 1.35,
    a3: float = 0.1,
    period: float = 21.0,
    doses: int = 6,
) -> System:
    """Return the Hahnfeldt tumour with effector cells under bevacizumab and atezolizumab every three weeks.

    Each drug alone holds the tumour or slows it, and together they shrink it. The drug amounts follow the
    two-compartment system whose bolus solution is the book's closed form, with the elimination rate
    `k10 = alpha beta / k21`. The effector cells grow as `r1 (A - c1 A^2) E`, so `1 / c1` is the volume above which the
    tumour suppresses them. The drug effects in the vasculature and effector balances are the saturating effects of
    the book's pharmacodynamics. The resistance capacity `delta_r = 0.01` and its threshold `u_theta = 5` mg per kg
    are a choice, since the book gives none. The toxicity of each drug is `exp(c / tox) - 1`, linear for a low amount
    and exponential for a high one, and the observable `toxicity` is the mean of both. Volumes are in mm³, drug amounts
    in mg per kg, and time is in days.

    Args:
        A:
            The initial tumour volume.

        Vb:
            The initial vascular capacity.

        E:
            The initial effector cells, dimensionless.

        dose_an:
            The dose of bevacizumab, in mg per kg, 0 for immunotherapy alone.

        dose_im:
            The dose of atezolizumab, in mg per kg, 0 for antiangiogenic therapy alone.

        a2:
            The highest destruction rate of the vasculature by bevacizumab, in the book's range of 1.35 to 11.25.

        a3:
            The highest boost of effector cells by atezolizumab, in the book's range of 0.1 to 0.5.

        period:
            The time between doses.

        doses:
            The number of doses.

    Returns:
        system:
            The system, with the tumour `tumour.A`, the capacity `tumour.Vb`, the effector cells `tumour.E`, the drug
            amounts `tumour.c_an` and `tumour.c_im`, and the observable `toxicity`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import angiogenesis_immunotherapy
        >>> angiogenesis_immunotherapy(dose_an=0.0).elements[0].dose_an
        0.0
    """
    tumour = AngiogenesisImmunotherapyTumourElement(
        'tumour',
        A=A,
        Vb=Vb,
        E=E,
        dose_an=dose_an,
        dose_im=dose_im,
        a2=a2,
        a3=a3,
        period=period,
        doses=doses,
    )
    interactions = [
        RateInteraction(
            'tumour_growth',
            target='tumour',
            state='A',
            rate=lambda s: -s.target.v1 * s.target.A * log(s.target.A / s.target.Vb),
        ),
        ReactionInteraction(
            'immune_kill',
            target='tumour',
            reactants={'A': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.a1 * s.target.A * s.target.E,
        ),
        ReactionInteraction(
            'vessel_loss',
            target='tumour',
            reactants={'Vb': 1},
            propensity=lambda s: s.target.v2 * s.target.Vb,
        ),
        RateInteraction('stimulation', target='tumour', state='Vb', rate=lambda s: s.target.xi_s * s.target.A),
        ReactionInteraction(
            'inhibition',
            target='tumour',
            reactants={'Vb': 1},
            propensity=lambda s: s.target.xi_i * s.target.Vb * s.target.A ** (2 / 3),
        ),
        ReactionInteraction(
            'pruning',
            target='tumour',
            reactants={'Vb': 1},
            propensity=lambda s: s.target.a2 * s.target.Vb * _drug_effect(s.target, 'an'),
        ),
        RateInteraction(
            'immune_growth',
            target='tumour',
            state='E',
            rate=lambda s: s.target.r1 * (s.target.A - s.target.c1 * s.target.A**2) * s.target.E,
        ),
        ReactionInteraction(
            'immune_death',
            target='tumour',
            reactants={'E': 1},
            propensity=lambda s: s.target.d1 * s.target.E,
        ),
        RateInteraction('immune_source', target='tumour', state='E', rate=lambda s: s.target.s_e),
        RateInteraction(
            'immune_boost',
            target='tumour',
            state='E',
            rate=lambda s: s.target.a3 * s.target.E * _drug_effect(s.target, 'im'),
        ),
        DosingInteraction(
            'dosing_an',
            target='tumour',
            state='c_an',
            dose=lambda s: s.target.dose_an / tumour.width,
            period=tumour.period,
            width=tumour.width,
            count=tumour.doses,
        ),
        ReactionInteraction(
            'elimination_an',
            target='tumour',
            reactants={'c_an': 1},
            propensity=lambda s: _elimination(s.target, 'an') * s.target.c_an,
        ),
        ReactionInteraction(
            'distribution_an',
            target='tumour',
            reactants={'c_an': 1},
            products={'q_an': 1},
            propensity=lambda s: s.target.k12_an * s.target.c_an,
        ),
        ReactionInteraction(
            'redistribution_an',
            target='tumour',
            reactants={'q_an': 1},
            products={'c_an': 1},
            propensity=lambda s: s.target.k21_an * s.target.q_an,
        ),
        RateInteraction(
            'resistance_an',
            target='tumour',
            state='R_an',
            rate=lambda s: where(s.target.c_an < s.target.u_theta, s.target.u_theta - s.target.c_an, 0),
        ),
        DosingInteraction(
            'dosing_im',
            target='tumour',
            state='c_im',
            dose=lambda s: s.target.dose_im / tumour.width,
            period=tumour.period,
            width=tumour.width,
            count=tumour.doses,
        ),
        ReactionInteraction(
            'elimination_im',
            target='tumour',
            reactants={'c_im': 1},
            propensity=lambda s: _elimination(s.target, 'im') * s.target.c_im,
        ),
        ReactionInteraction(
            'distribution_im',
            target='tumour',
            reactants={'c_im': 1},
            products={'q_im': 1},
            propensity=lambda s: s.target.k12_im * s.target.c_im,
        ),
        ReactionInteraction(
            'redistribution_im',
            target='tumour',
            reactants={'q_im': 1},
            products={'c_im': 1},
            propensity=lambda s: s.target.k21_im * s.target.q_im,
        ),
        RateInteraction(
            'resistance_im',
            target='tumour',
            state='R_im',
            rate=lambda s: where(s.target.c_im < s.target.u_theta, s.target.u_theta - s.target.c_im, 0),
        ),
    ]
    return System(elements=[tumour], interactions=interactions)
