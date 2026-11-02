"""Oncology models: prostate and breast cancers under hormone therapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, log

from ....core.interactions import BaseInteraction, RateInteraction, ReactionInteraction
from ....core.laws.utils import hill, pulse_train
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ....errors import ValidationError
from ..._catalog import register_model
from ..common.pharmacokinetics import DosingInteraction
from ._androgen_receptor_prostate import AndrogenReceptorProstateElement
from ._androgen_suppression import AndrogenSuppressionElement
from ._cell_quota_prostate import CellQuotaProstateElement
from ._intermittent_suppression import IntermittentSuppressionElement
from ._piecewise_linear_prostate import PiecewiseLinearProstateElement
from ._tamoxifen_tumour import TamoxifenTumourElement
from ._therapy_switch import TherapySwitchInteraction

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


PATIENT_FACTORS = {
    1: {
        'on': (0.956156, 0.00130716, 0.995718, 0.0, 0.0, 1.00181),
        'off': (1.00223, 0.1, 1.00392, 0.8),
        'cells': (17.84, 0.778, 0.382),
    },
    2: {
        'on': (0.925733, 0.00737385, 0.990242, 0.000130721, 0.0, 1.00339),
        'off': (1.00089, 0.1, 1.00147, 1.00079),
        'cells': (13.43, 1.57, 0.0),
    },
    3: {
        'on': (0.931568, 0.0, 0.992614, 0.00083635, 0.0, 1.00242),
        'off': (1.00306, 0.1, 1.0025, 1.00282),
        'cells': (11.02, 0.976, 0.0),
    },
}


def _prostate_volume(target: SimpleNamespace, t: Expression) -> Expression:
    """Return the prostate volume in cm³ at the age of the patient, `v1 a^phi / (v2^phi + a^phi) + v3`."""
    age = target.age + t / 8760
    volume: Expression = target.v1 * age**target.phi / (target.v2**target.phi + age**target.phi) + target.v3
    return volume


def _receptor_growth(target: SimpleNamespace, psi1: Expression, phi1: Expression) -> Expression:
    """Return the growth rate that the receptors activated by DHT and testosterone drive."""
    UD, UT = target.UD, target.UT
    growth: Expression = psi1 * UD**2 / (target.psi2**2 + UD**2) + phi1 * UT**2 / (target.phi2**2 + UT**2)
    return growth


def _receptor_death(target: SimpleNamespace, psi3: Expression) -> Expression:
    """Return the death rate that the activated receptors lower."""
    death: Expression = psi3 / (target.psi4 + target.UD + target.UT)
    return death


def _dependent_division(target: SimpleNamespace) -> Expression:
    """Return the division of androgen-dependent cells, crowded by the normal and the cancer cells."""
    crowding = (target.c1 * target.N + target.AD + target.AI) / target.b2
    division: Expression = _receptor_growth(target, target.psi1_d, target.phi1_d) * target.AD * (1 - crowding)
    return division


def _quota_uptake(target: SimpleNamespace, Q: Expression, q: Expression) -> Expression:
    """Return the uptake of androgen into a quota, which slows as the quota fills."""
    uptake: Expression = target.vm * (target.qm - Q) / (target.qm - q) * target.U / (target.U + target.v50)
    return uptake


def _room(target: SimpleNamespace) -> Expression:
    """Return the Gompertz room `log(Asat / (Ag + As + Am))` that slows the passage through the cell cycle."""
    room: Expression = log(target.Asat / (target.Ag + target.As + target.Am))
    return room


def _get_switches(signal: Callable[[SimpleNamespace], object], h0: float, h1: float) -> list[BaseInteraction]:
    """Return the relay with hysteresis that starts the therapy above `h1` and stops it below `h0` of a signal."""
    return [
        TherapySwitchInteraction('on', target='tumour', signal=signal, threshold=h1, value=1.0, direction=1),
        TherapySwitchInteraction('off', target='tumour', signal=signal, threshold=h0, value=0.0, direction=-1),
    ]


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'androgen'),
    references=(BOOK, 'Zazoua and Wang 2019'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def continuous_androgen_suppression(
    AD: float = 10.0,
    AI: float = 0.14,
    U: float = 20.0,
    *,
    r1: float = 0.025,
    r2: float = 0.006,
    b: float = 0.091e-9,
    c1: float = 0.9,
    c2: float = 0.8,
    d1: float = 0.064,
    m1: float = 5e-5,
    U0: float = 20.0,
    du: float = 0.08,
    a: float = 0.9,
) -> System:
    """Return prostate cancer under continuous androgen suppression, which androgen-independent cells survive.

    The suppression lowers the serum androgen from `U0` to `U0 (1 - a)` at the rate `du`. The androgen-dependent cells
    grow in proportion to the androgen and die and turn independent in proportion to the deprivation `1 - U / U0`, so
    they decline once the suppression is deep enough, while the independent cells grow regardless and cause the
    relapse. Cells are counted, androgen is in ng/mL, and time is in days. The book's efficacy lies between 0.6 and
    0.9, and only the upper end makes the dependent cells decline.

    Args:
        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        U:
            The initial serum androgen.

        r1:
            The growth rate of the dependent cells per unit of androgen.

        r2:
            The growth rate of the independent cells.

        b:
            The reciprocal carrying capacity.

        c1:
            The competition of the independent cells on the dependent ones.

        c2:
            The competition of the dependent cells on the independent ones.

        d1:
            The death rate of the dependent cells under androgen deprivation.

        m1:
            The rate at which androgen deprivation turns dependent cells independent.

        U0:
            The normal androgen concentration.

        du:
            The elimination rate of the androgen.

        a:
            The efficacy of the suppression.

    Returns:
        system:
            The model, with the cells `tumour.AD` and `tumour.AI` and the androgen `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import continuous_androgen_suppression
        >>> continuous_androgen_suppression(a=0.6).elements[0].a
        0.6
    """
    tumour = AndrogenSuppressionElement(
        'tumour',
        AD=AD,
        AI=AI,
        U=U,
        r1=r1,
        r2=r2,
        b=b,
        c1=c1,
        c2=c2,
        d1=d1,
        m1=m1,
        U0=U0,
        du=du,
        a=a,
    )
    interactions = [
        RateInteraction(
            'dependent_growth',
            target='tumour',
            state='AD',
            rate=lambda s: s.target.r1
            * s.target.U
            * (1 - s.target.b * (s.target.AD + s.target.c1 * s.target.AI))
            * s.target.AD,
        ),
        ReactionInteraction(
            'deprivation_death',
            target='tumour',
            reactants={'AD': 1},
            propensity=lambda s: s.target.d1 * (1 - s.target.U / s.target.U0) * s.target.AD,
        ),
        ReactionInteraction(
            'mutation',
            target='tumour',
            reactants={'AD': 1},
            products={'AI': 1},
            propensity=lambda s: s.target.m1 * (1 - s.target.U / s.target.U0) * s.target.AD,
        ),
        RateInteraction(
            'independent_growth',
            target='tumour',
            state='AI',
            rate=lambda s: s.target.r2 * (1 - s.target.b * (s.target.AI + s.target.c2 * s.target.AD)) * s.target.AI,
        ),
        RateInteraction(
            'androgen_production',
            target='tumour',
            state='U',
            rate=lambda s: -s.target.du * (s.target.U - s.target.U0),
        ),
        RateInteraction(
            'suppression',
            target='tumour',
            state='U',
            rate=lambda s: -s.target.U0 * s.target.du * s.target.a,
        ),
    ]
    return System(elements=[tumour], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'events', 'intermittent'),
    references=(BOOK, 'Ideta et al. 2008'),
    defaults={'until': 600.0, 'dt': 2.0},
)
def ideta_intermittent_androgen(
    AD: float = 5.0,
    AI: float = 0.0,
    *,
    r1: float = 0.0204,
    r2: float = 0.0242,
    d1: float = 0.0076,
    d2: float = 0.0168,
    m1: float = 5e-5,
    k1: float = 0.0,
    k2: float = 2.0,
    k3: float = 8.0,
    k4: float = 0.5,
    U0: float = 30.0,
    du: float = 0.08,
    h0: float = 10.0,
    h1: float = 15.0,
) -> System:
    """Return the Ideta model of intermittent androgen suppression, switched on and off by PSA thresholds.

    The therapy starts when the serum PSA, the number of cancer cells, rises through `h1` and stops when it falls
    through `h0`. The androgen follows with the delay `1 / du`, so the dependent cells shrink during therapy and regrow
    after it, and the PSA cycles between the thresholds while the independent cells stay few. Cells are counted, the
    PSA is in ng/mL, the androgen in nmol/L, starting at its normal level `U0` within the book's 8 to 35, and time is
    in days.

    Args:
        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        r1:
            The growth rate of the dependent cells.

        r2:
            The growth rate of the independent cells.

        d1:
            The death rate of the dependent cells.

        d2:
            The death rate of the independent cells.

        m1:
            The maximum rate at which dependent cells turn independent.

        k1:
            The growth of the dependent cells without androgen, relative to `r1`.

        k2:
            The androgen at which the growth of the dependent cells is half way.

        k3:
            The death of the dependent cells without androgen, relative to `d1`.

        k4:
            The androgen at which the death of the dependent cells is half way.

        U0:
            The normal androgen concentration.

        du:
            The elimination rate of the androgen.

        h0:
            The PSA below which the therapy stops.

        h1:
            The PSA above which the therapy starts.

    Returns:
        system:
            The model, with the cells `tumour.AD` and `tumour.AI`, the androgen `tumour.U`, the therapy `tumour.u`, and
            the switches as `on` and `off` events.

    Examples:
        >>> from skcomplex.models.biology.oncology import ideta_intermittent_androgen
        >>> ideta_intermittent_androgen(h1=20.0).interactions[-2].threshold
        20.0
    """
    tumour = IntermittentSuppressionElement(
        'tumour',
        AD=AD,
        AI=AI,
        U=U0,
        r1=r1,
        r2=r2,
        d1=d1,
        d2=d2,
        m1=m1,
        k1=k1,
        k2=k2,
        k3=k3,
        k4=k4,
        U0=U0,
        du=du,
    )
    switches = _get_switches(lambda s: s.target.x1 * s.target.AD + s.target.x2 * s.target.AI, h0, h1)
    interactions = [
        RateInteraction(
            'dependent_growth',
            target='tumour',
            state='AD',
            rate=lambda s: s.target.r1
            * (s.target.k1 + (1 - s.target.k1) * s.target.U / (s.target.U + s.target.k2))
            * s.target.AD,
        ),
        ReactionInteraction(
            'dependent_death',
            target='tumour',
            reactants={'AD': 1},
            propensity=lambda s: s.target.d1
            * (s.target.k3 + (1 - s.target.k3) * s.target.U / (s.target.U + s.target.k4))
            * s.target.AD,
        ),
        ReactionInteraction(
            'mutation',
            target='tumour',
            reactants={'AD': 1},
            products={'AI': 1},
            propensity=lambda s: s.target.m1 * (1 - s.target.U / s.target.U0) * s.target.AD,
        ),
        RateInteraction(
            'independent_growth',
            target='tumour',
            state='AI',
            rate=lambda s: s.target.r2 * (1 - s.target.U / s.target.U0) * s.target.AI,
        ),
        ReactionInteraction(
            'independent_death',
            target='tumour',
            reactants={'AI': 1},
            propensity=lambda s: s.target.d2 * s.target.AI,
        ),
        RateInteraction(
            'androgen_production',
            target='tumour',
            state='U',
            rate=lambda s: -s.target.du * (s.target.U - s.target.U0),
        ),
        RateInteraction(
            'suppression',
            target='tumour',
            state='U',
            rate=lambda s: -s.target.U0 * s.target.du * s.target.u,
        ),
        *switches,
    ]
    return System(elements=[tumour], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'cell quota', 'intermittent'),
    references=(BOOK, 'Portz, Kuang, and Nagy 2012'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def prostate_cell_quota(
    AD: float = 15.0,
    AI: float = 0.0,
    *,
    rm: float = 0.025,
    q1: float = 0.3,
    q2: float = 0.15,
    d1: float = 0.015,
    d2: float = 0.015,
    r0: float = 0.01,
    r1: float = 0.15,
    r2: float = 0.15,
    n: float = 3.0,
    theta: float = 3.0,
    U0: float = 15.0,
    Umin: float = 0.5,
    du1: float = 0.08,
    period: float = 500.0,
    width: float = 250.0,
) -> System:
    """Return the Portz cell-quota model of prostate cancer, whose cells grow on their internal androgen.

    The androgen-dependent and independent cells grow at the Droop rate `rm (1 - q_i / Q_i)` of their cell quotas,
    which take up serum androgen, and the PSA they release follows the quotas. A deprived dependent cell turns
    independent faster, and an independent cell full of androgen turns back. The serum androgen follows the
    exponential relaxation of the book's equation 7.18 between the normal level `U0` and `Umin`, with therapy during
    the first `width` days of every `period`. The book gives neither the steepness `n` of the mutation rates nor
    `theta` of the PSA production, nor the serum levels, their relaxation rate, and the schedule, so these defaults
    are chosen, and the other defaults lie within the book's ranges. Cells are in billions, quotas in nM, PSA in
    ng/mL, and time in days. The PSA starts near its value for the initial quotas.

    Args:
        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        rm:
            The maximum proliferation rate.

        q1:
            The minimum quota of the dependent cells.

        q2:
            The minimum quota of the independent cells.

        d1:
            The death rate of the dependent cells.

        d2:
            The death rate of the independent cells.

        r0:
            The baseline PSA production per cell.

        r1:
            The androgen-driven PSA production per dependent cell.

        r2:
            The androgen-driven PSA production per independent cell.

        n:
            The steepness of the mutation rates.

        theta:
            The steepness of the PSA production.

        U0:
            The normal serum androgen.

        Umin:
            The serum androgen under therapy.

        du1:
            The rate at which the serum androgen relaxes.

        period:
            The time between the starts of therapy periods.

        width:
            The duration of a therapy period.

    Returns:
        system:
            The model, with the cells `tumour.AD` and `tumour.AI`, the quotas `tumour.Q1` and `tumour.Q2`, the PSA
            `tumour.P`, and the androgen `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import prostate_cell_quota
        >>> prostate_cell_quota(q1=0.45).elements[0].q1
        0.45
    """
    tumour = CellQuotaProstateElement(
        'tumour',
        AD=AD,
        AI=AI,
        U=U0,
        rm=rm,
        q1=q1,
        q2=q2,
        d1=d1,
        d2=d2,
        r0=r0,
        r1=r1,
        r2=r2,
        n=n,
        theta=theta,
        U0=U0,
        Umin=Umin,
        du1=du1,
        period=period,
        width=width,
    )
    interactions = [
        RateInteraction(
            'dependent_growth',
            target='tumour',
            state='AD',
            rate=lambda s: s.target.rm * (1 - s.target.q1 / s.target.Q1) * s.target.AD,
        ),
        ReactionInteraction(
            'dependent_death',
            target='tumour',
            reactants={'AD': 1},
            propensity=lambda s: s.target.d1 * s.target.AD,
        ),
        ReactionInteraction(
            'to_independent',
            target='tumour',
            reactants={'AD': 1},
            products={'AI': 1},
            propensity=lambda s: s.target.mx1 * (1 - hill(s.target.Q1, s.target.m501, s.target.n)) * s.target.AD,
        ),
        ReactionInteraction(
            'to_dependent',
            target='tumour',
            reactants={'AI': 1},
            products={'AD': 1},
            propensity=lambda s: s.target.mx2 * hill(s.target.Q2, s.target.m502, s.target.n) * s.target.AI,
        ),
        RateInteraction(
            'independent_growth',
            target='tumour',
            state='AI',
            rate=lambda s: s.target.rm * (1 - s.target.q2 / s.target.Q2) * s.target.AI,
        ),
        ReactionInteraction(
            'independent_death',
            target='tumour',
            reactants={'AI': 1},
            propensity=lambda s: s.target.d2 * s.target.AI,
        ),
        RateInteraction(
            'dependent_uptake',
            target='tumour',
            state='Q1',
            rate=lambda s: _quota_uptake(s.target, s.target.Q1, s.target.q1),
        ),
        RateInteraction(
            'dependent_use',
            target='tumour',
            state='Q1',
            rate=lambda s: -s.target.rm * (s.target.Q1 - s.target.q1),
        ),
        ReactionInteraction(
            'dependent_degradation',
            target='tumour',
            reactants={'Q1': 1},
            propensity=lambda s: s.target.d3 * s.target.Q1,
        ),
        RateInteraction(
            'independent_uptake',
            target='tumour',
            state='Q2',
            rate=lambda s: _quota_uptake(s.target, s.target.Q2, s.target.q2),
        ),
        RateInteraction(
            'independent_use',
            target='tumour',
            state='Q2',
            rate=lambda s: -s.target.rm * (s.target.Q2 - s.target.q2),
        ),
        ReactionInteraction(
            'independent_degradation',
            target='tumour',
            reactants={'Q2': 1},
            propensity=lambda s: s.target.d3 * s.target.Q2,
        ),
        RateInteraction(
            'psa_leak',
            target='tumour',
            state='P',
            rate=lambda s: s.target.r0 * (s.target.AD + s.target.AI),
        ),
        RateInteraction(
            'psa_production',
            target='tumour',
            state='P',
            rate=lambda s: s.target.r1 * s.target.AD * hill(s.target.Q1, s.target.rho1, s.target.theta)
            + s.target.r2 * s.target.AI * hill(s.target.Q2, s.target.rho2, s.target.theta),
        ),
        ReactionInteraction(
            'psa_clearance',
            target='tumour',
            reactants={'P': 1},
            propensity=lambda s: s.target.du * s.target.P,
        ),
        RateInteraction(
            'androgen',
            target='tumour',
            state='U',
            rate=lambda s: s.target.du1
            * (
                s.target.U0 - (s.target.U0 - s.target.Umin) * pulse_train(s.t, tumour.period, tumour.width) - s.target.U
            ),
        ),
    ]
    return System(elements=[tumour], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'androgen receptor', 'PSA'),
    references=(BOOK, 'Jain et al. 2011'),
    defaults={'until': 26280.0, 'dt': 168.0, 'solver': ScipyODESolver(method='LSODA')},
)
def prostate_androgen_receptor(
    N: float = 1.5,
    AD: float = 1.0,
    AI: float = 0.0,
    *,
    UT: float = 40.0,
    UD: float = 40.0,
    m1: float = 1e-4,
    age: float = 60.0,
) -> System:
    """Return the Jain model of prostate epithelium and cancer driven by activated androgen receptors, with PSA.

    The growth and death rates of the normal, androgen-dependent, and androgen-independent cells follow the receptors
    activated by testosterone, `UT`, and dihydrotestosterone, `UD`, which are parameters, so androgen ablation is a
    lower activation. The normal cells fill a capacity that grows with the age-dependent prostate volume, and the PSA
    the cells produce in the tissue leaks into the serum through the normal epithelium and, faster, through the
    tumour. Cells are in billions, receptors in nM, PSA in ng/mL, and time in hours, the age in years. The book
    leaves several parts open, filled as follows. The half saturations `psi2`, `phi2`, and `psi4` of the cancer cells
    are those of the normal cells. The death rate is `psi3_i / (psi4 + UD + UT)`. One activation `UT`, `UD` holds
    for all cell types, and so does `sigma`. The independent cells gain the part `m1` of the divisions of the
    dependent cells. The negligible interaction `c1` is 0. The capacity of the normal cells is the prostate volume
    over the volume of a billion cells, and the PSA leak rate per million cells becomes `0.3125` per billion.

    Args:
        N:
            The initial normal epithelial cells.

        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        UT:
            The concentration of receptors activated by testosterone.

        UD:
            The concentration of receptors activated by dihydrotestosterone.

        m1:
            The fraction of the divisions of dependent cells that yields independent cells.

        age:
            The age of the patient at the start.

    Returns:
        system:
            The model, with the cells `prostate.N`, `prostate.AD`, and `prostate.AI`, and the PSA `prostate.UIP` in
            the tissue and `prostate.UP` in the serum.

    Examples:
        >>> from skcomplex.models.biology.oncology import prostate_androgen_receptor
        >>> prostate_androgen_receptor(UT=5.0).elements[0].UT
        5.0
    """
    prostate = AndrogenReceptorProstateElement('prostate', N=N, AD=AD, AI=AI, UT=UT, UD=UD, m1=m1, age=age)
    interactions = [
        RateInteraction(
            'normal_growth',
            target='prostate',
            state='N',
            rate=lambda s: _receptor_growth(s.target, s.target.psi1_n, s.target.phi1_n)
            * s.target.N
            * (
                1
                - (s.target.N + s.target.c1 * (s.target.AD + s.target.AI))
                / (_prostate_volume(s.target, s.t) / s.target.Vc)
            ),
        ),
        ReactionInteraction(
            'normal_death',
            target='prostate',
            reactants={'N': 1},
            propensity=lambda s: _receptor_death(s.target, s.target.psi3_n) * s.target.N,
        ),
        RateInteraction(
            'dependent_division',
            target='prostate',
            state='AD',
            rate=lambda s: _dependent_division(s.target),
        ),
        ReactionInteraction(
            'mutation',
            target='prostate',
            reactants={'AD': 1},
            products={'AI': 1},
            propensity=lambda s: s.target.m1 * _dependent_division(s.target),
        ),
        ReactionInteraction(
            'dependent_death',
            target='prostate',
            reactants={'AD': 1},
            propensity=lambda s: _receptor_death(s.target, s.target.psi3_d) * s.target.AD,
        ),
        RateInteraction(
            'independent_growth',
            target='prostate',
            state='AI',
            rate=lambda s: _receptor_growth(s.target, s.target.psi1_i, s.target.phi1_i)
            * s.target.AI
            * (1 - (s.target.c1 * s.target.N + s.target.AD + s.target.AI) / s.target.b3),
        ),
        ReactionInteraction(
            'independent_death',
            target='prostate',
            reactants={'AI': 1},
            propensity=lambda s: _receptor_death(s.target, s.target.psi3_i) * s.target.AI,
        ),
        RateInteraction(
            'psa_production',
            target='prostate',
            state='UIP',
            rate=lambda s: (s.target.UT + s.target.UD + s.target.sigma)
            * (s.target.beta_n * s.target.N + s.target.beta_d * s.target.AD + s.target.beta_i * s.target.AI),
        ),
        ReactionInteraction(
            'normal_leak',
            target='prostate',
            reactants={'UIP': 1},
            products={'UP': 1},
            propensity=lambda s: s.target.du1
            * s.target.UIP
            * s.target.N
            / (_prostate_volume(s.target, s.t) / s.target.Vc),
        ),
        ReactionInteraction(
            'cancer_leak',
            target='prostate',
            reactants={'UIP': 1},
            products={'UP': 1},
            propensity=lambda s: s.target.gamma_p
            * s.target.UIP
            * (s.target.AD + s.target.AI) ** 2
            / (s.target.KP + s.target.AD + s.target.AI),
        ),
        ReactionInteraction(
            'intracellular_clearance',
            target='prostate',
            reactants={'UIP': 1},
            propensity=lambda s: s.target.du2 * s.target.UIP,
        ),
        ReactionInteraction(
            'serum_clearance',
            target='prostate',
            reactants={'UP': 1},
            propensity=lambda s: s.target.du3 * s.target.UP,
        ),
    ]
    return System(elements=[prostate], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'piecewise linear', 'events', 'intermittent'),
    references=(BOOK, 'Hirata, Bruchovsky, and Aihara 2010'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def piecewise_linear_ias(patient: int = 1, *, h0: float = 4.0, h1: float = 10.0) -> System:
    """Return the Hirata piecewise-linear model of intermittent androgen suppression for one of three patient types.

    The androgen-dependent, adapted, and mutated cells grow and change into each other linearly, with one set of
    rates during therapy and one off therapy, and the therapy switches on when the PSA, the sum of the cells, rises
    through `h1` and off when it falls through `h0`. The rates are the book's daily factors, `w = omega - 1` on the
    diagonal and `w = omega` off it. Patient 1 does not relapse, patient 2 relapses late, and patient 3 relapses,
    once the mutated cells outgrow the therapy. The book only states that therapy starts above 10 ng/mL, so the
    therapy stops below 4 ng/mL and is on at the start when the PSA exceeds `h1`. The cells are in units of the PSA,
    ng/mL, and time is in days.

    Args:
        patient:
            The patient type, 1, 2, or 3, which sets the rates and the initial cells.

        h0:
            The PSA below which the therapy stops.

        h1:
            The PSA above which the therapy starts.

    Returns:
        system:
            The model, with the cells `tumour.AD`, `tumour.AIR`, and `tumour.AII`, the therapy `tumour.u`, and the
            switches as `on` and `off` events.

    Raises:
        ValidationError:
            The patient type is not 1, 2, or 3.

    Examples:
        >>> from skcomplex.models.biology.oncology import piecewise_linear_ias
        >>> piecewise_linear_ias(patient=3).elements[0].AD
        11.02
    """
    if patient not in PATIENT_FACTORS:
        error_msg = f'The patient type must be 1, 2, or 3, got {patient!r}.'
        raise ValidationError(error_msg)
    factors = PATIENT_FACTORS[patient]
    on11, on21, on22, on31, on32, on33 = factors['on']
    off11, off12, off22, off33 = factors['off']
    AD, AIR, AII = factors['cells']
    tumour = PiecewiseLinearProstateElement(
        'tumour',
        AD=AD,
        AIR=AIR,
        AII=AII,
        u=float(h1 < AD + AIR + AII),
        w11_on=on11 - 1,
        w21_on=on21,
        w22_on=on22 - 1,
        w31_on=on31,
        w32_on=on32,
        w33_on=on33 - 1,
        w11_off=off11 - 1,
        w12_off=off12,
        w22_off=off22 - 1,
        w33_off=off33 - 1,
    )
    switches = _get_switches(
        lambda s: s.target.x1 * s.target.AD + s.target.x2 * s.target.AIR + s.target.x3 * s.target.AII,
        h0,
        h1,
    )
    return System(elements=[tumour], interactions=switches)


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'breast cancer', 'cell cycle', 'pharmacokinetics'),
    references=(BOOK, 'Chen, Kirkby, and Jena 2012'),
    defaults={'until': 2400.0, 'dt': 8.0, 'solver': ScipyODESolver(method='LSODA')},
)
def tamoxifen_cell_cycle(
    Ag: float = 900.0,
    As: float = 50.0,
    Am: float = 50.0,
    *,
    u: float = 12.5,
    a1: float = 0.0062,
    a2: float = 0.010,
) -> System:
    """Return a breast tumour in Gompertzian cell-cycle phases under daily tamoxifen and its active metabolite.

    The cells pass from the growth phase through synthesis to mitosis and divide back into the growth phase, at
    rates that slow as the tumour approaches `Asat`. Tamoxifen is absorbed from the gut into the plasma, where it is
    metabolized into 4-hydroxytamoxifen, and both kill cells in the growth phase, the metabolite 25 times more. They
    also kill circulating lymphocytes, which stay at `s / d1` without the drug. The book gives no infusion rate, so the
    default `u` is 300 μg a day, given continuously. Volumes are in mm³, drug masses in μg, concentrations in
    μg/mL, lymphocytes in millions, and time in hours.

    Args:
        Ag:
            The initial volume of cells in the growth phase.

        As:
            The initial volume of cells in the synthesis phase.

        Am:
            The initial volume of cells in mitosis.

        u:
            The infusion rate of tamoxifen.

        a1:
            The kill rate of the tumour per unit of drug.

        a2:
            The kill rate of the lymphocytes per unit of drug.

    Returns:
        system:
            The model, with the phases `tumour.Ag`, `tumour.As`, and `tumour.Am`, the drug masses `tumour.X0` to
            `tumour.X3`, and the lymphocytes `tumour.E`.

    Examples:
        >>> from skcomplex.models.biology.oncology import tamoxifen_cell_cycle
        >>> tamoxifen_cell_cycle(u=0.0).elements[0].u
        0.0
    """
    tumour = TamoxifenTumourElement('tumour', Ag=Ag, As=As, Am=Am, u=u, a1=a1, a2=a2)
    interactions = [
        ReactionInteraction(
            'g1_exit',
            target='tumour',
            reactants={'Ag': 1},
            products={'As': 1},
            propensity=lambda s: s.target.mg * s.target.Ag * _room(s.target),
        ),
        ReactionInteraction(
            'synthesis',
            target='tumour',
            reactants={'As': 1},
            products={'Am': 1},
            propensity=lambda s: s.target.ms * s.target.As,
        ),
        ReactionInteraction(
            'mitosis',
            target='tumour',
            reactants={'Am': 1},
            products={'Ag': 2},
            propensity=lambda s: s.target.mm * s.target.Am * _room(s.target),
        ),
        ReactionInteraction(
            'drug_kill',
            target='tumour',
            reactants={'Ag': 1},
            propensity=lambda s: s.target.a1
            * s.target.Ag
            * (s.target.X2 / s.target.V + s.target.delta1 * s.target.X3 / s.target.V),
        ),
        DosingInteraction('dosing', target='tumour', state='X0', dose=lambda s: s.target.u),
        ReactionInteraction(
            'absorption',
            target='tumour',
            reactants={'X0': 1},
            products={'X1': 1},
            propensity=lambda s: s.target.k01 * s.target.X0,
        ),
        ReactionInteraction(
            'transport',
            target='tumour',
            reactants={'X1': 1},
            products={'X2': 1},
            propensity=lambda s: s.target.k12 * s.target.X1,
        ),
        ReactionInteraction(
            'metabolism',
            target='tumour',
            reactants={'X2': 1},
            products={'X3': 1},
            propensity=lambda s: s.target.k23 * s.target.X2,
        ),
        ReactionInteraction(
            'tamoxifen_clearance',
            target='tumour',
            reactants={'X2': 1},
            propensity=lambda s: s.target.du1 * s.target.X2,
        ),
        ReactionInteraction(
            'metabolite_clearance',
            target='tumour',
            reactants={'X3': 1},
            propensity=lambda s: s.target.du2 * s.target.X3,
        ),
        RateInteraction('effector_source', target='tumour', state='E', rate=lambda s: s.target.s),
        ReactionInteraction(
            'effector_death',
            target='tumour',
            reactants={'E': 1},
            propensity=lambda s: s.target.d1 * s.target.E,
        ),
        ReactionInteraction(
            'effector_drug_kill',
            target='tumour',
            reactants={'E': 1},
            propensity=lambda s: s.target.a2
            * s.target.E
            * (s.target.X2 / s.target.V + s.target.delta2 * s.target.X3 / s.target.V),
        ),
    ]
    return System(elements=[tumour], interactions=interactions)
