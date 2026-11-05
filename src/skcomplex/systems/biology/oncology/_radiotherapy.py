"""Oncology systems: tumours under radiotherapy and radiosensitizers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import exp

from ....core.components.interactions import RateInteraction, ReactionInteraction
from ....core.components.laws.utils import pulse_train
from ....core.components.system import System
from ..._catalog import register_system
from ._irradiated_tumour import IrradiatedTumourElement
from ._irradiation import IrradiationInteraction
from ._radiation_competition import RadiationCompetitionElement

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


@register_system(
    field='biology/oncology',
    tags=('radiotherapy', 'ode', 'competition', 'dosing'),
    references=(BOOK, 'Belostotski and Freedman 2005', 'Liu and Yang 2014'),
    defaults={'until': 1400.0, 'dt': 5.0},
)
def radiotherapy_competition(
    N: float = 0.5,
    A: float = 0.8,
    *,
    r1: float = 0.008,
    r2: float = 0.4,
    c1: float = 0.01,
    c2: float = 0.15,
    a2: float = 0.2,
    epsilon: float = 0.001,
    period: float = 7.0,
    width: float = 5.0,
) -> System:
    """Return normal and tumour cells in Lotka-Volterra competition, the tumour irradiated periodically.

    The tumour cells outcompete the normal cells without treatment. The irradiation kills tumour cells at the per
    capita rate `a2` for `width` days every `period` days, which takes the tumour below the level at which the normal
    cells can invade, so the two coexist, and the radiation that spills over removes normal cells at the rate
    `epsilon`. The book combines a periodic constant dose with the dose proportional to the tumour. A constant dose
    drives an eradicated tumour negative, so the periodic dose here is proportional to the tumour. The book gives no
    schedule, so the defaults irradiate five days a week. The concentrations are normalized by the carrying
    capacities, `b1 = b2 = 1`, and time is in days.

    Args:
        N:
            The initial concentration of normal cells.

        A:
            The initial concentration of tumour cells.

        r1:
            The growth rate of the normal cells.

        r2:
            The growth rate of the tumour cells.

        c1:
            The rate at which the tumour cells suppress the normal cells.

        c2:
            The rate at which the normal cells suppress the tumour cells.

        a2:
            The per capita kill rate of the radiation during an irradiation, between 0.01 and 0.2.

        epsilon:
            The per capita kill rate of the radiation that spills over to the normal cells.

        period:
            The time between the starts of irradiations.

        width:
            The duration of an irradiation.

    Returns:
        system:
            The system, with the normal cells `tissue.N` and the tumour cells `tissue.A`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import radiotherapy_competition
        >>> radiotherapy_competition(a2=0.1).elements[0].a2
        0.1
    """
    tissue = RadiationCompetitionElement(
        'tissue',
        N=N,
        A=A,
        r1=r1,
        r2=r2,
        c1=c1,
        c2=c2,
        a2=a2,
        epsilon=epsilon,
        period=period,
        width=width,
    )
    interactions = [
        RateInteraction(
            'normal_growth',
            target='tissue',
            state='N',
            rate=lambda s: s.target.r1 * s.target.N * (1 - s.target.b1 * s.target.N),
        ),
        ReactionInteraction(
            'normal_competition',
            target='tissue',
            reactants={'N': 1},
            modifiers={'A': 1},
            propensity=lambda s: s.target.c1 * s.target.N * s.target.A,
        ),
        ReactionInteraction(
            'spillover',
            target='tissue',
            reactants={'N': 1},
            propensity=lambda s: s.target.epsilon * s.target.N,
        ),
        RateInteraction(
            'tumour_growth',
            target='tissue',
            state='A',
            rate=lambda s: s.target.r2 * s.target.A * (1 - s.target.b2 * s.target.A),
        ),
        ReactionInteraction(
            'tumour_competition',
            target='tissue',
            reactants={'A': 1},
            modifiers={'N': 1},
            propensity=lambda s: s.target.c2 * s.target.N * s.target.A,
        ),
        ReactionInteraction(
            'irradiation',
            target='tissue',
            reactants={'A': 1},
            propensity=lambda s: s.target.a2 * pulse_train(s.t, period, width) * s.target.A,
        ),
    ]
    return System(elements=[tissue], interactions=interactions)


@register_system(
    field='biology/oncology',
    tags=('radiotherapy', 'ode', 'linear-quadratic', 'events', 'dosing'),
    references=(BOOK, 'Cardilin et al. 2019'),
    defaults={'until': 120.0, 'dt': 0.5},
)
def lq_radiosensitizer(
    A1: float = 27.0,
    Us: float = 2.0,
    *,
    r: float = 0.04,
    m1: float = 0.26,
    a_gamma: float = 0.004,
    alpha: float = 0.054,
    beta: float = 0.0054,
    ap: float = 0.42,
    au: float = 0.15,
    du: float = 5.544,
    dose: float = 2.0,
    period: float = 1.0,
    start: float = 0.125,
    fractions: int = 10,
) -> System:
    """Return a xenograft tumour irradiated in fractions scored by the linear-quadratic system, with a radiosensitizer.

    Every fraction kills the part `1 - exp(-(1 + ap Us)(alpha d + beta d^2))` of the proliferating cells at once,
    which divide once more and die through a chain of damaged compartments, and the accumulated dose slows the growth
    of the survivors as `exp(-a_gamma u_tot)`. The radiosensitizer, given once at time 0, raises both effects while it
    lasts. Volumes are in mm³, doses in Gy, concentrations in μg/mL, and time in days, so the book's `alpha = 54` and
    `a_gamma = 4` per kGy are 0.054 and 0.004 per Gy, `beta = alpha / 10` per Gy², and the elimination rate of 0.231
    per hour is 5.544 per day. Fractions are instantaneous, so the dose-protraction factor is 1. The book's equations
    6.10 and 6.13 also move `m1 A1` from the proliferating cells to the dying chain, which would make the untreated
    tumour shrink at `r - m1 < 0`, so the chain is fed by the irradiated compartments alone and the tumour starts
    with no damaged cells. The book derives `Us(0)` from a dose without its bioavailability and volume, so the
    initial concentration is given directly.

    Args:
        A1:
            The initial tumour volume, all proliferating.

        Us:
            The initial plasma concentration of the radiosensitizer.

        r:
            The growth rate.

        m1:
            The transit rate of the damaged cells.

        a_gamma:
            The inhibition of proliferation per unit of accumulated dose.

        alpha:
            The linear coefficient of the linear-quadratic system.

        beta:
            The quadratic coefficient of the linear-quadratic system.

        ap:
            The potency of the radiosensitizer in raising the cell kill.

        au:
            The potency of the radiosensitizer in raising the accumulated dose.

        du:
            The elimination rate of the radiosensitizer.

        dose:
            The dose of a fraction.

        period:
            The time between fractions.

        start:
            The time of the first fraction.

        fractions:
            The number of fractions.

    Returns:
        system:
            The system, with the compartments `tumour.A1` to `tumour.A6`, the radiosensitizer `tumour.Us`, the
            accumulated dose `tumour.u_tot`, and the fractions as `fraction` events.

    Examples:
        >>> from skcomplex.systems.biology.oncology import lq_radiosensitizer
        >>> lq_radiosensitizer(Us=0.0).elements[0].Us
        0.0
    """
    tumour = IrradiatedTumourElement(
        'tumour',
        A1=A1,
        Us=Us,
        r=r,
        m1=m1,
        a_gamma=a_gamma,
        du=du,
    )
    irradiation = IrradiationInteraction(
        'fraction',
        target='tumour',
        alpha=alpha,
        beta=beta,
        ap=ap,
        au=au,
        dose=dose,
        period=period,
        start=start,
        fractions=fractions,
    )
    interactions = [
        RateInteraction(
            'growth',
            target='tumour',
            state='A1',
            rate=lambda s: s.target.r * s.target.A1 * exp(-s.target.a_gamma * s.target.u_tot),
        ),
        ReactionInteraction(
            'damaged_division',
            target='tumour',
            reactants={'A5': 1},
            products={'A6': 2},
            propensity=lambda s: s.target.r * s.target.A5,
        ),
        ReactionInteraction(
            'arrest',
            target='tumour',
            reactants={'A5': 1},
            products={'A2': 1},
            propensity=lambda s: s.target.m1 * s.target.A5,
        ),
        ReactionInteraction(
            'daughter_arrest',
            target='tumour',
            reactants={'A6': 1},
            products={'A2': 1},
            propensity=lambda s: s.target.m1 * s.target.A6,
        ),
        ReactionInteraction(
            'first_dying_transit',
            target='tumour',
            reactants={'A2': 1},
            products={'A3': 1},
            propensity=lambda s: s.target.m1 * s.target.A2,
        ),
        ReactionInteraction(
            'second_dying_transit',
            target='tumour',
            reactants={'A3': 1},
            products={'A4': 1},
            propensity=lambda s: s.target.m1 * s.target.A3,
        ),
        ReactionInteraction(
            'removal',
            target='tumour',
            reactants={'A4': 1},
            propensity=lambda s: s.target.m1 * s.target.A4,
        ),
        ReactionInteraction(
            'sensitizer_elimination',
            target='tumour',
            reactants={'Us': 1},
            propensity=lambda s: s.target.du * s.target.Us,
        ),
        irradiation,
    ]
    return System(elements=[tumour], interactions=interactions)
