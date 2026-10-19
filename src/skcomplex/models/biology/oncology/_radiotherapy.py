"""Oncology models: tumours under radiotherapy and radiosensitizers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, parameter, state
from ....core.model import Element, Event, Law, Rate, Symbols, System, pulse_train, where
from ....core.runtime import register_model

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class RadiationCompetition(Element):
    """Normal and tumour cells in Lotka-Volterra competition under periodic irradiation.

    The normal cells follow `dN/dt = r1 N (1 - b1 N) - c1 N A - epsilon N` and the tumour cells
    `dA/dt = r2 A (1 - b2 A) - c2 N A - a2 D(t) A`, where the irradiation `D` is 1 for `width` time units every `period`
    and 0 otherwise, and `epsilon N` is the radiation that spills over to the normal tissue.

    Args:
        label:
            The label of the tissue.

        N:
            The initial concentration of normal cells.

        A:
            The initial concentration of tumour cells.

        r1:
            The growth rate of the normal cells.

        r2:
            The growth rate of the tumour cells.

        b1:
            The reciprocal carrying capacity of the normal cells.

        b2:
            The reciprocal carrying capacity of the tumour cells.

        c1:
            The rate at which the tumour cells suppress the normal cells.

        c2:
            The rate at which the normal cells suppress the tumour cells.

        a2:
            The per capita rate at which the radiation kills tumour cells during an irradiation.

        epsilon:
            The per capita rate at which the spilled radiation kills normal cells.

        period:
            The time between the starts of irradiations.

        width:
            The duration of an irradiation.

    Examples:
        >>> from skcomplex.models.biology.oncology import RadiationCompetition
        >>> RadiationCompetition('tissue', N=0.5, A=0.8).A
        0.8
    """

    N: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    A: float = state(default=0.8, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.008, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.4, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=1.0, domain=Domain.POSITIVE)
    b2: float = parameter(default=1.0, domain=Domain.POSITIVE)
    c1: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=0.15, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    epsilon: float = parameter(default=0.001, domain=Domain.NONNEGATIVE)
    period: float = 7.0
    width: float = 5.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the competition of the normal and tumour cells and the kill of the radiation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        irradiation = pulse_train(s.t, self.period, self.width)
        return [
            Rate(s.N, s.r1 * s.N * (1 - s.b1 * s.N) - s.c1 * s.N * s.A - s.epsilon * s.N),
            Rate(s.A, s.r2 * s.A * (1 - s.b2 * s.A) - s.c2 * s.N * s.A - s.a2 * irradiation * s.A),
        ]


class IrradiatedTumour(Element):
    """Tumour in six compartments under linear-quadratic radiation fractions and a radiosensitizer.

    The proliferating cells follow `dA1/dt = r A1 exp(-a_gamma u_tot)`. A fraction of dose `d` at time `t_i` moves the
    lethally irradiated part `M = 1 - exp(-(1 + ap Us)(alpha d + beta d^2))` of `A1` to `A5` and raises the accumulated
    dose `u_tot` by `(1 + au Us) d`. The cells of `A5` divide once into `A6`, `dA5/dt = -(r + m1) A5` and
    `dA6/dt = 2 r A5 - m1 A6`, and both feed the dying chain `A2 -> A3 -> A4`, which leaves the tumour at the rate
    `m1`. The radiosensitizer is eliminated as `dUs/dt = -du Us`.

    Args:
        label:
            The label of the tumour.

        A1:
            The initial volume of proliferating cells.

        A2:
            The initial volume of the first dying compartment.

        A3:
            The initial volume of the second dying compartment.

        A4:
            The initial volume of the third dying compartment.

        A5:
            The initial volume of lethally irradiated cells that divide once more.

        A6:
            The initial volume of lethally irradiated cells that no longer divide.

        Us:
            The initial plasma concentration of the radiosensitizer.

        u_tot:
            The initial accumulated dose.

        k:
            The number of fractions delivered.

        r:
            The growth rate.

        m1:
            The transit rate of the damaged cells.

        a_gamma:
            The inhibition of proliferation per unit of accumulated dose.

        alpha:
            The linear coefficient of the linear-quadratic model.

        beta:
            The quadratic coefficient of the linear-quadratic model.

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

    Examples:
        >>> from skcomplex.models.biology.oncology import IrradiatedTumour
        >>> IrradiatedTumour('tumour', fractions=10).fractions
        10
    """

    A1: float = state(default=27.0, domain=Domain.NONNEGATIVE)
    A2: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    A3: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    A4: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    A5: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    A6: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Us: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    u_tot: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    k: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r: float = parameter(default=0.04, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=0.26, domain=Domain.NONNEGATIVE)
    a_gamma: float = parameter(default=0.004, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.054, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.0054, domain=Domain.NONNEGATIVE)
    ap: float = parameter(default=0.42, domain=Domain.NONNEGATIVE)
    au: float = parameter(default=0.15, domain=Domain.NONNEGATIVE)
    du: float = parameter(default=5.544, domain=Domain.NONNEGATIVE)
    dose: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    period: float = 1.0
    start: float = 0.125
    fractions: int = 10

    def laws(self, s: Symbols) -> list[Law]:
        """Return the growth and transit of the compartments and the fractions as events.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        damage = (1 + s.ap * s.Us) * (s.alpha * s.dose + s.beta * s.dose**2)
        lethal = 1 - exp(-damage)
        guard = where(s.k < self.fractions, s.t - self.start - s.k * self.period, -1)
        reset = {
            s.A1: s.A1 * exp(-damage),
            s.A5: s.A5 + lethal * s.A1,
            s.u_tot: s.u_tot + (1 + s.au * s.Us) * s.dose,
            s.k: s.k + 1,
        }
        return [
            Rate(s.A1, s.r * s.A1 * exp(-s.a_gamma * s.u_tot)),
            Rate(s.A2, s.m1 * (s.A5 + s.A6) - s.m1 * s.A2),
            Rate(s.A3, s.m1 * (s.A2 - s.A3)),
            Rate(s.A4, s.m1 * (s.A3 - s.A4)),
            Rate(s.A5, -(s.r + s.m1) * s.A5),
            Rate(s.A6, 2 * s.r * s.A5 - s.m1 * s.A6),
            Rate(s.Us, -s.du * s.Us),
            Event(guard, reset, direction=1, label='fraction'),
        ]


@register_model(
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
            The model, with the normal cells `tissue.N` and the tumour cells `tissue.A`.

    Examples:
        >>> from skcomplex.models.biology.oncology import radiotherapy_competition
        >>> radiotherapy_competition(a2=0.1).elements[0].a2
        0.1
    """
    tissue = RadiationCompetition(
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
    return System(elements=[tissue])


@register_model(
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
    """Return a xenograft tumour irradiated in fractions that the linear-quadratic model scores, with a radiosensitizer.

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
            The linear coefficient of the linear-quadratic model.

        beta:
            The quadratic coefficient of the linear-quadratic model.

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
            The model, with the compartments `tumour.A1` to `tumour.A6`, the radiosensitizer `tumour.Us`, the
            accumulated dose `tumour.u_tot`, and the fractions as `fraction` events.

    Examples:
        >>> from skcomplex.models.biology.oncology import lq_radiosensitizer
        >>> lq_radiosensitizer(Us=0.0).elements[0].Us
        0.0
    """
    tumour = IrradiatedTumour(
        'tumour',
        A1=A1,
        Us=Us,
        r=r,
        m1=m1,
        a_gamma=a_gamma,
        alpha=alpha,
        beta=beta,
        ap=ap,
        au=au,
        du=du,
        dose=dose,
        period=period,
        start=start,
        fractions=fractions,
    )
    return System(elements=[tumour])
