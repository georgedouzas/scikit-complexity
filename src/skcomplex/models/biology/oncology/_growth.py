"""Oncology models: tumour growth laws and the cell kill of chemotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import log

from ....core.base import Domain, parameter, role, state
from ....core.model import Element, Interaction, Law, Rate, Symbols, System, pulse_train, where
from ....core.runtime import register_model

GROWTH_REFERENCES = ('Padmanabhan, Meskin, and Al Moustafa 2021', 'Benzekry et al. 2014')


class Tumour(Element):
    """Tumour of a size measured in cells or in volume.

    Args:
        label:
            The label of the tumour.

        n:
            The initial size.

    Examples:
        >>> from skcomplex.models.biology.oncology import Tumour
        >>> Tumour('tumour', n=50.0).n
        50.0
    """

    n: float = state(default=1.0, domain=Domain.NONNEGATIVE)


class Vasculature(Element):
    """Vasculature of a tumour, which sets its carrying capacity.

    Args:
        label:
            The label of the vasculature.

        K:
            The initial carrying capacity.

    Examples:
        >>> from skcomplex.models.biology.oncology import Vasculature
        >>> Vasculature('vessels', K=12.7).K
        12.7
    """

    K: float = state(default=1.0, domain=Domain.POSITIVE)


class ExponentialLinearGrowth(Interaction):
    """Growth that is exponential, `dn/dt = r0 n`, until it reaches the linear rate `r`, and linear after.

    The switch happens at `tau = log(r / (r0 n0)) / r0`, where the two rates meet, so the solution is continuously
    differentiable.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        r0:
            The exponential growth rate.

        r:
            The linear growth rate.

    Examples:
        >>> from skcomplex.models.biology.oncology import ExponentialLinearGrowth
        >>> ExponentialLinearGrowth('growth', target='tumour', r=67.8).r
        67.8
    """

    target: str = role()
    r0: float = parameter(default=0.31, domain=Domain.POSITIVE)
    r: float = parameter(default=67.8, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `min(r0 n, r)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        exponential = s.r0 * s.target.n
        return [Rate(s.target.n, where(exponential < s.r, exponential, s.r))]


class PowerLawGrowth(Interaction):
    """Growth of a proliferating part that scales as a power of the size, `dn/dt = r n^p`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        r:
            The growth rate.

        p:
            The exponent, `2/3` for a proliferating rim and 1 for exponential growth.

    Examples:
        >>> from skcomplex.models.biology.oncology import PowerLawGrowth
        >>> PowerLawGrowth('growth', target='tumour').p
        0.58
    """

    target: str = role()
    r: float = parameter(default=1.32, domain=Domain.POSITIVE)
    p: float = parameter(default=0.58, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `r n^p`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, s.r * s.target.n**s.p)]


class ExponentialGrowth(Interaction):
    """Growth at a constant per capita rate, `dn/dt = r n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        r:
            The per capita growth rate.

    Examples:
        >>> from skcomplex.models.biology.oncology import ExponentialGrowth
        >>> ExponentialGrowth('growth', target='tumour', r=0.223).r
        0.223
    """

    target: str = role()
    r: float = parameter(default=0.223, domain=Domain.REAL)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `r n`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, s.r * s.target.n)]


class Loss(Interaction):
    """Loss at a constant per capita rate, `dn/dt = -d n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        d:
            The per capita loss rate.

    Examples:
        >>> from skcomplex.models.biology.oncology import Loss
        >>> Loss('loss', target='tumour', d=0.8).d
        0.8
    """

    target: str = role()
    d: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `-d n`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, -s.d * s.target.n)]


class GeneralizedLogisticGrowth(Interaction):
    """Growth that saturates at a carrying capacity, `dn/dt = r n (1 - (n / K)^p)`, logistic for `p = 1`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        r:
            The growth rate at low density.

        K:
            The carrying capacity.

        p:
            The exponent of the saturation.

    Examples:
        >>> from skcomplex.models.biology.oncology import GeneralizedLogisticGrowth
        >>> GeneralizedLogisticGrowth('growth', target='tumour', p=2.0).p
        2.0
    """

    target: str = role()
    r: float = parameter(default=0.305, domain=Domain.POSITIVE)
    K: float = parameter(default=1221.0, domain=Domain.POSITIVE)
    p: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `r n (1 - (n / K)^p)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, s.r * s.target.n * (1 - (s.target.n / s.K) ** s.p))]


class GompertzGrowth(Interaction):
    """Growth that slows as the logarithm of the room left, `dn/dt = r n log(K / n)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        r:
            The rate at which the growth slows.

        K:
            The carrying capacity.

    Examples:
        >>> from skcomplex.models.biology.oncology import GompertzGrowth
        >>> GompertzGrowth('growth', target='tumour', K=2000.0).K
        2000.0
    """

    target: str = role()
    r: float = parameter(default=0.0719, domain=Domain.POSITIVE)
    K: float = parameter(default=1964.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `r n log(K / n)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, s.r * s.target.n * log(s.K / s.target.n))]


class VascularSupport(Interaction):
    """Gompertz growth under a carrying capacity that the vessels on the tumour surface raise.

    The tumour follows `dn/dt = r n log(K / n)` and the capacity `dK/dt = d n^(2/3)`.

    Args:
        label:
            The label of the interaction.

        tumour:
            The path of the tumour.

        vasculature:
            The path of the vasculature.

        r:
            The rate at which the growth slows.

        d:
            The rate at which the surface of the tumour raises the capacity.

    Examples:
        >>> from skcomplex.models.biology.oncology import VascularSupport
        >>> VascularSupport('support', tumour='tumour', vasculature='vessels').d
        0.829
    """

    tumour: str = role()
    vasculature: str = role()
    r: float = parameter(default=2.63, domain=Domain.POSITIVE)
    d: float = parameter(default=0.829, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the Gompertz growth of the tumour and the growth of the capacity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n, K = s.tumour.n, s.vasculature.K
        return [Rate(n, s.r * n * log(K / n)), Rate(K, s.d * n ** (2 / 3))]


class LogKill(Interaction):
    """Cell kill proportional to the tumour and to a drug exposure given in a train of rectangular doses.

    The kill is `dn/dt = -a n U(t)`, where the exposure `U` is `dose` for `width` time units every `period`, from
    `start` on. Without growth, every dose lowers `log n` by `a dose width`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tumour.

        a:
            The kill rate per unit of exposure.

        dose:
            The exposure during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Examples:
        >>> from skcomplex.models.biology.oncology import LogKill
        >>> LogKill('kill', target='tumour', period=21.0).period
        21.0
    """

    target: str = role()
    a: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    dose: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    period: float = 7.0
    width: float = 1.0
    start: float = 0.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `-a n U(t)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        exposure = pulse_train(s.t, self.period, self.width, start=self.start)
        return [Rate(s.target.n, -s.a * s.dose * exposure * s.target.n)]


@register_model(
    field='biology/oncology',
    tags=('tumour growth', 'ode', 'piecewise'),
    references=GROWTH_REFERENCES,
    defaults={'until': 40.0, 'dt': 0.2},
)
def exponential_linear_growth(n: float = 2.0, *, r0: float = 0.31, r: float = 67.8) -> System:
    """Return a tumour that grows exponentially while small and linearly once its growth reaches a limiting rate.

    The defaults are the fit to breast cancer volumes in mm³ and days, which switches near day 15.

    Args:
        n:
            The initial volume.

        r0:
            The exponential growth rate.

        r:
            The linear growth rate.

    Returns:
        system:
            The model, with the volume `tumour.n`.

    Examples:
        >>> from skcomplex.models.biology.oncology import exponential_linear_growth
        >>> exponential_linear_growth().interactions[0].r0
        0.31
    """
    return System(
        components=[Tumour('tumour', n=n)],
        interactions=[ExponentialLinearGrowth('growth', target='tumour', r0=r0, r=r)],
    )


@register_model(
    field='biology/oncology',
    tags=('tumour growth', 'ode', 'power law'),
    references=GROWTH_REFERENCES,
    defaults={'until': 40.0, 'dt': 0.2},
)
def power_law_growth(n: float = 10.0, *, r: float = 1.32, p: float = 0.58) -> System:
    """Return a tumour whose proliferating part scales as a power of its size, `dn/dt = r n^p`.

    For `p < 1` the volume follows `(n0^(1-p) + (1-p) r t)^(1/(1-p))`, a growth slower than exponential. The defaults
    are the fit to breast cancer volumes in mm³ and days.

    Args:
        n:
            The initial volume.

        r:
            The growth rate.

        p:
            The exponent.

    Returns:
        system:
            The model, with the volume `tumour.n`.

    Examples:
        >>> from skcomplex.models.biology.oncology import power_law_growth
        >>> power_law_growth(p=2 / 3).interactions[0].p
        0.6666666666666666
    """
    return System(
        components=[Tumour('tumour', n=n)],
        interactions=[PowerLawGrowth('growth', target='tumour', r=r, p=p)],
    )


@register_model(
    field='biology/oncology',
    tags=('tumour growth', 'ode', 'allometry'),
    references=(*GROWTH_REFERENCES, 'von Bertalanffy 1957'),
    defaults={'until': 80.0, 'dt': 0.4},
)
def von_bertalanffy(n: float = 1.0, *, r: float = 2.32, p: float = 0.918, d: float = 0.808) -> System:
    """Return a tumour that grows by the balance of anabolism and catabolism, `dn/dt = r n^p - d n`.

    The volume tends to `(r / d)^(1/(1-p))`. The defaults are the fit to breast cancer volumes in mm³ and days.

    Args:
        n:
            The initial volume.

        r:
            The growth rate.

        p:
            The exponent of the growth, below 1.

        d:
            The loss rate.

    Returns:
        system:
            The model, with the volume `tumour.n`.

    Examples:
        >>> from skcomplex.models.biology.oncology import von_bertalanffy
        >>> von_bertalanffy().interactions[1].d
        0.808
    """
    return System(
        components=[Tumour('tumour', n=n)],
        interactions=[PowerLawGrowth('growth', target='tumour', r=r, p=p), Loss('loss', target='tumour', d=d)],
    )


@register_model(
    field='biology/oncology',
    tags=('tumour growth', 'ode', 'logistic', 'Richards'),
    references=(*GROWTH_REFERENCES, 'Richards 1959'),
    defaults={'until': 40.0, 'dt': 0.2},
)
def generalized_logistic(n: float = 10.0, *, r: float = 0.305, K: float = 1221.0, p: float = 1.0) -> System:
    """Return a tumour that saturates at a carrying capacity, `dn/dt = r n (1 - (n / K)^p)`, logistic for `p = 1`.

    The volume follows `n0 K / (n0^p + (K^p - n0^p) exp(-r p t))^(1/p)`. The defaults are the logistic fit to breast
    cancer volumes in mm³ and days.

    Args:
        n:
            The initial volume.

        r:
            The growth rate at low density.

        K:
            The carrying capacity.

        p:
            The exponent of the saturation.

    Returns:
        system:
            The model, with the volume `tumour.n`.

    Examples:
        >>> from skcomplex.models.biology.oncology import generalized_logistic
        >>> generalized_logistic(p=0.5).interactions[0].p
        0.5
    """
    return System(
        components=[Tumour('tumour', n=n)],
        interactions=[GeneralizedLogisticGrowth('growth', target='tumour', r=r, K=K, p=p)],
    )


@register_model(
    field='biology/oncology',
    tags=('tumour growth', 'ode', 'sigmoid'),
    references=(*GROWTH_REFERENCES, 'Gompertz 1825'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def gompertz(n: float = 10.0, *, r: float = 0.0719, K: float = 1964.0) -> System:
    """Return a tumour whose growth slows as the logarithm of the room left, `dn/dt = r n log(K / n)`.

    The volume follows `K exp(log(n0 / K) exp(-r t))` and tends to `K`. The rate is the fitted decay of the
    proliferation of breast cancer, per day, and the capacity is in mm³.

    Args:
        n:
            The initial volume.

        r:
            The rate at which the growth slows.

        K:
            The carrying capacity.

    Returns:
        system:
            The model, with the volume `tumour.n`.

    Examples:
        >>> from skcomplex.models.biology.oncology import gompertz
        >>> gompertz(K=1000.0).interactions[0].K
        1000.0
    """
    return System(
        components=[Tumour('tumour', n=n)],
        interactions=[GompertzGrowth('growth', target='tumour', r=r, K=K)],
    )


@register_model(
    field='biology/oncology',
    tags=('tumour growth', 'ode', 'angiogenesis', 'sigmoid'),
    references=(*GROWTH_REFERENCES, 'Hahnfeldt et al. 1999'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def dynamic_carrying_capacity(n: float = 1.0, K: float = 12.7, *, r: float = 2.63, d: float = 0.829) -> System:
    """Return Gompertz growth under a carrying capacity that the vessels on the tumour surface keep raising.

    The tumour follows `dn/dt = r n log(K / n)` and the capacity `dK/dt = d n^(2/3)`, so the tumour first fills its
    initial capacity and then grows as fast as its vasculature, roughly as `t^3`. The defaults are the fit to breast
    cancer volumes in mm³ and days.

    Args:
        n:
            The initial volume.

        K:
            The initial carrying capacity.

        r:
            The rate at which the growth slows.

        d:
            The rate at which the surface of the tumour raises the capacity.

    Returns:
        system:
            The model, with the volume `tumour.n` and the capacity `vessels.K`.

    Examples:
        >>> from skcomplex.models.biology.oncology import dynamic_carrying_capacity
        >>> dynamic_carrying_capacity().elements[1].K
        12.7
    """
    return System(
        components=[Tumour('tumour', n=n), Vasculature('vessels', K=K)],
        interactions=[VascularSupport('support', tumour='tumour', vasculature='vessels', r=r, d=d)],
    )


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'log kill', 'dosing'),
    references=('Padmanabhan, Meskin, and Al Moustafa 2021', 'Skipper, Schabel, and Wilcox 1964'),
    defaults={'until': 42.0, 'dt': 0.1},
)
def skipper_log_kill(
    n: float = 1e6,
    *,
    r: float = 0.223,
    a: float = 2.0,
    dose: float = 1.0,
    period: float = 7.0,
    width: float = 1.0,
) -> System:
    """Return the log-kill model, an exponentially growing tumour of which every dose kills a fixed fraction.

    The tumour follows `dn/dt = r n - a n U(t)`, where the exposure `U` is `dose` for `width` days every `period`
    days. Every dose lowers `log n` by `a dose width`, whatever the size, and the tumour regrows between doses. The
    growth rate is the exponential fit to breast cancer, per day.

    Args:
        n:
            The initial number of cells.

        r:
            The growth rate.

        a:
            The kill rate per unit of exposure.

        dose:
            The exposure during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

    Returns:
        system:
            The model, with the cells `tumour.n`.

    Examples:
        >>> from skcomplex.models.biology.oncology import skipper_log_kill
        >>> skipper_log_kill(period=21.0).interactions[1].period
        21.0
    """
    return System(
        components=[Tumour('tumour', n=n)],
        interactions=[
            ExponentialGrowth('growth', target='tumour', r=r),
            LogKill('kill', target='tumour', a=a, dose=dose, period=period, width=width),
        ],
    )


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'log kill', 'Gompertz', 'dosing'),
    references=('Padmanabhan, Meskin, and Al Moustafa 2021', 'Norton and Simon 1977'),
    defaults={'until': 150.0, 'dt': 0.5},
)
def norton_simon(
    n: float = 100.0,
    *,
    r: float = 0.0719,
    K: float = 1964.0,
    a: float = 1.0,
    dose: float = 1.0,
    period: float = 21.0,
    width: float = 1.0,
    start: float = 42.0,
) -> System:
    """Return the Norton-Simon model, a Gompertzian tumour killed in proportion to its size by a dose schedule.

    The tumour follows `dn/dt = r n log(K / n) - a n U(t)`. A Gompertzian tumour regrows faster the smaller it is, so
    a dose that shrinks it is followed by fast regrowth, and the schedule settles on a cycle below the capacity.

    Args:
        n:
            The initial volume.

        r:
            The rate at which the growth slows.

        K:
            The carrying capacity.

        a:
            The kill rate per unit of exposure.

        dose:
            The exposure during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Returns:
        system:
            The model, with the volume `tumour.n`.

    Examples:
        >>> from skcomplex.models.biology.oncology import norton_simon
        >>> norton_simon().interactions[1].start
        42.0
    """
    return System(
        components=[Tumour('tumour', n=n)],
        interactions=[
            GompertzGrowth('growth', target='tumour', r=r, K=K),
            LogKill('kill', target='tumour', a=a, dose=dose, period=period, width=width, start=start),
        ],
    )
