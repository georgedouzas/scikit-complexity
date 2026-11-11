"""Oncology systems: tumour growth laws and the cell kill of chemotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, log, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import Interaction, RateInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import minimum, pulse_train
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.populations import (
    ExponentialGrowthInteraction,
    GrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)

GROWTH_REFERENCES = ('Padmanabhan, Meskin, and Al Moustafa 2021', 'Benzekry et al. 2014')


class VasculatureElement(Element):
    """Vasculature of a tumour.

    It sets the carrying capacity of the tumour.

    Args:
        label:
            The label of the vasculature.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        K:
            The initial carrying capacity.

    Examples:
        >>> from skcomplex.systems.biology.oncology import VasculatureElement
        >>> VasculatureElement('vessels', K=12.7).K
        12.7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        K: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.K = K


class ExponentialLinearGrowthInteraction(RateInteraction):
    """Growth that is exponential until it reaches a linear rate, and linear after.

    The exponential growth is `dx/dt = r0 x`, and the linear rate is `r`.

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
        >>> from skcomplex.systems.biology.oncology import ExponentialLinearGrowthInteraction
        >>> ExponentialLinearGrowthInteraction('growth', target='tumour', r=67.8).r
        67.8
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.31,
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 67.8,
    ) -> None:
        super().__init__(label, target=target, state='x')
        self.r0 = r0
        self.r = r

    def _get_rate(self, s: SimpleNamespace) -> Expression:
        """Return the rate `min(r0 x, r)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            rate:
                The rate of the size of the tumour.
        """
        exponential = s.r0 * s.target.x
        return minimum(exponential, s.r)


class LogKillInteraction(RateInteraction):
    """Cell kill proportional to the tumour and to a drug exposure.

    The exposure is given in a train of rectangular doses.

    The kill is `dx/dt = -a x U(t)`, where the exposure `U` is `dose` for `width` time units every `period`, from
    `start` on. Without growth, every dose lowers `log x` by `a dose width`.

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
        >>> from skcomplex.systems.biology.oncology import LogKillInteraction
        >>> LogKillInteraction('kill', target='tumour', period=21.0).period
        21.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        period: float = 7.0,
        width: float = 1.0,
        start: float = 0.0,
    ) -> None:
        super().__init__(label, target=target, state='x')
        self.a = a
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start

    def _get_rate(self, s: SimpleNamespace) -> Expression:
        """Return the rate `-a x U(t)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            rate:
                The rate of the size of the tumour.
        """
        exposure = pulse_train(s.t, self.period, self.width, start=self.start)
        return -s.a * s.dose * exposure * s.target.x


class VascularSupportInteraction(Interaction):
    """Gompertz growth under a carrying capacity that grows with the tumour.

    The vessels on the tumour surface raise the capacity.

    The tumour follows `dx/dt = r x log(K / x)` and the capacity `dK/dt = d x^(2/3)`.

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
        >>> from skcomplex.systems.biology.oncology import VascularSupportInteraction
        >>> VascularSupportInteraction('support', tumour='tumour', vasculature='vessels').d
        0.829
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        tumour: Annotated[str, RoleSpec()],
        vasculature: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.63,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.829,
    ) -> None:
        super().__init__(label)
        self.tumour = tumour
        self.vasculature = vasculature
        self.r = r
        self.d = d

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Gompertz growth of the tumour and the growth of the capacity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n, K = s.tumour.x, s.vasculature.K
        return [RateLaw(n, s.r * n * log(K / n)), RateLaw(K, s.d * n ** (2 / 3))]


class GompertzGrowthInteraction(GrowthInteraction):
    """Gompertz growth towards a capacity, without harvest.

    The growth is `dx/dt = r x log(K / x(t - tau))`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose population grows.

        r:
            The growth rate.

        K:
            The capacity.

        tau:
            The delay of the limiting factor, 0 for none.

        state:
            The name of the state of the target that holds the population.

    Examples:
        >>> from skcomplex.systems.biology.common.kinetics import CompartmentElement
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.biology.oncology import GompertzGrowthInteraction
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = CompositeSystem(
        ...     elements=[CompartmentElement('tumour', x=1.0)],
        ...     interactions=[GompertzGrowthInteraction('growth', target='tumour', r=1.0, K=20.0)],
        ... )
        >>> trajectory = simulate(system, until=30.0, dt=0.1)
        >>> round(float(trajectory.get_values('tumour.x')[-1]), 3)
        20.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0719,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1964.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        state: str = 'x',
    ) -> None:
        super().__init__(label, target=target, r=r, K=K, tau=tau, form='gompertz', state=state)

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the Gompertz growth, with no harvest.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        return self._get_growth(namespace)


class PowerLawGrowthInteraction(GrowthInteraction):
    """Growth that scales as a power of the size, `dx/dt = r x ** p`, without harvest.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose population grows.

        r:
            The growth rate.

        p:
            The exponent, `2/3` for a growing rim and 1 for exponential growth.

        state:
            The name of the state of the target that holds the population.

    Examples:
        >>> from skcomplex.systems.biology.common.kinetics import CompartmentElement
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.biology.oncology import PowerLawGrowthInteraction
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = CompositeSystem(
        ...     elements=[CompartmentElement('tumour', x=1.0)],
        ...     interactions=[PowerLawGrowthInteraction('growth', target='tumour', r=1.0, p=0.5)],
        ... )
        >>> trajectory = simulate(system, until=2.0, dt=0.01)
        >>> round(float(trajectory.get_values('tumour.x')[-1]), 3)
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.32,
        p: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.58,
        state: str = 'x',
    ) -> None:
        super().__init__(label, target=target, r=r, p=p, form='power', state=state)

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the power-law growth, with no harvest.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        return self._get_growth(namespace)


class ExponentialLinearGrowthSystem(BaseSystem):
    """A tumour that grows exponentially while small and linearly later.

    The growth turns linear once it reaches a limiting rate.

    The defaults are the fit to breast cancer volumes in mm³ and days, which switches near day 15. The system has the
    volume `tumour.x`.

    Args:
        n:
            The initial volume.

        r0:
            The exponential growth rate.

        r:
            The linear growth rate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ExponentialLinearGrowthSystem
        >>> ExponentialLinearGrowthSystem().prepare().system_.interactions[0].r0
        0.31
    """

    field = 'biology/oncology'
    topics = ('tumour growth', 'ode', 'piecewise')
    references = GROWTH_REFERENCES
    until = 40.0
    dt = 0.2

    def __init__(
        self,
        n: float = 2.0,
        *,
        r0: float = 0.31,
        r: float = 67.8,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r0 = r0
        self.r = r

    def _build_system(self) -> CompositeSystem:
        """Return the components of a tumour."""
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('tumour', x=self.n)],
            interactions=[ExponentialLinearGrowthInteraction('growth', target='tumour', r0=self.r0, r=self.r)],
        )


class PowerLawGrowthSystem(BaseSystem):
    """A tumour whose proliferating part scales as a power of its size.

    The growth is `dn/dt = r n^p`.

    For `p < 1` the volume follows `(n0^(1-p) + (1-p) r t)^(1/(1-p))`, a growth slower than exponential. The defaults
    are the fit to breast cancer volumes in mm³ and days. The system has the volume `tumour.x`.

    Args:
        n:
            The initial volume.

        r:
            The growth rate.

        p:
            The exponent.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import PowerLawGrowthSystem
        >>> PowerLawGrowthSystem(p=2 / 3).prepare().system_.interactions[0].p
        0.6666666666666666
    """

    field = 'biology/oncology'
    topics = ('tumour growth', 'ode', 'power law')
    references = GROWTH_REFERENCES
    until = 40.0
    dt = 0.2

    def __init__(
        self,
        n: float = 10.0,
        *,
        r: float = 1.32,
        p: float = 0.58,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.p = p

    def _build_system(self) -> CompositeSystem:
        """Return the components of a tumour whose proliferating part scales as a power of its size.

        The growth is `dn/dt = r n^p`.
        """
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('tumour', x=self.n)],
            interactions=[PowerLawGrowthInteraction('growth', target='tumour', r=self.r, p=self.p)],
        )


class VonBertalanffySystem(BaseSystem):
    """A tumour that grows by the balance of anabolism and catabolism.

    The growth is `dn/dt = r n^p - d n`.

    The volume tends to `(r / d)^(1/(1-p))`. The defaults are the fit to breast cancer volumes in mm³ and days. The
    system has the volume `tumour.x`.

    Args:
        n:
            The initial volume.

        r:
            The growth rate.

        p:
            The exponent of the growth, below 1.

        d:
            The loss rate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import VonBertalanffySystem
        >>> VonBertalanffySystem().prepare().system_.interactions[1].d
        0.808
    """

    field = 'biology/oncology'
    topics = ('tumour growth', 'ode', 'allometry')
    references = (*GROWTH_REFERENCES, 'von Bertalanffy 1957')
    until = 80.0
    dt = 0.4

    def __init__(
        self,
        n: float = 1.0,
        *,
        r: float = 2.32,
        p: float = 0.918,
        d: float = 0.808,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.p = p
        self.d = d

    def _build_system(self) -> CompositeSystem:
        """Return the components of a tumour that grows by the balance of anabolism and catabolism.

        The growth is `dn/dt = r n^p - d n`.
        """
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('tumour', x=self.n)],
            interactions=[
                PowerLawGrowthInteraction('growth', target='tumour', r=self.r, p=self.p),
                MortalityInteraction('loss', target='tumour', d=self.d),
            ],
        )


class GeneralizedLogisticSystem(BaseSystem):
    """A tumour that saturates at a carrying capacity.

    The growth is `dn/dt = r n (1 - (n / K)^p)`, logistic for `p = 1`.

    The volume follows `n0 K / (n0^p + (K^p - n0^p) exp(-r p t))^(1/p)`. The defaults are the logistic fit to breast
    cancer volumes in mm³ and days. The system has the volume `tumour.x`.

    Args:
        n:
            The initial volume.

        r:
            The growth rate at low density.

        K:
            The carrying capacity.

        p:
            The exponent of the saturation.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import GeneralizedLogisticSystem
        >>> GeneralizedLogisticSystem(p=0.5).prepare().system_.interactions[0].p
        0.5
    """

    field = 'biology/oncology'
    topics = ('tumour growth', 'ode', 'logistic', 'Richards')
    references = (*GROWTH_REFERENCES, 'Richards 1959')
    until = 40.0
    dt = 0.2

    def __init__(
        self,
        n: float = 10.0,
        *,
        r: float = 0.305,
        K: float = 1221.0,
        p: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.K = K
        self.p = p

    def _build_system(self) -> CompositeSystem:
        """Return the components of a tumour that saturates at a carrying capacity."""
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('tumour', x=self.n)],
            interactions=[GrowthInteraction('growth', target='tumour', r=self.r, K=self.K, p=self.p, form='logistic')],
        )


class GompertzSystem(BaseSystem):
    """A tumour whose growth slows as the logarithm of the room left.

    The growth is `dn/dt = r n log(K / n)`.

    The volume follows `K exp(log(n0 / K) exp(-r t))` and tends to `K`. The rate is the fitted decay of the
    proliferation of breast cancer, per day, and the capacity is in mm³. The system has the volume `tumour.x`.

    Args:
        n:
            The initial volume.

        r:
            The rate at which the growth slows.

        K:
            The carrying capacity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import GompertzSystem
        >>> GompertzSystem(K=1000.0).prepare().system_.interactions[0].K
        1000.0
    """

    field = 'biology/oncology'
    topics = ('tumour growth', 'ode', 'sigmoid')
    references = (*GROWTH_REFERENCES, 'Gompertz 1825')
    until = 100.0
    dt = 0.5

    def __init__(
        self,
        n: float = 10.0,
        *,
        r: float = 0.0719,
        K: float = 1964.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.K = K

    def _build_system(self) -> CompositeSystem:
        """Return the components of a tumour whose growth slows as the room left shrinks.

        The growth slows as the logarithm of the room left.
        """
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('tumour', x=self.n)],
            interactions=[GompertzGrowthInteraction('growth', target='tumour', r=self.r, K=self.K)],
        )


class DynamicCarryingCapacitySystem(BaseSystem):
    """Gompertz growth under a carrying capacity that keeps rising.

    The vessels on the tumour surface keep raising the capacity.

    The tumour follows `dn/dt = r n log(K / n)` and the capacity `dK/dt = d n^(2/3)`, so the tumour first fills its
    initial capacity and then grows as fast as its vasculature, roughly as `t^3`. The defaults are the fit to breast
    cancer volumes in mm³ and days. The system has the volume `tumour.x` and the capacity `vessels.K`.

    Args:
        n:
            The initial volume.

        K:
            The initial carrying capacity.

        r:
            The rate at which the growth slows.

        d:
            The rate at which the surface of the tumour raises the capacity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import DynamicCarryingCapacitySystem
        >>> DynamicCarryingCapacitySystem().prepare().system_.elements[1].K
        12.7
    """

    field = 'biology/oncology'
    topics = ('tumour growth', 'ode', 'angiogenesis', 'sigmoid')
    references = (*GROWTH_REFERENCES, 'Hahnfeldt et al. 1999')
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        n: float = 1.0,
        K: float = 12.7,
        *,
        r: float = 2.63,
        d: float = 0.829,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.K = K
        self.r = r
        self.d = d

    def _build_system(self) -> CompositeSystem:
        """Return the components of Gompertz growth under a carrying capacity."""
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('tumour', x=self.n), VasculatureElement('vessels', K=self.K)],
            interactions=[
                VascularSupportInteraction('support', tumour='tumour', vasculature='vessels', r=self.r, d=self.d),
            ],
        )


class SkipperLogKillSystem(BaseSystem):
    """The log-kill system of an exponentially growing tumour.

    Every dose kills a fixed fraction of the tumour.

    The tumour follows `dn/dt = r n - a n U(t)`, where the exposure `U` is `dose` for `width` days every `period` days.
    Every dose lowers `log n` by `a dose width`, whatever the size, and the tumour regrows between doses. The growth
    rate is the exponential fit to breast cancer, per day. The system has the cells `tumour.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import SkipperLogKillSystem
        >>> SkipperLogKillSystem(period=21.0).prepare().system_.interactions[1].period
        21.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'ode', 'log kill', 'dosing')
    references = ('Padmanabhan, Meskin, and Al Moustafa 2021', 'Skipper, Schabel, and Wilcox 1964')
    until = 42.0
    dt = 0.1

    def __init__(
        self,
        n: float = 1e6,
        *,
        r: float = 0.223,
        a: float = 2.0,
        dose: float = 1.0,
        period: float = 7.0,
        width: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.a = a
        self.dose = dose
        self.period = period
        self.width = width

    def _build_system(self) -> CompositeSystem:
        """Return the components of the log-kill system."""
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('tumour', x=self.n)],
            interactions=[
                ExponentialGrowthInteraction('growth', target='tumour', r=self.r),
                LogKillInteraction(
                    'kill',
                    target='tumour',
                    a=self.a,
                    dose=self.dose,
                    period=self.period,
                    width=self.width,
                ),
            ],
        )


class NortonSimonSystem(BaseSystem):
    """The Norton-Simon system of a Gompertzian tumour under a dose schedule.

    The schedule kills the tumour in proportion to its size.

    The tumour follows `dn/dt = r n log(K / n) - a n U(t)`. A Gompertzian tumour regrows faster the smaller it is, so a
    dose that shrinks it is followed by fast regrowth, and the schedule settles on a cycle below the capacity. The
    system has the volume `tumour.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import NortonSimonSystem
        >>> NortonSimonSystem().prepare().system_.interactions[1].start
        42.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'ode', 'log kill', 'Gompertz', 'dosing')
    references = ('Padmanabhan, Meskin, and Al Moustafa 2021', 'Norton and Simon 1977')
    until = 150.0
    dt = 0.5

    def __init__(
        self,
        n: float = 100.0,
        *,
        r: float = 0.0719,
        K: float = 1964.0,
        a: float = 1.0,
        dose: float = 1.0,
        period: float = 21.0,
        width: float = 1.0,
        start: float = 42.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.K = K
        self.a = a
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start

    def _build_system(self) -> CompositeSystem:
        """Return the components of the Norton-Simon system."""
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('tumour', x=self.n)],
            interactions=[
                GompertzGrowthInteraction('growth', target='tumour', r=self.r, K=self.K),
                LogKillInteraction(
                    'kill',
                    target='tumour',
                    a=self.a,
                    dose=self.dose,
                    period=self.period,
                    width=self.width,
                    start=self.start,
                ),
            ],
        )
