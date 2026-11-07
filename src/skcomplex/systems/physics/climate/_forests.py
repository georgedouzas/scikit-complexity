"""Forest dieback: a tropical forest that warms its soil, alone or coupled with the mitigation of a population."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import delayed
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ForestCoverElement(Element):
    """Vegetation cover of a tropical forest, whose growth rate falls on both sides of an optimal temperature.

    The cover `v` runs from 0 for bare soil to 1 for full forest. Bare soil is warmer than forest by `a`, so the local
    temperature is `T = Tv + a (1 - v)`, and the growth rate `g = g0 (1 - ((T - Topt) / beta)²)` peaks at `Topt`. A
    rate interaction gives the logistic growth `g v (1 - v)` and the disturbance `gamma v`, so the element holds the
    state and the parameters it reads.

    Args:
        label:
            The label of the forest.

        visual:
            How the forest is drawn, or `None` for `default_visual_`.

        v:
            The initial vegetation cover.

        Tv:
            The ambient temperature, the temperature under full forest cover, in degrees Celsius.

        g0:
            The largest growth rate.

        beta:
            The half width of the growth rate against the temperature.

        Topt:
            The optimal temperature of growth.

        a:
            The warming of bare soil over forest.

        gamma:
            The disturbance rate.

    Attributes:
        default_visual_:
            The visual of the forest when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import ForestCoverElement
        >>> ForestCoverElement('forest').Topt
        28.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.8,
        Tv: Annotated[float, ParameterSpec()] = 30.0,
        g0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        Topt: Annotated[float, ParameterSpec()] = 28.0,
        a: Annotated[float, ParameterSpec()] = 5.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.Tv = Tv
        self.g0 = g0
        self.beta = beta
        self.Topt = Topt
        self.a = a
        self.gamma = gamma


class MitigatorsElement(Element):
    """Share of mitigators in a population that weighs the cost of mitigation against the perceived cost of warming.

    The share `x` follows the replicator equation `x' = x (1 - x) (-alpha + f + delta (2 x - 1))`, with the cost of
    mitigation `alpha`, the strength `delta` of the social norm, and the cost of warming
    `f = fmax / (1 + exp(-w (Tf - Tc)))`. The perceived warming `Tf = tf / tp (T(t) - T(t - tp))` projects the warming
    of the last `tp` years over `tf` years. A rate interaction reads the temperature it perceives, so the element holds
    the state and the parameters it reads.

    Args:
        label:
            The label of the population.

        visual:
            How the population is drawn, or `None` for `default_visual_`.

        x:
            The initial share of mitigators.

        alpha:
            The cost of mitigation.

        delta:
            The strength of the social norm.

        fmax:
            The largest cost of warming.

        w:
            The steepness of the cost of warming.

        Tc:
            The critical warming.

        tp:
            The years of past warming that the projection reads.

        tf:
            The years ahead of the projection.

    Attributes:
        default_visual_:
            The visual of the population when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import MitigatorsElement
        >>> MitigatorsElement('people').tp
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.5,
        alpha: Annotated[float, ParameterSpec()] = 1.0,
        delta: Annotated[float, ParameterSpec()] = 1.0,
        fmax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        Tc: Annotated[float, ParameterSpec()] = 1.0,
        tp: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        tf: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.alpha = alpha
        self.delta = delta
        self.fmax = fmax
        self.w = w
        self.Tc = Tc
        self.tp = tp
        self.tf = tf


def _temperature(forest: SimpleNamespace, cover: Expression) -> Expression:
    """Return the local temperature `Tv + a (1 - v)` of a forest of a given cover."""
    return forest.Tv + forest.a * (1 - cover)


def _growth(forest: SimpleNamespace) -> Expression:
    """Return the growth rate `g0 (1 - ((T - Topt) / beta)²)` of a forest at its local temperature."""
    return forest.g0 * (1 - ((_temperature(forest, forest.v) - forest.Topt) / forest.beta) ** 2)


def _logistic(forest: SimpleNamespace, efficiency: Expression | float) -> Expression:
    """Return the logistic growth of a forest, scaled by an efficiency, minus its disturbance."""
    return _growth(forest) * efficiency * forest.v * (1 - forest.v) - forest.gamma * forest.v


class ForestDiebackSystem(BaseSystem):
    """The forest dieback model of the Amazon, whose cover tips to bare soil when the ambient temperature rises.

    The cover follows `v' = g v (1 - v) - gamma v`, with the growth rate `g = g0 (1 - ((T - Topt) / beta)²)` at the
    local temperature `T = Tv + a (1 - v)`. Losing forest warms the soil, which lowers the growth and loses more forest,
    a positive feedback. Bare soil `v = 0` is always a fixed point, and the forest survives at the root of `(1 - v) g(v)
    = gamma`. For the defaults the forest and bare soil are both stable for ambient temperatures between about 32.5 and
    34.7 degrees, the bistable range of a hysteresis loop, and the forest dies beyond it. The system has the vegetation
    cover `forest.v`.

    Args:
        v:
            The initial vegetation cover, between 0 and 1.

        Tv:
            The ambient temperature under full forest cover, in degrees Celsius.

        g0:
            The largest growth rate.

        beta:
            The half width of the growth rate against the temperature.

        Topt:
            The optimal temperature of growth.

        a:
            The warming of bare soil over forest.

        gamma:
            The disturbance rate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import ForestDiebackSystem
        >>> ForestDiebackSystem(Tv=33.0).prepare().system_.elements[0].Tv
        33.0
    """

    field = 'physics/climate'
    topics = ('tipping points', 'vegetation', 'bistability', 'hysteresis', 'ode')
    references = ('Van Nes et al. 2014', 'Bury, Bauch, and Anand 2019', 'Liu and Bauch 2023')
    until = 200.0
    dt = 0.5

    def __init__(
        self,
        v: float = 0.8,
        *,
        Tv: float = 30.0,
        g0: float = 2.0,
        beta: float = 10.0,
        Topt: float = 28.0,
        a: float = 5.0,
        gamma: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.v = v
        self.Tv = Tv
        self.g0 = g0
        self.beta = beta
        self.Topt = Topt
        self.a = a
        self.gamma = gamma

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        forest = ForestCoverElement(
            'forest',
            v=self.v,
            Tv=self.Tv,
            g0=self.g0,
            beta=self.beta,
            Topt=self.Topt,
            a=self.a,
            gamma=self.gamma,
        )
        growth = RateInteraction('growth', target='forest', state='v', rate=lambda s: _logistic(s.target, 1))
        return System(self.label, elements=[forest], interactions=[growth])


class SocialForestDiebackSystem(BaseSystem):
    """The forest dieback model coupled with the replicator dynamics of mitigators who perceive its warming.

    The forest grows as in `ForestDiebackSystem`, with its growth rate scaled by `0.2 + 0.4 x`, so that mitigation by
    the share `x` of the population helps it. The share follows the replicator equation `x' = x (1 - x) (-alpha + f(Tf)
    + delta (2 x - 1))`, where the cost of warming `f = fmax / (1 + exp(-w (Tf - Tc)))` reads the warming `Tf = tf / tp
    (T(t) - T(t - tp))` projected from the last `tp` years, a delay differential equation. Before the start the cover
    keeps its initial value. With no warming the shares 0 and 1 are stable and the share `x* = 1 - (fmax / 2) / (1 +
    exp(w Tc))` between them is unstable, for the defaults of `alpha`, `delta`, and `w`. The population converges to all
    or no mitigators, and the forest to a positive cover or to bare soil, the green, brown, and barren outcomes, with
    long swings for some temperatures. The system has the vegetation cover `forest.v` and the share of mitigators
    `people.x`.

    Args:
        v:
            The initial vegetation cover, and its value before the start.

        x:
            The initial share of mitigators.

        Tv:
            The ambient temperature under full forest cover, in degrees Celsius.

        Tc:
            The critical warming.

        alpha:
            The cost of mitigation.

        delta:
            The strength of the social norm.

        fmax:
            The largest cost of warming.

        w:
            The steepness of the cost of warming.

        tp:
            The years of past warming that the projection reads.

        tf:
            The years ahead of the projection.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import SocialForestDiebackSystem
        >>> SocialForestDiebackSystem(Tc=2.5).prepare().system_.elements[1].Tc
        2.5
    """

    field = 'physics/climate'
    topics = ('tipping points', 'vegetation', 'social dynamics', 'replicator', 'delay', 'dde')
    references = ('Liu and Bauch 2023', 'Bury, Bauch, and Anand 2019')
    until = 200.0
    dt = 0.5

    def __init__(
        self,
        v: float = 0.8,
        x: float = 0.5,
        *,
        Tv: float = 30.0,
        Tc: float = 1.0,
        alpha: float = 1.0,
        delta: float = 1.0,
        fmax: float = 5.0,
        w: float = 3.0,
        tp: float = 10.0,
        tf: float = 15.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.v = v
        self.x = x
        self.Tv = Tv
        self.Tc = Tc
        self.alpha = alpha
        self.delta = delta
        self.fmax = fmax
        self.w = w
        self.tp = tp
        self.tf = tf

    @staticmethod
    def _replicator(s: SimpleNamespace) -> Expression:
        """Return the replicator rate of the share of mitigators under the perceived warming of the forest."""
        people, forest = s.target, s.sources[0]
        past = delayed(forest.v, people.tp)
        perceived = people.tf / people.tp * (_temperature(forest, forest.v) - _temperature(forest, past))
        cost = people.fmax / (1 + exp(-people.w * (perceived - people.Tc)))
        return people.x * (1 - people.x) * (-people.alpha + cost + people.delta * (2 * people.x - 1))

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        forest = ForestCoverElement('forest', v=self.v, Tv=self.Tv)
        people = MitigatorsElement(
            'people',
            x=self.x,
            alpha=self.alpha,
            delta=self.delta,
            fmax=self.fmax,
            w=self.w,
            Tc=self.Tc,
            tp=self.tp,
            tf=self.tf,
        )
        growth = RateInteraction(
            'growth',
            target='forest',
            state='v',
            sources=['people'],
            rate=lambda s: _logistic(s.target, 0.2 + 0.4 * s.sources[0].x),
        )
        mitigation = RateInteraction(
            'mitigation',
            target='people',
            state='x',
            sources=['forest'],
            rate=self._replicator,
        )
        return System(self.label, elements=[forest, people], interactions=[growth, mitigation])
