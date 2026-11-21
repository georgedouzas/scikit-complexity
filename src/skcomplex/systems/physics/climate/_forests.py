"""Forest dieback.

A tropical forest warms its soil, alone or coupled with the mitigation of a population.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ForestCoverElement(Element):
    """Vegetation cover of a tropical forest and the temperature of its soil.

    The cover `v` runs from 0 for bare soil to 1 for full forest. Bare soil is warmer than forest by `a`, so the local
    temperature is `T = Tv + a (1 - v)`. `ForestGrowthInteraction` gives the growth of the cover.

    Args:
        label:
            The label of the forest.

        v:
            The initial vegetation cover.

        Tv:
            The ambient temperature, the temperature under full forest cover, in degrees Celsius.

        a:
            The warming of bare soil over forest.

    Examples:
        >>> from skcomplex.systems.physics.climate import ForestCoverElement
        >>> ForestCoverElement('forest').Tv
        30.0
    """

    def __init__(
        self,
        label: str,
        *,
        v: Annotated[float, RealSet.closed(0, 1)] = 0.8,
        Tv: float = 30.0,
        a: float = 5.0,
    ) -> None:
        super().__init__(label)
        self.v = v
        self.Tv = Tv
        self.a = a


class MitigatorsElement(Element):
    """Share of mitigators in a population.

    The share `x` runs from 0 to 1, and `MitigationInteraction` gives its replicator dynamics.

    Args:
        label:
            The label of the population.

        x:
            The initial share of mitigators.

    Examples:
        >>> from skcomplex.systems.physics.climate import MitigatorsElement
        >>> MitigatorsElement('people', x=0.3).x
        0.3
    """

    def __init__(self, label: str, *, x: Annotated[float, RealSet.closed(0, 1)] = 0.5) -> None:
        super().__init__(label)
        self.x = x


class ForestGrowthInteraction(Interaction):
    """Logistic growth of the cover of a forest under its disturbance.

    The cover `v` of the target follows `v' = g v (1 - v) - gamma v`, with the growth rate `g = g0 (1 - ((T - Topt) /
    beta)²)` at the local temperature `T = Tv + a (1 - v)` of the forest, which peaks at `Topt`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the forest.

        g0:
            The largest growth rate.

        beta:
            The half width of the growth rate against the temperature.

        Topt:
            The optimal temperature of growth.

        gamma:
            The disturbance rate.

    Examples:
        >>> from skcomplex.systems.physics.climate import ForestGrowthInteraction
        >>> ForestGrowthInteraction('growth', target='forest').Topt
        28.0
    """

    law_classes = (RateLaw,)

    @staticmethod
    def _get_temperature(forest: SimpleNamespace, cover: Expression) -> Expression:
        """Return the local temperature `Tv + a (1 - v)` of a forest of a given cover."""
        return forest.Tv + forest.a * (1 - cover)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        g0: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        beta: Annotated[float, RealSet.open(0, oo)] = 10.0,
        Topt: float = 28.0,
        gamma: Annotated[float, RealSet.closed_open(0, oo)] = 0.2,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.g0 = g0
        self.beta = beta
        self.Topt = Topt
        self.gamma = gamma

    def _get_efficiency(self, context: SimpleNamespace) -> Expression | float:
        """Return the factor of the growth rate, 1 for a forest on its own."""
        return 1

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the logistic growth of the cover minus its disturbance.

        Args:
            context:
                The namespace of the forest, the time `t`, and the parameters.
        """
        super()._prepare(context)
        forest = context.target
        temperature = self._get_temperature(forest, forest.v)
        growth = context.g0 * (1 - ((temperature - context.Topt) / context.beta) ** 2)
        rate = growth * self._get_efficiency(context) * forest.v * (1 - forest.v) - context.gamma * forest.v
        self.laws_ = [RateLaw(forest.v.diff(context.t), rate)]


class MitigatedForestGrowthInteraction(ForestGrowthInteraction):
    """Growth of the cover of a forest helped by the share of mitigators of a population.

    The growth rate of `ForestGrowthInteraction` is scaled by `0.2 + 0.4 x`, where `x` is the share of mitigators.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the forest.

        mitigators:
            The path of the population whose share of mitigators helps the forest.

        g0:
            The largest growth rate.

        beta:
            The half width of the growth rate against the temperature.

        Topt:
            The optimal temperature of growth.

        gamma:
            The disturbance rate.

    Examples:
        >>> from skcomplex.systems.physics.climate import MitigatedForestGrowthInteraction
        >>> MitigatedForestGrowthInteraction('growth', target='forest', mitigators='people').mitigators
        'people'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        mitigators: Role,
        g0: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        beta: Annotated[float, RealSet.open(0, oo)] = 10.0,
        Topt: float = 28.0,
        gamma: Annotated[float, RealSet.closed_open(0, oo)] = 0.2,
    ) -> None:
        super().__init__(label, target=target, g0=g0, beta=beta, Topt=Topt, gamma=gamma)
        self.mitigators = mitigators

    def _get_efficiency(self, context: SimpleNamespace) -> Expression | float:
        """Return the factor `0.2 + 0.4 x` of the growth rate."""
        efficiency: Expression = 0.2 + 0.4 * context.mitigators.x
        return efficiency


class MitigationInteraction(Interaction):
    """Replicator dynamics of the share of mitigators, who perceive the warming of a forest.

    The share `x` of the target follows `x' = x (1 - x) (-alpha + f + delta (2 x - 1))`, with the cost of mitigation
    `alpha`, the strength `delta` of the social norm, and the cost of warming `f = fmax / (1 + exp(-w (Tf - Tc)))`. The
    perceived warming `Tf = tf / tp (T(t) - T(t - tp))` projects the warming of the forest over the last `tp` years
    over `tf` years.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        forest:
            The path of the forest whose warming the population perceives.

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

    Examples:
        >>> from skcomplex.systems.physics.climate import MitigationInteraction
        >>> MitigationInteraction('mitigation', target='people', forest='forest').tp
        10.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        forest: Role,
        alpha: float = 1.0,
        delta: float = 1.0,
        fmax: Annotated[float, RealSet.closed_open(0, oo)] = 5.0,
        w: Annotated[float, RealSet.closed_open(0, oo)] = 3.0,
        Tc: float = 1.0,
        tp: Annotated[float, RealSet.open(0, oo)] = 10.0,
        tf: Annotated[float, RealSet.closed_open(0, oo)] = 15.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.forest = forest
        self.alpha = alpha
        self.delta = delta
        self.fmax = fmax
        self.w = w
        self.Tc = Tc
        self.tp = tp
        self.tf = tf

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the replicator rate of the share of mitigators.

        Args:
            context:
                The namespace of the population and the forest, the time `t`, and the parameters.
        """
        super()._prepare(context)
        people, forest, t = context.target, context.forest, context.t
        past = forest.v(t=t - context.tp)
        perceived = (
            context.tf
            / context.tp
            * (
                ForestGrowthInteraction._get_temperature(forest, forest.v)
                - ForestGrowthInteraction._get_temperature(forest, past)
            )
        )
        cost = context.fmax / (1 + exp(-context.w * (perceived - context.Tc)))
        rate = people.x * (1 - people.x) * (-context.alpha + cost + context.delta * (2 * people.x - 1))
        self.laws_ = [RateLaw(people.x.diff(t), rate)]


class ForestDiebackSystem(BaseSystem):
    """The forest dieback model of the Amazon.

    The cover tips to bare soil when the ambient temperature rises.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.v = v
        self.Tv = Tv
        self.g0 = g0
        self.beta = beta
        self.Topt = Topt
        self.a = a
        self.gamma = gamma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the forest and the growth of its cover.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        growth = ForestGrowthInteraction(
            'growth',
            target='forest',
            g0=self.g0,
            beta=self.beta,
            Topt=self.Topt,
            gamma=self.gamma,
        )
        self.system_ = System(
            self.label,
            elements=[ForestCoverElement('forest', v=self.v, Tv=self.Tv, a=self.a)],
            interactions=[growth],
        )


class SocialForestDiebackSystem(BaseSystem):
    """The forest dieback model coupled with the replicator dynamics of mitigators.

    The mitigators perceive the warming of the forest.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import SocialForestDiebackSystem
        >>> SocialForestDiebackSystem(Tc=2.5).prepare().system_.interactions[1].Tc
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the forest, the population, and their coupling.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        mitigation = MitigationInteraction(
            'mitigation',
            target='people',
            forest='forest',
            alpha=self.alpha,
            delta=self.delta,
            fmax=self.fmax,
            w=self.w,
            Tc=self.Tc,
            tp=self.tp,
            tf=self.tf,
        )
        self.system_ = System(
            self.label,
            elements=[ForestCoverElement('forest', v=self.v, Tv=self.Tv), MitigatorsElement('people', x=self.x)],
            interactions=[MitigatedForestGrowthInteraction('growth', target='forest', mitigators='people'), mitigation],
        )
