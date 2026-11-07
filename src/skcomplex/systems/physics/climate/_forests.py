"""Forest dieback: a tropical forest that warms its soil, alone or coupled with the mitigation of a population."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, exp

from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import delayed
from ....core.components.system import System
from ..._catalog import register_system
from ._forest_cover import ForestCoverElement
from ._mitigators import MitigatorsElement


def _temperature(forest: SimpleNamespace, cover: Expression) -> Expression:
    """Return the local temperature `Tv + a (1 - v)` of a forest of a given cover."""
    return forest.Tv + forest.a * (1 - cover)


def _growth(forest: SimpleNamespace) -> Expression:
    """Return the growth rate `g0 (1 - ((T - Topt) / beta)²)` of a forest at its local temperature."""
    return forest.g0 * (1 - ((_temperature(forest, forest.v) - forest.Topt) / forest.beta) ** 2)


def _logistic(forest: SimpleNamespace, efficiency: Expression | float) -> Expression:
    """Return the logistic growth of a forest, scaled by an efficiency, minus its disturbance."""
    return _growth(forest) * efficiency * forest.v * (1 - forest.v) - forest.gamma * forest.v


def _replicator(s: SimpleNamespace) -> Expression:
    """Return the replicator rate of the share of mitigators under the perceived warming of the forest."""
    people, forest = s.target, s.sources[0]
    past = delayed(forest.v, people.tp)
    perceived = people.tf / people.tp * (_temperature(forest, forest.v) - _temperature(forest, past))
    cost = people.fmax / (1 + exp(-people.w * (perceived - people.Tc)))
    return people.x * (1 - people.x) * (-people.alpha + cost + people.delta * (2 * people.x - 1))


@register_system(
    field='physics/climate',
    tags=('tipping points', 'vegetation', 'bistability', 'hysteresis', 'ode'),
    references=('Van Nes et al. 2014', 'Bury, Bauch, and Anand 2019', 'Liu and Bauch 2023'),
    defaults={'until': 200.0, 'dt': 0.5},
)
def forest_dieback(
    v: float = 0.8,
    *,
    Tv: float = 30.0,
    g0: float = 2.0,
    beta: float = 10.0,
    Topt: float = 28.0,
    a: float = 5.0,
    gamma: float = 0.2,
) -> System:
    """Return the forest dieback model of the Amazon, whose cover tips to bare soil when the ambient temperature rises.

    The cover follows `v' = g v (1 - v) - gamma v`, with the growth rate `g = g0 (1 - ((T - Topt) / beta)²)` at the
    local temperature `T = Tv + a (1 - v)`. Losing forest warms the soil, which lowers the growth and loses more forest,
    a positive feedback. Bare soil `v = 0` is always a fixed point, and the forest survives at the root of
    `(1 - v) g(v) = gamma`. For the defaults the forest and bare soil are both stable for ambient temperatures between
    about 32.5 and 34.7 degrees, the bistable range of a hysteresis loop, and the forest dies beyond it.

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

    Returns:
        system:
            The system, with the vegetation cover `forest.v`.

    Examples:
        >>> from skcomplex.systems.physics.climate import forest_dieback
        >>> forest_dieback(Tv=33.0).elements[0].Tv
        33.0
    """
    forest = ForestCoverElement('forest', v=v, Tv=Tv, g0=g0, beta=beta, Topt=Topt, a=a, gamma=gamma)
    growth = RateInteraction('growth', target='forest', state='v', rate=lambda s: _logistic(s.target, 1))
    return System(elements=[forest], interactions=[growth])


@register_system(
    field='physics/climate',
    tags=('tipping points', 'vegetation', 'social dynamics', 'replicator', 'delay', 'dde'),
    references=('Liu and Bauch 2023', 'Bury, Bauch, and Anand 2019'),
    defaults={'until': 200.0, 'dt': 0.5},
)
def social_forest_dieback(
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
) -> System:
    """Return the forest dieback model coupled with the replicator dynamics of mitigators who perceive its warming.

    The forest grows as in `forest_dieback`, with its growth rate scaled by `0.2 + 0.4 x`, so that mitigation by the
    share `x` of the population helps it. The share follows the replicator equation
    `x' = x (1 - x) (-alpha + f(Tf) + delta (2 x - 1))`, where the cost of warming `f = fmax / (1 + exp(-w (Tf - Tc)))`
    reads the warming `Tf = tf / tp (T(t) - T(t - tp))` projected from the last `tp` years, a delay differential
    equation. Before the start the cover keeps its initial value. With no warming the shares 0 and 1 are stable and the
    share `x* = 1 - (fmax / 2) / (1 + exp(w Tc))` between them is unstable, for the defaults of `alpha`, `delta`, and
    `w`. The population converges to all or no mitigators, and the forest to a positive cover or to bare soil, the
    green, brown, and barren outcomes, with long swings for some temperatures.

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

    Returns:
        system:
            The system, with the vegetation cover `forest.v` and the share of mitigators `people.x`.

    Examples:
        >>> from skcomplex.systems.physics.climate import social_forest_dieback
        >>> social_forest_dieback(Tc=2.5).elements[1].Tc
        2.5
    """
    forest = ForestCoverElement('forest', v=v, Tv=Tv)
    people = MitigatorsElement('people', x=x, alpha=alpha, delta=delta, fmax=fmax, w=w, Tc=Tc, tp=tp, tf=tf)
    growth = RateInteraction(
        'growth',
        target='forest',
        state='v',
        sources=['people'],
        rate=lambda s: _logistic(s.target, 0.2 + 0.4 * s.sources[0].x),
    )
    mitigation = RateInteraction('mitigation', target='people', state='x', sources=['forest'], rate=_replicator)
    return System(elements=[forest, people], interactions=[growth, mitigation])
