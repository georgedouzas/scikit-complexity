"""Optimal harvesting: the effort that maximises the discounted profit of a renewable resource."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, parameter, state
from ....core.components import Element
from ....core.laws import Law, Objective, Rate, Symbols
from ....core.runtime import register_model
from ....core.solvers import Collocation
from ....core.system import Observable, System

MAX_EFFORT = 1.0


def _catch(s: Symbols) -> Expression:
    """Return the catch per unit time, `q E x`."""
    resource = s.resource
    catch: Expression = resource.q * resource.effort * resource.x
    return catch


class HarvestedResource(Element):
    """Logistic resource whose harvest effort is a bounded control input, after Clark.

    The stock follows `x' = r x (1 - x / K) - q E x` under the effort `E` between 0 and its largest value. The
    harvest earns the price `p` per unit of catch and costs `c` per unit of effort, and the owner maximises the
    discounted profit `∫ exp(-delta t) (p q x - c) E dt`. The profit is linear in the effort, so the optimal effort is
    bang-bang or singular: it takes a bound until the stock reaches the level where the yield net of costs earns the
    discount rate, holds the stock there, and harvests at the largest effort at the end of the horizon.

    Args:
        label:
            The label of the resource.

        x:
            The initial stock.

        effort:
            The harvest effort, the control input.

        r:
            The intrinsic growth rate.

        K:
            The carrying capacity.

        q:
            The catchability.

        p:
            The price per unit of catch.

        c:
            The cost per unit of effort.

        delta:
            The discount rate.

    Examples:
        >>> from skcomplex.models.economics.resources import HarvestedResource
        >>> HarvestedResource('resource', x=0.5).x
        0.5
    """

    x: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    effort: float = parameter(default=0.0, adaptive=True, bounds=(0.0, MAX_EFFORT))
    r: float = parameter(default=1.0, domain=Domain.POSITIVE)
    K: float = parameter(default=1.0, domain=Domain.POSITIVE)
    q: float = parameter(default=1.0, domain=Domain.POSITIVE)
    p: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    c: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    delta: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the regrowth less the catch, and the discounted profit to maximise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        profit = exp(-s.delta * s.t) * (s.p * s.q * s.x - s.c) * s.effort
        return [
            Rate(s.x, s.r * s.x * (1 - s.x / s.K) - s.q * s.effort * s.x),
            Objective(profit, sense='max', label='profit'),
        ]


@register_model(
    field='economics/resources',
    tags=('optimal control', 'harvesting', 'bang-bang', 'singular arc', 'ode'),
    references=('Clark 1990',),
    defaults={'until': 20.0, 'dt': 0.2, 'solver': Collocation(nodes=101, scheme='trapezoidal')},
    observables=lambda system: [Observable('catch', _catch)],
)
def optimal_harvesting(
    x: float = 1.0,
    *,
    r: float = 1.0,
    K: float = 1.0,
    q: float = 1.0,
    p: float = 1.0,
    c: float = 0.2,
    delta: float = 0.1,
) -> System:
    """Return a logistic resource harvested with the effort that maximises its discounted profit.

    Run with an optimal control solver, the effort starts at a bound, drives the stock to the level `x*` where
    `r (1 - 2 x* / K) + c r (1 - x* / K) / (p q x* - c) = delta`, holds it there with the effort `r (1 - x* / K) / q`,
    and rises to its largest value at the end of the horizon, when the stock left has no value.

    Args:
        x:
            The initial stock.

        r:
            The intrinsic growth rate.

        K:
            The carrying capacity.

        q:
            The catchability.

        p:
            The price per unit of catch.

        c:
            The cost per unit of effort.

        delta:
            The discount rate.

    Returns:
        system:
            The model, with the stock `resource.x`, the effort `resource.effort`, the discounted profit per unit time
            `resource.profit`, and the observable `catch`.

    Examples:
        >>> from skcomplex.models.economics.resources import optimal_harvesting
        >>> optimal_harvesting(delta=0.05).components[0].delta
        0.05
    """
    resource = HarvestedResource('resource', x=x, r=r, K=K, q=q, p=p, c=c, delta=delta)
    return System(components=[resource])
