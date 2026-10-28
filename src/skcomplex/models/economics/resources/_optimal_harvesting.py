"""Optimal harvesting: the effort that maximises the discounted profit of a renewable resource."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import CollocationSolver
from ..._catalog import register_model

MAX_EFFORT = 1.0


def _catch(s: SimpleNamespace) -> Expression:
    """Return the catch per unit time, `q E x`."""
    resource = s.resource
    catch: Expression = resource.q * resource.effort * resource.x
    return catch


def _profit(s: SimpleNamespace) -> Expression:
    """Return the discounted profit per unit time, `exp(-delta t) (p q x - c) E`."""
    resource = s.resource
    margin = resource.p * resource.q * resource.x - resource.c
    profit: Expression = exp(-resource.delta * s.t) * margin * resource.effort
    return profit


class HarvestedResourceElement(Element):
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
        >>> from skcomplex.models.economics.resources import HarvestedResourceElement
        >>> HarvestedResourceElement('resource', x=0.5).x
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        effort: Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, MAX_EFFORT))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        delta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.effort = effort
        self.r = r
        self.K = K
        self.q = q
        self.p = p
        self.c = c
        self.delta = delta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the regrowth less the catch.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.x, s.r * s.x * (1 - s.x / s.K) - s.q * s.effort * s.x)]


@register_model(
    field='economics/resources',
    tags=('optimal control', 'harvesting', 'bang-bang', 'singular arc', 'ode'),
    references=('Clark 1990',),
    defaults={
        'until': 20.0,
        'dt': 0.2,
        'solver': CollocationSolver(objective=_profit, sense='max', nodes=101, scheme='trapezoidal'),
    },
    observables=lambda system: [ObservableSpec('catch', _catch)],
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
            The model, with the stock `resource.x`, the effort `resource.effort`, and the observable `catch`.

    Examples:
        >>> from skcomplex.models.economics.resources import optimal_harvesting
        >>> optimal_harvesting(delta=0.05).elements[0].delta
        0.05
    """
    resource = HarvestedResourceElement('resource', x=x, r=r, K=K, q=q, p=p, c=c, delta=delta)
    return System(elements=[resource])
