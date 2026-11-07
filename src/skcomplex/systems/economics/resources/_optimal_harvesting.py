"""Optimal harvesting: the effort that maximises the discounted profit of a renewable resource."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import CollocationSolver
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

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

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.economics.resources import HarvestedResourceElement
        >>> HarvestedResourceElement('resource', x=0.5).x
        0.5
    """

    law_classes = (RateLaw,)

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


class OptimalHarvestingSystem(BaseSystem):
    """A logistic resource harvested with the effort that maximises its discounted profit.

    Run with an optimal control solver, the effort starts at a bound, drives the stock to the level `x*` where `r (1 - 2
    x* / K) + c r (1 - x* / K) / (p q x* - c) = delta`, holds it there with the effort `r (1 - x* / K) / q`, and rises
    to its largest value at the end of the horizon, when the stock left has no value. The system has the stock
    `resource.x`, the effort `resource.effort`, and the observable `catch`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.resources import OptimalHarvestingSystem
        >>> OptimalHarvestingSystem(delta=0.05).prepare().system_.elements[0].delta
        0.05
    """

    field = 'economics/resources'
    topics = ('optimal control', 'harvesting', 'bang-bang', 'singular arc', 'ode')
    references = ('Clark 1990',)
    until = 20.0
    dt = 0.2
    solver = CollocationSolver(objective=_profit, sense='max', nodes=101, scheme='trapezoidal')
    observables = (ObservableSpec('catch', _catch),)

    def __init__(
        self,
        x: float = 1.0,
        *,
        r: float = 1.0,
        K: float = 1.0,
        q: float = 1.0,
        p: float = 1.0,
        c: float = 0.2,
        delta: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.r = r
        self.K = K
        self.q = q
        self.p = p
        self.c = c
        self.delta = delta

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        resource = HarvestedResourceElement(
            'resource',
            x=self.x,
            r=self.r,
            K=self.K,
            q=self.q,
            p=self.p,
            c=self.c,
            delta=self.delta,
        )
        return System(self.label, elements=[resource])
