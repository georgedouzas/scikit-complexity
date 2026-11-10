"""Optimal harvesting: the effort that maximises the profit of a resource.

The resource is renewable. The profit is discounted.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.engine.solvers import CollocationSolver
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations

MAX_EFFORT = 1.0


def _get_profit(s: SimpleNamespace) -> Expression:
    """Return the discounted profit per unit time, `exp(-delta t) (p q x - c) E`."""
    resource = s.resource
    margin = resource.p * resource.q * resource.x - resource.c
    profit: Expression = exp(-resource.delta * s.t) * margin * resource.effort
    return profit


def _catch(s: SimpleNamespace) -> Expression:
    """Return the catch per unit time, `q E x`."""
    resource = s.resource
    catch: Expression = resource.q * resource.effort * resource.x
    return catch


class OptimalHarvestingSystem(DynamicalSystem):
    """A logistic resource harvested with the effort that maximises its profit.

    The profit is discounted. Run with an optimal control solver, the effort starts at a bound, drives the stock to the
    level `x*` where `r (1 - 2 x* / K) + c r (1 - x* / K) / (p q x* - c) = delta`, holds it there with the effort
    `r (1 - x* / K) / q`, and rises to its largest value at the end of the horizon, when the stock left has no value.
    The system has the stock `resource.x`, the effort `resource.effort`, and the observable `catch`.

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
        >>> OptimalHarvestingSystem(delta=0.05).prepare().system_.elements[0].get_value('delta')
        0.05
    """

    field = 'economics/resources'
    topics = ('optimal control', 'harvesting', 'bang-bang', 'singular arc', 'ode')
    references = ('Clark 1990',)
    until = 20.0
    dt = 0.2
    solver = CollocationSolver(objective=_get_profit, sense='max', nodes=101, scheme='trapezoidal')
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
        super().__init__(label=label)
        self.x = x
        self.r = r
        self.K = K
        self.q = q
        self.p = p
        self.c = c
        self.delta = delta

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'resource'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec(domain=RealSet.closed_open(0, oo))),
            'effort': DeclaredVariableSpec(0.0, ParameterSpec(adaptive=True, bounds=(0.0, MAX_EFFORT))),
            'r': DeclaredVariableSpec(self.r, ParameterSpec(domain=RealSet.open(0, oo))),
            'K': DeclaredVariableSpec(self.K, ParameterSpec(domain=RealSet.open(0, oo))),
            'q': DeclaredVariableSpec(self.q, ParameterSpec(domain=RealSet.open(0, oo))),
            'p': DeclaredVariableSpec(self.p, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'c': DeclaredVariableSpec(self.c, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'delta': DeclaredVariableSpec(self.delta, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the regrowth less the catch."""
        return {
            'x': lambda s: (
                s.target.r * s.target.x * (1 - s.target.x / s.target.K) - s.target.q * s.target.effort * s.target.x
            ),
        }
