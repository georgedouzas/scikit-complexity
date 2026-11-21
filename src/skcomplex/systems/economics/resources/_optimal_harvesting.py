"""Optimal harvesting: the effort that maximises the profit of a resource.

The resource is renewable. The profit is discounted.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import ObjectiveLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, CollocationSolver
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations

MAX_EFFORT = 1.0


def _catch(s: SimpleNamespace) -> Expression:
    """Return the catch per unit time, `q E x`."""
    resource = s.resource
    catch: Expression = resource.q * resource.effort * resource.x
    return catch


class OptimalHarvestingSystem(DynamicalSystem):
    """A logistic resource harvested with the effort that maximises its profit.

    The profit is discounted, and the objective law `ObjectiveLaw(-profit, controls=[s.target.effort])` maximises it
    over the effort, which lies between 0 and 1. Run with an optimal control solver, the effort starts at a bound,
    drives the stock to the level `x*` where `r (1 - 2 x* / K) + c r (1 - x* / K) / (p q x* - c) = delta`, holds it
    there with the effort `r (1 - x* / K) / q`, and rises to its largest value at the end of the horizon, when the stock
    left has no value. The system has the stock `resource.x`, the effort `resource.effort`, and the observable `catch`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
    observables = (Observable('catch', _catch),)

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.x = x
        self.r = r
        self.K = K
        self.q = q
        self.p = p
        self.c = c
        self.delta = delta

    def _get_solver(self) -> BaseSolver:
        """Return the solver the system is tuned to."""
        return CollocationSolver(nodes=101, scheme='trapezoidal')

    @staticmethod
    def _get_profit(s: SimpleNamespace) -> Expression:
        """Return the discounted profit per unit time, `exp(-delta t) (p q x - c) E`."""
        resource = s.target
        margin = resource.p * resource.q * resource.x - resource.c
        profit: Expression = exp(-resource.delta * s.t) * margin * resource.effort
        return profit

    def _get_interactions(self) -> list[BaseInteraction]:
        """Return the regrowth less the catch, and the profit that the effort maximises."""
        profit = Interaction(
            'profit',
            roles={'target': self._get_element_label()},
            laws=lambda s: [ObjectiveLaw(-self._get_profit(s), controls=[s.target.effort])],
        )
        return [*super()._get_interactions(), profit]

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'resource'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'x': self.x,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the effort, which the profit steers, and the parameters of the model."""
        return {'effort': 0.0, 'r': self.r, 'K': self.K, 'q': self.q, 'p': self.p, 'c': self.c, 'delta': self.delta}

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the stock, the effort, and the parameters."""
        positive, nonnegative = RealSet.open(0, oo), RealSet.closed_open(0, oo)
        return {
            'x': nonnegative,
            'effort': RealSet.closed(0, MAX_EFFORT),
            'r': positive,
            'K': positive,
            'q': positive,
            'p': nonnegative,
            'c': nonnegative,
            'delta': nonnegative,
        }

    def _get_equations(self) -> Equations:
        """Return the regrowth less the catch."""
        return {
            'x': lambda s: (
                s.target.r * s.target.x * (1 - s.target.x / s.target.K) - s.target.q * s.target.effort * s.target.x
            ),
        }
