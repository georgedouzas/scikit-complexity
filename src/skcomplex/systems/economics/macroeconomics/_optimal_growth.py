"""Optimal growth: the consumption that maximises discounted utility.

The economy accumulates capital.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, exp, log, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import ObjectiveLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, CollocationSolver
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations

MIN_CONSUMPTION = 0.05
MAX_CONSUMPTION = 5.0


def _get_saving_rate(s: SimpleNamespace) -> Expression:
    """Return the share of output saved, `1 - c / (A k^alpha)`."""
    economy = s.economy
    rate: Expression = 1 - economy.c / (economy.A * economy.k**economy.alpha)
    return rate


class OptimalConsumptionSystem(DynamicalSystem):
    """The Ramsey economy that chooses its consumption path.

    The path maximises its discounted utility and the value of the capital left at the horizon, the objective law
    `ObjectiveLaw(-utility, controls=[s.target.c], terminal=-value)`. Run with an optimal control solver from little
    capital, consumption rises with capital along the saddle path towards the modified golden rule, and the economy
    saves a falling share of its output. The system has the capital `economy.k`, the consumption `economy.c`, and the
    observable `saving_rate`.

    Args:
        k:
            The initial capital per worker.

        A:
            The total factor productivity.

        alpha:
            The capital share of output.

        delta:
            The depreciation rate.

        rho:
            The discount rate.

        v:
            The value per unit of capital left at the horizon.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import OptimalConsumptionSystem
        >>> OptimalConsumptionSystem(rho=0.05).prepare().system_.elements[0].get_params()['rho']
        0.05
    """

    field = 'economics/macroeconomics'
    topics = ('optimal control', 'growth', 'capital', 'collocation', 'ode')
    references = ('Dilão 2023',)
    until = 40.0
    dt = 0.5
    observables = (Observable('saving_rate', _get_saving_rate),)

    def __init__(
        self,
        k: float = 1.0,
        *,
        A: float = 1.0,
        alpha: float = 0.3,
        delta: float = 0.05,
        rho: float = 0.03,
        v: float = 0.7,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.k = k
        self.A = A
        self.alpha = alpha
        self.delta = delta
        self.rho = rho
        self.v = v

    def _get_solver(self) -> BaseSolver:
        """Return the solver the system is tuned to."""
        return CollocationSolver(nodes=81)

    @staticmethod
    def _get_utility(s: SimpleNamespace) -> Expression:
        """Return the discounted utility of consumption per unit time.

        It is `exp(-rho t) log(c)`.
        """
        economy = s.target
        utility: Expression = exp(-economy.rho * s.t) * log(economy.c)
        return utility

    @staticmethod
    def _get_capital_value(s: SimpleNamespace) -> Expression:
        """Return the discounted value of the capital left at the horizon.

        It is `exp(-rho T) v k(T)`.
        """
        economy = s.target
        value: Expression = exp(-economy.rho * s.t) * economy.v * economy.k
        return value

    def _get_interactions(self) -> list[BaseInteraction]:
        """Return the net saving, and the utility and capital value that the consumption maximises."""
        welfare = Interaction(
            'welfare',
            roles={'target': self._get_element_label()},
            laws=lambda s: [
                ObjectiveLaw(-self._get_utility(s), controls=[s.target.c], terminal=-self._get_capital_value(s)),
            ],
        )
        return [*super()._get_interactions(), welfare]

    @staticmethod
    def _get_net_saving(s: SimpleNamespace) -> Expression:
        """Return the output not consumed minus depreciation, `A k^alpha - delta k - c`."""
        economy = s.target
        rate: Expression = economy.A * economy.k**economy.alpha - economy.delta * economy.k - economy.c
        return rate

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'economy'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'k': self.k,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the consumption, which the welfare steers, and the parameters of the model."""
        return {'c': 0.5, 'A': self.A, 'alpha': self.alpha, 'delta': self.delta, 'rho': self.rho, 'v': self.v}

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the capital, the consumption, and the parameters."""
        positive, nonnegative = RealSet.open(0, oo), RealSet.closed_open(0, oo)
        return {
            'k': positive,
            'c': RealSet.closed(MIN_CONSUMPTION, MAX_CONSUMPTION),
            'A': positive,
            'alpha': positive,
            'delta': nonnegative,
            'rho': positive,
            'v': nonnegative,
        }

    def _get_equations(self) -> Equations:
        """Return the components of the system built from its model parameters."""
        return {
            'k': self._get_net_saving,
        }
