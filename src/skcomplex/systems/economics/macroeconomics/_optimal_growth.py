"""Optimal growth: the consumption that maximises the discounted utility of an economy that accumulates capital."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, log, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.system import System
from ....core.engine.solvers import CollocationSolver
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system

MIN_CONSUMPTION = 0.05
MAX_CONSUMPTION = 5.0


def _saving_rate(s: SimpleNamespace) -> Expression:
    """Return the share of output saved, `1 - c / (A k^alpha)`."""
    economy = s.economy
    rate: Expression = 1 - economy.c / (economy.A * economy.k**economy.alpha)
    return rate


def _utility(s: SimpleNamespace) -> Expression:
    """Return the discounted utility of consumption per unit time, `exp(-rho t) log(c)`."""
    economy = s.economy
    utility: Expression = exp(-economy.rho * s.t) * log(economy.c)
    return utility


def _capital_value(s: SimpleNamespace) -> Expression:
    """Return the discounted value of the capital left at the horizon, `exp(-rho T) v k(T)`."""
    economy = s.economy
    value: Expression = exp(-economy.rho * s.t) * economy.v * economy.k
    return value


class RamseyEconomyElement(Element):
    """Economy that splits its output between consumption and capital, after Ramsey, Cass, and Koopmans.

    Capital follows `k' = A k^alpha - delta k - c`, a rate interaction of the factory, under the consumption `c`, a
    bounded control input. The planner maximises the discounted utility `∫ exp(-rho t) log(c) dt` plus the discounted
    value `exp(-rho T) v k(T)` of the capital left at the horizon `T`. Pontryagin's principle gives the Euler
    equation `c' / c = alpha A k^(alpha - 1) - delta - rho`, so capital approaches the modified golden rule
    `alpha A k*^(alpha - 1) = delta + rho` along the saddle path.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        k:
            The initial capital per worker.

        c:
            The consumption per worker, the control input.

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

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import RamseyEconomyElement
        >>> RamseyEconomyElement('economy', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        c: Annotated[float, ParameterSpec(adaptive=True, bounds=(MIN_CONSUMPTION, MAX_CONSUMPTION))] = 0.5,
        A: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        delta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        rho: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.03,
        v: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k
        self.c = c
        self.A = A
        self.alpha = alpha
        self.delta = delta
        self.rho = rho
        self.v = v


def _net_saving(s: SimpleNamespace) -> Expression:
    """Return the output not consumed minus depreciation, `A k^alpha - delta k - c`."""
    economy = s.target
    rate: Expression = economy.A * economy.k**economy.alpha - economy.delta * economy.k - economy.c
    return rate


@register_system(
    field='economics/macroeconomics',
    tags=('optimal control', 'growth', 'capital', 'collocation', 'ode'),
    references=('Dilão 2023',),
    defaults={
        'until': 40.0,
        'dt': 0.5,
        'solver': CollocationSolver(objective=_utility, terminal=_capital_value, sense='max', nodes=81),
    },
    observables=lambda system: [ObservableSpec('saving_rate', _saving_rate)],
)
def optimal_consumption(
    k: float = 1.0,
    *,
    A: float = 1.0,
    alpha: float = 0.3,
    delta: float = 0.05,
    rho: float = 0.03,
    v: float = 0.7,
) -> System:
    """Return the Ramsey economy that chooses the consumption path maximising its discounted utility.

    Run with an optimal control solver from little capital, consumption rises with capital along the saddle path
    towards the modified golden rule, and the economy saves a falling share of its output.

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

    Returns:
        system:
            The system, with the capital `economy.k`, the consumption `economy.c`, and the observable `saving_rate`.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import optimal_consumption
        >>> optimal_consumption(rho=0.05).elements[0].rho
        0.05
    """
    return System(
        elements=[RamseyEconomyElement('economy', k=k, A=A, alpha=alpha, delta=delta, rho=rho, v=v)],
        interactions=[RateInteraction('accumulation', target='economy', state='k', rate=_net_saving)],
    )
