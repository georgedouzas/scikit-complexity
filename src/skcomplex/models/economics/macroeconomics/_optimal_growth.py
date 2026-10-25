"""Optimal growth: the consumption that maximises the discounted utility of an economy that accumulates capital."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp, log

from ....core.base import Expression, NumericDomain, parameter, state
from ....core.components import ElementComponent
from ....core.laws import BaseLaw, Objective, RateLaw, Symbols
from ....core.system import Observable, System
from ....engine.runtime import register_model
from ....formalisms.control import CollocationSolver

MIN_CONSUMPTION = 0.05
MAX_CONSUMPTION = 5.0


def _saving_rate(s: Symbols) -> Expression:
    """Return the share of output saved, `1 - c / (A k^alpha)`."""
    economy = s.economy
    rate: Expression = 1 - economy.c / (economy.A * economy.k**economy.alpha)
    return rate


class RamseyEconomyComponent(ElementComponent):
    """Economy that splits its output between consumption and capital, after Ramsey, Cass, and Koopmans.

    Capital follows `k' = A k^alpha - delta k - c` under the consumption `c`, a bounded control input. The planner
    maximises the discounted utility `∫ exp(-rho t) log(c) dt` plus the discounted value `exp(-rho T) v k(T)` of the
    capital left at the horizon `T`. Pontryagin's principle gives the Euler equation `c' / c = alpha A k^(alpha - 1) -
    delta - rho`, so capital approaches the modified golden rule `alpha A k*^(alpha - 1) = delta + rho` along the
    saddle path.

    Args:
        label:
            The label of the economy.

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
        >>> from skcomplex.models.economics.macroeconomics import RamseyEconomyComponent
        >>> RamseyEconomyComponent('economy', k=2.0).k
        2.0
    """

    k: float = state(default=1.0, domain=NumericDomain.POSITIVE)
    c: float = parameter(default=0.5, adaptive=True, bounds=(MIN_CONSUMPTION, MAX_CONSUMPTION))
    A: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    alpha: float = parameter(default=0.3, domain=NumericDomain.POSITIVE)
    delta: float = parameter(default=0.05, domain=NumericDomain.NONNEGATIVE)
    rho: float = parameter(default=0.03, domain=NumericDomain.POSITIVE)
    v: float = parameter(default=0.7, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the accumulation of capital and the discounted utility to maximise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        discount = exp(-s.rho * s.t)
        return [
            RateLaw(s.k, s.A * s.k**s.alpha - s.delta * s.k - s.c),
            Objective(discount * log(s.c), terminal=discount * s.v * s.k, sense='max', label='utility'),
        ]


@register_model(
    field='economics/macroeconomics',
    tags=('optimal control', 'growth', 'capital', 'collocation', 'ode'),
    references=('Dilão 2023',),
    defaults={'until': 40.0, 'dt': 0.5, 'solver': CollocationSolver(nodes=81)},
    observables=lambda system: [Observable('saving_rate', _saving_rate)],
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
            The model, with the capital `economy.k`, the consumption `economy.c`, the discounted utility per unit time
            `economy.utility`, and the observable `saving_rate`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import optimal_consumption
        >>> optimal_consumption(rho=0.05).components[0].rho
        0.05
    """
    return System(components=[RamseyEconomyComponent('economy', k=k, A=A, alpha=alpha, delta=delta, rho=rho, v=v)])
