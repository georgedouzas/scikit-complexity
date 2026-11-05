"""Macroeconomics models: heterogeneous firms that invest out of their profits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MapInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._investing_firm_view import InvestingFirmView


class InvestingFirmElement(Element):
    """Firm of the toy business-cycle model, which invests a share of its last profit and sells at a random price.

    Every period the firm invests `I = gamma * Z`, accumulates capital `K = K + I`, produces `Y = phi * K`, borrows
    `B = max(K - A, 0)`, sells at the price `P = pbar + 2 U(0, 1)`, earns `Z = P * Y - r * K`, and adds the profit to
    its net worth `A`. A firm whose net worth turns negative is replaced by an entrant with `A = K = 1` and `Z = 0`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial net worth.

        K:
            The initial capital.

        Z:
            The initial profit.

        B:
            The initial debt.

        Y:
            The initial production.

        P:
            The initial price.

        gamma:
            The investment accelerator.

        phi:
            The productivity of capital.

        r:
            The interest rate.

        pbar:
            The constant part of the price.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import InvestingFirmElement
        >>> GroupElement('firms', element=InvestingFirmElement, size=10, values={'gamma': 1.2}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec()] = 1.0,
        K: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        Z: Annotated[float, StateSpec()] = 0.0,
        B: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Y: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.1,
        phi: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        pbar: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.K = K
        self.Z = Z
        self.B = B
        self.Y = Y
        self.P = P
        self.gamma = gamma
        self.phi = phi
        self.r = r
        self.pbar = pbar

    @classmethod
    def _get_view(cls) -> type[InvestingFirmView]:
        """Return the view of a firm, with its states and parameters."""
        return InvestingFirmView

    @staticmethod
    def get_price(agent: InvestingFirmView, model: ModelView) -> float:
        """Return a random price, `pbar + 2 U(0, 1)`.

        Args:
            agent:
                The firm.

            model:
                The model view of the pricing.

        Returns:
            price:
                The price of the period.
        """
        return agent.pbar + 2.0 * float(model.rng.random())

    @staticmethod
    def get_steps() -> tuple[tuple[str, str, Callable[[InvestingFirmView, ModelView], float]], ...]:
        """Return the steps of a period in their order, each a label, a state, and the function of its next value.

        A firm invests, produces, borrows, sells, earns, adds the profit to its net worth, and, when the net worth
        turns negative, is replaced by an entrant, whose profit, capital, and net worth the last three steps reset.

        Returns:
            steps:
                The label of every step, the state it sets, and the function of its next value of the firm and the
                model view.
        """
        return (
            ('investment', 'K', lambda agent, model: agent.K + agent.gamma * agent.Z),
            ('production', 'Y', lambda agent, model: agent.phi * agent.K),
            ('borrowing', 'B', lambda agent, model: max(agent.K - agent.A, 0.0)),
            ('pricing', 'P', InvestingFirmElement.get_price),
            ('profit', 'Z', lambda agent, model: agent.P * agent.Y - agent.r * agent.K),
            ('accumulation', 'A', lambda agent, model: agent.A + agent.Z),
            ('entry_profit', 'Z', lambda agent, model: 0.0 if agent.A < 0 else agent.Z),
            ('entry_capital', 'K', lambda agent, model: 1.0 if agent.A < 0 else agent.K),
            ('entry_worth', 'A', lambda agent, model: 1.0 if agent.A < 0 else agent.A),
        )


@register_model(
    field='economics/macroeconomics',
    tags=('business cycle', 'agents', 'firms'),
    references=('Caiani et al. 2016', 'Delli Gatti et al. 2010'),
    defaults={'until': 100},
    observables=lambda system: [ObservableSpec('production', group='firms', reduce='sum', variable='Y')],
)
def toy_firm_investment(
    n_firms: int = 100,
    *,
    gamma: float = 1.1,
    phi: float = 0.1,
    r: float = 0.1,
    pbar: float = 0.01,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return heterogeneous firms that invest out of past profits and sell at random prices, with entry and exit.

    The firms start with net worth and capital 1 and a profit drawn from `pbar + 2 U(0, 1)`, as in the toy model of
    Bargigli et al. in Caiani et al. (2016, sect. 2.1). Aggregate production fluctuates as bankrupt firms are replaced.

    Args:
        n_firms:
            The number of firms.

        gamma:
            The investment accelerator.

        phi:
            The productivity of capital.

        r:
            The interest rate.

        pbar:
            The constant part of the price.

        random_state:
            The seed or generator of the initial profits.

    Returns:
        system:
            The model, with the group `firms` and the aggregate production `production`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import toy_firm_investment
        >>> toy_firm_investment(20).elements[0].prepare().size_
        20
    """
    profits = pbar + 2.0 * np.random.default_rng(random_state).random(n_firms)
    firms = GroupElement(
        'firms',
        element=InvestingFirmElement,
        size=n_firms,
        values={'Z': profits, 'gamma': gamma, 'phi': phi, 'r': r, 'pbar': pbar},
    )
    steps = [
        MapInteraction(label, target='firms', state=name, function=function, form='agent')
        for label, name, function in InvestingFirmElement.get_steps()
    ]
    return System(elements=[firms], interactions=steps, time=DiscreteTime())
