"""Macroeconomics systems: heterogeneous firms that invest out of their profits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class InvestingFirmView(AgentView):
    """Firm as its rule sees it: its states `A`, `K`, `Z`, `B`, `Y`, `P` and parameters `gamma`, `phi`, `r`, `pbar`."""

    A: float
    K: float
    Z: float
    B: float
    Y: float
    P: float
    gamma: float
    phi: float
    r: float
    pbar: float


class InvestingFirmElement(Element):
    """Firm of the toy business-cycle system, which invests a share of its last profit and sells at a random price.

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
        >>> from skcomplex.systems.economics.macroeconomics import InvestingFirmElement
        >>> GroupElement('firms', element=InvestingFirmElement, size=10, values={'gamma': 1.2}).size
        10
    """

    draws = True

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
    def get_price(agent: InvestingFirmView, system: SystemView) -> float:
        """Return a random price, `pbar + 2 U(0, 1)`.

        Args:
            agent:
                The firm.

            system:
                The system view of the pricing.

        Returns:
            price:
                The price of the period.
        """
        return agent.pbar + 2.0 * float(system.rng.random())

    @staticmethod
    def get_steps() -> tuple[tuple[str, str, Callable[[InvestingFirmView, SystemView], float]], ...]:
        """Return the steps of a period in their order, each a label, a state, and the function of its next value.

        A firm invests, produces, borrows, sells, earns, adds the profit to its net worth, and, when the net worth
        turns negative, is replaced by an entrant, whose profit, capital, and net worth the last three steps reset.

        Returns:
            steps:
                The label of every step, the state it sets, and the function of its next value of the firm and the
                system view.
        """
        return (
            ('investment', 'K', lambda agent, system: agent.K + agent.gamma * agent.Z),
            ('production', 'Y', lambda agent, system: agent.phi * agent.K),
            ('borrowing', 'B', lambda agent, system: max(agent.K - agent.A, 0.0)),
            ('pricing', 'P', InvestingFirmElement.get_price),
            ('profit', 'Z', lambda agent, system: agent.P * agent.Y - agent.r * agent.K),
            ('accumulation', 'A', lambda agent, system: agent.A + agent.Z),
            ('entry_profit', 'Z', lambda agent, system: 0.0 if agent.A < 0 else agent.Z),
            ('entry_capital', 'K', lambda agent, system: 1.0 if agent.A < 0 else agent.K),
            ('entry_worth', 'A', lambda agent, system: 1.0 if agent.A < 0 else agent.A),
        )


class ToyFirmInvestmentSystem(BaseSystem):
    """Heterogeneous firms that invest out of past profits and sell at random prices, with entry and exit.

    The firms start with net worth and capital 1 and a profit drawn from `pbar + 2 U(0, 1)`, as in the toy system of
    Bargigli et al. in Caiani et al. (2016, sect. 2.1). Aggregate production fluctuates as bankrupt firms are replaced.
    The system has the group `firms` and the aggregate production `production`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import ToyFirmInvestmentSystem
        >>> ToyFirmInvestmentSystem(20).prepare().system_.elements[0].prepare().size_
        20
    """

    field = 'economics/macroeconomics'
    topics = ('business cycle', 'agents', 'firms')
    references = ('Caiani et al. 2016', 'Delli Gatti et al. 2010')
    until = 100
    observables = (ObservableSpec('production', group='firms', reduce='sum', variable='Y'),)

    def __init__(
        self,
        n_firms: int = 100,
        *,
        gamma: float = 1.1,
        phi: float = 0.1,
        r: float = 0.1,
        pbar: float = 0.01,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_firms = n_firms
        self.gamma = gamma
        self.phi = phi
        self.r = r
        self.pbar = pbar
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        profits = self.pbar + 2.0 * np.random.default_rng(self.random_state).random(self.n_firms)
        firms = GroupElement(
            'firms',
            element=InvestingFirmElement,
            size=self.n_firms,
            values={'Z': profits, 'gamma': self.gamma, 'phi': self.phi, 'r': self.r, 'pbar': self.pbar},
        )
        steps = [
            MapInteraction(label, target='firms', state=name, function=function, form='agent')
            for label, name, function in InvestingFirmElement.get_steps()
        ]
        return System(self.label, elements=[firms], interactions=steps, time=DiscreteTime())
