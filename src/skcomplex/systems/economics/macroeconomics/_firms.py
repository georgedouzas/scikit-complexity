"""Macroeconomics systems: heterogeneous firms that invest out of their profits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role


class InvestingFirmView(AgentViewProtocol, Protocol):
    """Firm as its rule sees it.

    It holds its states `A`, `K`, `Z`, `B`, `Y`, `P` and parameters `gamma`, `phi`, `r`, `pbar`.
    """

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
    """Firm of the toy business-cycle system.

    It invests a share of its last profit and sells at a random price. Every period the firm invests `I = gamma * Z`,
    accumulates capital `K = K + I`, produces `Y = phi * K`, borrows `B = max(K - A, 0)`, sells at the price
    `P = pbar + 2 U(0, 1)`, earns `Z = P * Y - r * K`, and adds the profit to its net worth `A`. A firm whose net worth
    turns negative is replaced by an entrant with `A = K = 1` and `Z = 0`.

    Args:
        label:
            The label of the agent or group.

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

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import InvestingFirmElement
        >>> InvestingFirmElement('firms', gamma=1.2, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        *,
        A: object = 1.0,
        K: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        Z: object = 0.0,
        B: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        Y: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        P: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        gamma: Annotated[object, RealSet.open(0, oo)] = 1.1,
        phi: Annotated[object, RealSet.open(0, oo)] = 0.1,
        r: Annotated[object, RealSet.open(0, oo)] = 0.1,
        pbar: Annotated[object, RealSet.closed_open(0, oo)] = 0.01,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
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


class InvestmentInteraction(Interaction):
    """Investment rule of the firms of the toy business-cycle system.

    Every period a firm invests `I = gamma * Z`, accumulates capital `K = K + I`, produces `Y = phi * K`, borrows
    `B = max(K - A, 0)`, sells at the random price `P = pbar + 2 U(0, 1)`, earns `Z = P * Y - r * K`, and adds the
    profit to its net worth `A`. A firm whose net worth turns negative is replaced by an entrant with `A = K = 1` and
    `Z = 0`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of firms.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import InvestmentInteraction
        >>> InvestmentInteraction('investment', target='firms').target
        'firms'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the period of every firm as its rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self.invest)]

    def invest(self, agent: InvestingFirmView, system: SystemView) -> None:
        """Run one period of a firm, and replace it by an entrant when its net worth turns negative.

        Args:
            agent:
                The firm.

            system:
                The system view of the rule.
        """
        agent.K = agent.K + agent.gamma * agent.Z
        agent.Y = agent.phi * agent.K
        agent.B = max(agent.K - agent.A, 0.0)
        agent.P = agent.pbar + 2.0 * float(system.rng.random())
        agent.Z = agent.P * agent.Y - agent.r * agent.K
        agent.A = agent.A + agent.Z
        if agent.A < 0:
            agent.Z = 0.0
            agent.K = 1.0
            agent.A = 1.0


class ToyFirmInvestmentSystem(BaseSystem):
    """Heterogeneous firms that invest out of past profits and sell at random prices.

    Firms enter and exit. The firms start with net worth and capital 1 and a profit drawn from `pbar + 2 U(0, 1)`, as in
    the toy system of Bargigli et al. in Caiani et al. (2016, sect. 2.1). Aggregate production fluctuates as bankrupt
    firms are replaced. The system has the group `firms` and the aggregate production `production`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import ToyFirmInvestmentSystem
        >>> ToyFirmInvestmentSystem(20).prepare().system_.elements[0].size
        20
    """

    field = 'economics/macroeconomics'
    topics = ('business cycle', 'agents', 'firms')
    references = ('Caiani et al. 2016', 'Delli Gatti et al. 2010')
    until = 100
    observables = (Observable('production', group='firms', reduce='sum', variable='Y'),)

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_firms = n_firms
        self.gamma = gamma
        self.phi = phi
        self.r = r
        self.pbar = pbar
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        profits = self.pbar + 2.0 * np.random.default_rng(self.random_state).random(self.n_firms)
        firms = InvestingFirmElement(
            'firms',
            Z=profits,
            gamma=self.gamma,
            phi=self.phi,
            r=self.r,
            pbar=self.pbar,
            size=self.n_firms,
        )
        self.system_ = System(
            self.label,
            elements=[firms],
            interactions=[InvestmentInteraction('investment', target='firms')],
            solver=MesaSolver(activation='sequential'),
        )
