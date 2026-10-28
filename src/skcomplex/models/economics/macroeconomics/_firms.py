"""Macroeconomics models: heterogeneous firms that invest out of their profits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ..._catalog import register_model


class InvestingFirmElement(Element):
    """Firm of the toy business-cycle model, which invests a share of its last profit and sells at a random price.

    Every period the firm invests `I = gamma * Z`, accumulates capital `K = K + I`, produces `Y = phi * K`, borrows
    `B = max(K - A, 0)`, sells at the price `P = pbar + 2 U(0, 1)`, earns `Z = P * Y - r * K`, and adds the profit to
    its net worth `A`. A firm whose net worth turns negative is replaced by an entrant with `A = K = 1` and `Z = 0`.

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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of one period.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.trade)]

    def trade(self, agent: AgentView, model: ModelView) -> None:
        """Invest, produce, borrow, sell, and update the net worth, with replacement of a bankrupt firm.

        Args:
            agent:
                The firm.

            model:
                The model.
        """
        agent.K = agent.K + agent.gamma * agent.Z
        agent.Y = agent.phi * agent.K
        agent.B = max(agent.K - agent.A, 0.0)
        agent.P = agent.pbar + 2.0 * float(model.rng.random())
        agent.Z = agent.P * agent.Y - agent.r * agent.K
        agent.A = agent.A + agent.Z
        if agent.A < 0:
            agent.Z, agent.K, agent.A = 0.0, 1.0, 1.0


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
        >>> toy_firm_investment(20).elements[0].check().prepare().size_
        20
    """
    profits = pbar + 2.0 * np.random.default_rng(random_state).random(n_firms)
    firms = GroupElement(
        'firms',
        element=InvestingFirmElement,
        size=n_firms,
        values={'Z': profits, 'gamma': gamma, 'phi': phi, 'r': r, 'pbar': pbar},
    )
    return System(elements=[firms])
