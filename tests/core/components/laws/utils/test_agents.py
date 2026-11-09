"""Test the `_agents` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.laws.utils import transfer
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import AgentView
from skcomplex.core.errors import ConservationError, ValidationError
from skcomplex.core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


def _balance(agent):
    """Return assets minus liabilities minus equity."""
    return agent.assets - agent.liabilities - agent.equity


class BankElement(Element):
    """Bank whose rule lowers its assets alone, which breaks its balance sheet."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        assets: Annotated[float, StateSpec()] = 10.0,
        liabilities: Annotated[float, StateSpec()] = 8.0,
        equity: Annotated[float, StateSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.assets = assets
        self.liabilities = liabilities
        self.equity = equity

    def _get_laws(self, s):
        """Return the rule that loses assets."""
        return [RuleLaw(self.lose)]

    def lose(self, agent, system):
        """Lower the assets by 1."""
        agent.assets -= 1.0


class LenderElement(Element):
    """Lender holding claims on firms, positive loans."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        loans: Annotated[float, StateSpec(quantity='loans')] = 0.0,
        money: Annotated[float, StateSpec(quantity='money')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.loans = loans
        self.money = money


class BorrowerElement(Element):
    """Firm that owes its loans, a negative holding, and goes bankrupt when its money runs out."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        loans: Annotated[float, StateSpec(quantity='loans')] = 0.0,
        money: Annotated[float, StateSpec(quantity='money')] = 0.0,
        settles: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.loans = loans
        self.money = money
        self.settles = settles

    def _get_laws(self, s):
        """Return the rule that removes bankrupt firms."""
        return [RuleLaw(self.default)]

    def default(self, agent, system):
        """Remove the firm when its money runs out, after writing off its debt when it settles."""
        if agent.money > 0:
            return
        if agent.settles:
            lender = system.agents('lenders', AgentView)[0]
            transfer(agent, lender, agent.loans, 'loans')
        agent.remove()


def _credit_system(settles):
    """Return a lender and two bankrupt borrowers that each owe 5."""
    return CompositeSystem(
        elements=[
            GroupElement('lenders', element=LenderElement, size=1, values={'loans': 10.0, 'money': 4.0}),
            GroupElement(
                'firms',
                element=BorrowerElement,
                size=2,
                values={'loans': -5.0, 'money': 0.0, 'settles': float(settles)},
                variable=True,
            ),
        ],
    )


class PaymentsInteraction(Interaction):
    """Payments between holders, which create money when asked."""

    def __init__(
        self,
        label: str,
        *,
        holders: Annotated[str, RoleSpec()],
        creates: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.holders = holders
        self.creates = creates

    def _get_laws(self, s):
        """Return the rule that moves money."""
        return [RuleLaw(self.move)]

    def move(self, system):
        """Move one unit from the first holder to the second, and create money when the payments create it."""
        first, second = system.agents('holders', AgentView)[:2]
        transfer(first, second, 1.0, 'money')
        first.money += system.parameter('creates')


def test_invariant_of_agents_names_the_observable_and_the_change():
    """Test that lowering assets alone breaks the balance sheet that an observable of the agents sums."""
    balance = ObservableSpec('balance', _balance, group='banks', reduce='sum')
    with pytest.raises(ConservationError, match='invariant balance changed from 0 to -1 at frame 1'):
        simulate(
            CompositeSystem(elements=[GroupElement('banks', element=BankElement, size=1)]),
            until=2,
            observables=[balance],
            invariants={'balance': 1e-9},
        )


def test_bankrupt_firm_writes_off_its_debt_against_its_lender():
    """Test that removed firms settle their debt with transfers and total loans stay zero."""
    trajectory = simulate(_credit_system(settles=True), until=1)
    assert trajectory.agents('lenders', 1).values['loans'].tolist() == [0.0]
    assert len(trajectory.agents('firms', 1).values['loans']) == 0


def test_removing_an_unsettled_agent_destroys_its_holdings():
    """Test that removing a firm that still owes raises a conservation error."""
    with pytest.raises(ConservationError, match='Settle its holdings'):
        simulate(_credit_system(settles=False), until=1)


def test_transfer_moves_between_states_and_checks_names():
    """Test that a transfer can move between differently named states and rejects missing ones."""
    lender, borrower = LenderElement('l'), BorrowerElement('b')
    transfer(lender, borrower, 3.0, 'money')
    assert (lender.money, borrower.money) == (-3.0, 3.0)
    transfer(borrower, lender, 1.0, 'money', 'loans')
    assert (borrower.money, lender.loans) == (2.0, 1.0)
    with pytest.raises(ValidationError, match='cash'):
        transfer(lender, borrower, 1.0, 'cash')


@pytest.mark.parametrize(('creates', 'violated'), [(0.0, False), (0.5, True)])
def test_interaction_invariant_checks_the_total_of_its_agents(creates, violated):
    """Test that the total money of the agents holds under transfers and the run names it when money appears."""
    system = CompositeSystem(
        elements=[GroupElement('holders', element=LenderElement, size=3, values={'money': 5.0}, variable=True)],
        interactions=[PaymentsInteraction('payments', holders='holders', creates=creates)],
    )
    total = ObservableSpec('total_money', group='holders', variable='money', reduce='sum')
    settings = {'until': 3, 'observables': [total], 'invariants': {'total_money': 1e-9}}
    if violated:
        with pytest.raises(ConservationError, match='invariant total_money'):
            simulate(system, **settings)
        return
    trajectory = simulate(system, **settings)
    assert trajectory.agents('holders', 3).values['money'].sum() == pytest.approx(15.0)
