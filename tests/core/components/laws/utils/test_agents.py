"""Test the `_agents` module of `skcomplex.core.components.laws.utils`."""

from types import SimpleNamespace
from typing import Annotated

import pytest

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element, Quantity
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import transfer
from skcomplex.core.engine.solvers import AgentView
from skcomplex.core.errors import ConservationError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role


def _balance(agent):
    """Return assets minus liabilities minus equity."""
    return agent.assets - agent.liabilities - agent.equity


def _lose(agent, system):
    """Lower the assets of a bank by 1, which breaks its balance sheet."""
    agent.assets -= 1.0


class LenderElement(Element):
    """Lenders holding claims on firms, positive loans."""

    def __init__(
        self,
        label: str,
        *,
        loans: Annotated[float, Quantity('loans')] = 0.0,
        money: Annotated[float, Quantity('money')] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.loans = loans
        self.money = money


class BorrowerElement(Element):
    """Firms that owe their loans, a negative holding."""

    def __init__(
        self,
        label: str,
        *,
        loans: Annotated[float, Quantity('loans')] = 0.0,
        money: Annotated[float, Quantity('money')] = 0.0,
        settles: float = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.loans = loans
        self.money = money
        self.settles = settles


def _default(agent, system):
    """Remove a firm when its money runs out, after writing off its debt when it settles."""
    if agent.money > 0:
        return
    if agent.settles:
        lender = system.get_agents('lenders', AgentView)[0]
        transfer(agent, lender, agent.loans, 'loans')
    agent.remove()


def _credit_system(settles):
    """Return a lender and two bankrupt borrowers that each owe 5."""
    return System(
        elements=[
            LenderElement('lenders', loans=10.0, money=4.0, size=1),
            BorrowerElement('firms', loans=-5.0, money=0.0, settles=float(settles), size=2),
        ],
        interactions=[
            Interaction(
                'bankruptcy',
                roles={'target': 'firms', 'lenders': 'lenders'},
                laws=lambda s: [UpdateLaw(s.target, _default)],
            ),
        ],
    )


class PaymentsInteraction(Interaction):
    """Payments between holders, which create money when asked."""

    def __init__(self, label: str, *, holders: Role, creates: float = 0.0) -> None:
        super().__init__(label)
        self.holders = holders
        self.creates = creates

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rule that moves money."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context, self.move)]

    def move(self, system):
        """Move one unit from the first holder to the second, and create money when the payments create it."""
        first, second = system.get_agents('holders', AgentView)[:2]
        transfer(first, second, 1.0, 'money')
        first.money += system.get_parameter('creates')


def test_invariant_of_agents_names_the_observable_and_the_change():
    """Test that lowering assets alone breaks the balance sheet that an observable of the agents sums."""
    balance = Observable('balance', _balance, group='banks', reduce='sum')
    banks = Element('banks', values={'assets': 10.0, 'liabilities': 8.0, 'equity': 2.0}, size=1)
    losses = Interaction('losses', roles={'target': 'banks'}, laws=lambda s: [UpdateLaw(s.target, _lose)])
    system = System(elements=[banks], interactions=[losses]).prepare()
    with pytest.raises(ConservationError, match='invariant balance changed from 0 to -1 at frame 1'):
        system.simulate(until=2, observables=[balance], invariants={'balance': 1e-9})


def test_bankrupt_firm_writes_off_its_debt_against_its_lender():
    """Test that removed firms settle their debt with transfers and total loans stay zero."""
    trajectory = _credit_system(settles=True).prepare().simulate(until=1)
    assert trajectory.get_agents('lenders', 1).values['loans'].tolist() == [0.0]
    assert len(trajectory.get_agents('firms', 1).values['loans']) == 0


def test_removing_an_unsettled_agent_destroys_its_holdings():
    """Test that removing a firm that still owes raises a conservation error."""
    with pytest.raises(ConservationError, match='Settle its holdings'):
        _credit_system(settles=False).prepare().simulate(until=1)


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
    system = System(
        elements=[LenderElement('holders', money=5.0, size=3)],
        interactions=[PaymentsInteraction('payments', holders='holders', creates=creates)],
    ).prepare()
    total = Observable('total_money', group='holders', variable='money', reduce='sum')
    settings = {'until': 3, 'observables': [total], 'invariants': {'total_money': 1e-9}}
    if violated:
        with pytest.raises(ConservationError, match='invariant total_money'):
            system.simulate(**settings)
        return
    trajectory = system.simulate(**settings)
    assert trajectory.get_agents('holders', 3).values['money'].sum() == pytest.approx(15.0)
