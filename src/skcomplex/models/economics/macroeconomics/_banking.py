"""Macroeconomics models: fractional-reserve banking, where loans redeposited at banks multiply the base money."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import TransferInteraction
from ....core.laws import Sink
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....specs import ObservableSpec, PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_model
from ._depositor import DepositorElement
from ._policy_switch import PolicySwitchInteraction
from ._reserve_bank import ReserveBankElement


@register_model(
    field='economics/macroeconomics',
    tags=('money', 'banks', 'agents', 'institution'),
    references=('Hamill and Gilbert 2016 ch. 9',),
    defaults={'until': 300},
    observables=lambda system: [
        ObservableSpec('broad_money', group='banks', variable='deposits', reduce='sum'),
        ObservableSpec('reserves', group='banks', variable='reserves', reduce='sum'),
        ObservableSpec('loans', group='banks', variable='loans', reduce='sum'),
    ],
)
def fractional_reserve_banking(
    customers: int = 100,
    banks: int = 5,
    *,
    base_money: float = 1000.0,
    ratio: float = 0.1,
    new_ratio: float | None = None,
    change_at: float = 150.0,
    spending: float = 0.1,
    repayment: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return customers and banks whose lending of excess reserves multiplies the base money by `1 / ratio`.

    The base money starts as equal deposits of the customers, held as reserves by their banks, picked at random. Every
    step the customers pay a share of their deposits to random customers and repay a share of their loans from their
    deposits, and every bank lends its reserves above the
    required share of its deposits; the loans are spent and redeposited at random banks. The reserves stay equal to the
    base money, so the broad money, the sum of deposits, settles near `base_money / ratio`. A regulator changes the
    reserve ratio to `new_ratio` at `change_at`, and the broad money moves to the new multiplier.

    Args:
        customers:
            The number of customers.

        banks:
            The number of banks.

        base_money:
            The base money, deposited by the customers in equal shares.

        ratio:
            The reserve ratio.

        new_ratio:
            The reserve ratio from `change_at` on, or `None` to keep `ratio`.

        change_at:
            The time the reserve ratio changes.

        spending:
            The share of its deposit a customer pays to a random customer every step.

        repayment:
            The share of its loan a customer repays every step.

        random_state:
            The seed or generator of the customers' banks.

    Returns:
        system:
            The model, with the groups `customers` and `banks`, the institution `regulator`, and the observables
            `broad_money`, `reserves`, and `loans`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import fractional_reserve_banking
        >>> fractional_reserve_banking(new_ratio=0.2).interactions[0].new_ratio
        0.2
    """
    rng = np.random.default_rng(random_state)
    owners = rng.integers(banks, size=customers)
    deposit = base_money / customers
    holdings = np.bincount(owners, minlength=banks) * deposit
    depositors = GroupElement(
        'customers',
        element=DepositorElement,
        size=customers,
        values={'bank': owners, 'deposit': deposit, 'spending': spending, 'repayment': repayment},
    )
    payments = TransferInteraction(
        'payments',
        source='customers',
        target='customers',
        state='deposit',
        amount=DepositorElement.get_payment,
        pairing='random',
        legs=(
            ('bank.deposits', None, DepositorElement.get_payment),
            ('bank.reserves', None, DepositorElement.get_payment),
        ),
        form='agent',
        phase='pay',
    )
    repayments = TransferInteraction(
        'repayments',
        source='customers',
        target='customers',
        state='deposit',
        into=Sink(),
        amount=DepositorElement.get_repayment,
        legs=tuple(
            (state, Sink(), DepositorElement.get_repayment) for state in ('loan', 'bank.deposits', 'bank.loans')
        ),
        guard=lambda customer, same, model: DepositorElement.get_repayment(customer, same, model) > 0,
        form='agent',
        phase='pay',
    )
    lenders = GroupElement(
        'banks',
        element=ReserveBankElement,
        size=banks,
        values={'reserves': holdings, 'deposits': holdings, 'reserve_ratio': ratio},
    )
    regulator = PolicySwitchInteraction(
        'regulator',
        phase='regulate',
        ratio=ratio,
        new_ratio=ratio if new_ratio is None else new_ratio,
        change_at=change_at,
    )
    return System(
        elements=[depositors, lenders],
        interactions=[regulator, payments, repayments],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('regulate'), PhaseSpec('pay'), PhaseSpec('lend', repeat=10)]),
        ),
    )
