"""Macroeconomics systems: credit networks of firms that borrow from banks.

The firms switch to cheaper lenders.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, Member
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.matching import MatchingInteraction, MatchSpec, PartnerSwitchSpec


@runtime_checkable
class FirmView(AgentViewProtocol, Protocol):
    """Firm as the rules see it.

    It reads its `net_worth`, `demand`, `loan`, `rate`, `partner`, `gross_return`, `injected`, and `defaulted`, and the
    parameters `leverage`, `mean_return`, `volatility`, and `entry`.
    """

    net_worth: float
    demand: float
    loan: float
    rate: float
    partner: float
    gross_return: float
    injected: float
    defaulted: float
    leverage: float
    mean_return: float
    volatility: float
    entry: float


class FirmElement(Element):
    """Firm that borrows in proportion to its net worth and invests.

    It invests at a random return, and defaults on negative worth. Every step the firm asks for `leverage` times its net
    worth from its bank, invests its net worth and the loan at the return `mean_return + volatility * z`, and pays the
    interest. When its net worth becomes negative it repays what it can, the bank writes off the rest, and a new firm
    with the net worth `entry` replaces it.

    Args:
        label:
            The label of the agent or group.

        net_worth:
            The initial net worth.

        demand:
            The loan still wanted in the current step.

        bid:
            The highest interest rate the firm pays.

        loan:
            The loan of the current step.

        rate:
            The interest rate of the loan, zero without a loan.

        partner:
            The bank of the firm among the banks, `-1` before its first loan.

        gross_return:
            The return on the capital of the last step.

        injected:
            The net worth of an entrant that replaced the firm in the last step, zero otherwise.

        defaulted:
            Whether the firm defaulted in the last step.

        leverage:
            The ratio of the loan wanted to the net worth.

        mean_return:
            The mean return on capital.

        volatility:
            The standard deviation of the return on capital.

        entry:
            The net worth of an entrant.

        size:
            The number of firms, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import FirmElement
        >>> FirmElement('firms', net_worth=5.0, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        *,
        net_worth: Annotated[object, RealSet.closed_open(0, oo)] = 10.0,
        demand: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        bid: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        loan: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        rate: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        partner: Annotated[object, Member('banks')] = -1,
        gross_return: object = 0.0,
        injected: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        defaulted: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0,
        leverage: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        mean_return: object = 0.03,
        volatility: Annotated[object, RealSet.closed_open(0, oo)] = 0.3,
        entry: Annotated[object, RealSet.open(0, oo)] = 10.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.net_worth = net_worth
        self.demand = demand
        self.bid = bid
        self.loan = loan
        self.rate = rate
        self.partner = partner
        self.gross_return = gross_return
        self.injected = injected
        self.defaulted = defaulted
        self.leverage = leverage
        self.mean_return = mean_return
        self.volatility = volatility
        self.entry = entry


@runtime_checkable
class LenderView(AgentViewProtocol, Protocol):
    """Lender as the rules see it.

    It holds its `equity`, `ask`, `supply`, `lent`, `losses`, `alpha`, and `capital_ratio`.
    """

    equity: float
    ask: float
    supply: float
    lent: float
    losses: float
    alpha: float
    capital_ratio: float


class LenderElement(Element):
    """Bank that lends up to a multiple of its equity.

    Its rate falls with its equity. The bank asks the rate `alpha * equity ** -alpha` of Delli Gatti et al. (2010), so
    better capitalised banks lend cheaper, and it offers loans up to `equity / capital_ratio`. The interest and
    the defaults of the firms reach the bank they name in their value `partner`, by the rules of
    `BalanceSheetsInteraction`.

    Args:
        label:
            The label of the agent or group.

        equity:
            The initial equity.

        ask:
            The interest rate the bank asks.

        supply:
            The loans the bank can still grant in the current step.

        lent:
            The loans granted in the current step.

        losses:
            The loans written off in the current step.

        alpha:
            The elasticity of the interest rate to the equity.

        capital_ratio:
            The smallest ratio of equity to loans.

        size:
            The number of banks, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import LenderElement
        >>> LenderElement('banks', equity=100.0, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        *,
        equity: object = 100.0,
        ask: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        supply: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        lent: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        losses: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        alpha: Annotated[object, RealSet.open(0, oo)] = 0.1,
        capital_ratio: Annotated[object, RealSet.open(0, oo)] = 0.1,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.equity = equity
        self.ask = ask
        self.supply = supply
        self.lent = lent
        self.losses = losses
        self.alpha = alpha
        self.capital_ratio = capital_ratio


class BalanceSheetsInteraction(Interaction):
    """Plan and settlement of the balance sheets of the firms and the banks of the credit network.

    In the plan of a step every firm requests a loan of `leverage` times its net worth, and every bank asks the rate
    `alpha * equity ** -alpha` of Delli Gatti et al. (2010), so better capitalised banks lend cheaper, and offers
    loans up to `equity / capital_ratio`. In the settlement every firm earns a normal rate of return, at least -1, on
    its net worth and its loan, every bank earns the interest of its borrowers less what the defaulting ones cannot
    repay, and a firm whose net worth after interest turns negative defaults and takes the net worth of an entrant.

    Args:
        label:
            The label of the interaction.

        firms:
            The path of the group of firms.

        banks:
            The path of the group of banks.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import BalanceSheetsInteraction
        >>> BalanceSheetsInteraction('sheets', firms='firms', banks='banks').banks
        'banks'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, firms: Role, banks: Role) -> None:
        super().__init__(label)
        self.firms = firms
        self.banks = banks

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the plan and the settlement of every step as the rules of the system, in the phases of their names.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [
            UpdateLaw(context, self.plan, phase='plan'),
            UpdateLaw(context, self.settle, phase='settle'),
        ]

    @staticmethod
    def _get_worth(firm: FirmView) -> float:
        """Return the net worth of a firm after the return and the interest, negative for a firm that defaults."""
        return firm.net_worth + firm.gross_return - firm.loan * firm.rate

    def plan(self, system: SystemView) -> None:
        """Set the loan request of every firm, and then the rate and the lending capacity of every bank.

        Args:
            system:
                The system view of the plan.
        """
        for firm in cast(list[FirmView], system.get_agents('firms')):
            firm.demand = firm.leverage * firm.net_worth
            firm.loan = firm.rate = firm.defaulted = firm.injected = 0.0
        for bank in cast(list[LenderView], system.get_agents('banks')):
            bank.ask = bank.alpha * max(bank.equity, 1e-6) ** -bank.alpha
            bank.supply = max(bank.equity, 0.0) / bank.capital_ratio
            bank.lent = bank.losses = 0.0

    def settle(self, system: SystemView) -> None:
        """Draw the return of every firm, settle the banks, and replace every defaulting firm by an entrant.

        Args:
            system:
                The system view of the settlement.
        """
        firms = cast(list[FirmView], system.get_agents('firms'))
        for firm in firms:
            draw = firm.mean_return + firm.volatility * system.rng.standard_normal()
            firm.gross_return = (firm.net_worth + firm.loan) * max(draw, -1.0)
        borrowers: dict[int, list[FirmView]] = {}
        for firm in firms:
            bank = firm.get_referenced('partner') if firm.loan > 0 else None
            if bank is not None:
                borrowers.setdefault(bank.id, []).append(firm)
        banks = cast(list[LenderView], system.get_agents('banks'))
        for bank in banks:
            owed = borrowers.get(bank.id, [])
            bank.equity += sum(firm.loan * firm.rate + min(self._get_worth(firm), 0.0) for firm in owed)
        for bank in banks:
            bank.losses += sum(max(-self._get_worth(firm), 0.0) for firm in borrowers.get(bank.id, []))
        for firm in firms:
            worth = self._get_worth(firm)
            if worth < 0:
                firm.injected, firm.defaulted, firm.partner, firm.net_worth = firm.entry, 1.0, -1.0, firm.entry
            else:
                firm.net_worth = worth


class CreditMarketInteraction(MatchingInteraction[SystemView]):
    """Credit market that also reports the spread of the rates firms pay.

    The spread is over the cheapest bank's rate. The firms request loans and the banks provide them. The spread is the
    mean of `r_i / r_min - 1` over the borrowing firms, the dispersion of rates that a market with costly search leaves
    and that switching to cheaper lenders erodes.

    Args:
        label:
            The label of the market.

        requesters:
            The path of the group of firms.

        providers:
            The path of the group of banks.

        rule:
            The spec of the algorithm that matches firms with banks, such as `PartnerSwitchSpec()`.

        reserve:
            The state with the rate of both sides, or a pair of the firms' and the banks' states.

        quantity:
            The state with the loan of both sides, or a pair of the firms' and the banks' states.

        cap:
            The state of a firm with what it can spend.

        transfers:
            The states every loan moves, each with the sign `-1` for a payment of the firm or `1` for a delivery to
            it.

        phase:
            The phase the market clears in, or `None` for the default phase.

        on_match:
            A function of every loan and the system view, run after the transfers, or `None`.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import CreditMarketInteraction
        >>> CreditMarketInteraction('credit', requesters='firms', providers='banks').reports[-1]
        'spread'
    """

    reports: ClassVar[tuple[str, ...]] = (*MatchingInteraction.reports, 'spread')

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, system: SystemView) -> None:
        """Report the price, the volume, the unmatched quantity, and the spread.

        The spread is that of the rates over the cheapest rate.

        Args:
            matches:
                The loans of the clearing.

            unmatched:
                The quantity of both sides left after the clearing.

            system:
                The system view of the market.
        """
        super()._report_matches(matches, unmatched, system)
        cheapest = min(
            (bank.ask for bank in cast(list[LenderView], system.get_agents('providers')) if bank.ask > 0),
            default=0.0,
        )
        paid = [firm.rate for firm in cast(list[FirmView], system.get_agents('requesters')) if firm.loan > 0]
        spread = float(np.mean(paid) / cheapest - 1) if paid and cheapest > 0 else float('nan')
        system.report('spread', spread)


def _get_herfindahl(values: object) -> float:
    """Return the Herfindahl index of the banks' shares of the loans."""
    lent = np.asarray(values, dtype=float)
    total = lent.sum()
    return float(np.sum((lent / total) ** 2)) if total > 0 else float('nan')


def _get_largest_share(values: object) -> float:
    """Return the share of the loans of the largest lender."""
    lent = np.asarray(values, dtype=float)
    total = lent.sum()
    return float(lent.max() / total) if total > 0 else float('nan')


class CreditNetworksSystem(BaseSystem):
    """A credit market where firms sample banks and switch to cheaper lenders.

    Defaults hit the lenders. Every step the banks quote rates that fall with their equity, every firm samples `sample`
    banks and switches from its lender to the cheapest of them with probability
    `1 - exp(intensity * (r_new - r_old) / r_new)`, borrows, and invests at a random return. Interest and write-offs of
    defaulted firms move net worth between firms and banks, so the total net worth changes only by the firms' returns
    and the entrants' net worth. A higher intensity of switching concentrates the loans in the cheapest banks and
    narrows the spread of the rates that firms pay.

    The system has the groups `firms` and `banks`, the market `credit` with its reporter `credit.spread`, and the
    observables `herfindahl`, `largest_share`, `bank_equity`, `firm_worth`, `returns`, `injected`, `defaults`, and
    `bad_debt`.

    Args:
        firms:
            The number of firms.

        banks:
            The number of banks.

        sample:
            The number of banks every firm samples.

        intensity:
            The intensity of switching to a cheaper bank.

        leverage:
            The ratio of the loan a firm wants to its net worth.

        mean_return:
            The mean return on capital.

        volatility:
            The standard deviation of the return on capital.

        net_worth:
            The initial net worth of every firm and of every entrant.

        equity:
            The mean initial equity of the banks, drawn uniformly within 50% of it.

        alpha:
            The elasticity of the interest rate to the equity.

        capital_ratio:
            The smallest ratio of a bank's equity to its loans.

        random_state:
            The seed or generator of the initial equities.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import CreditNetworksSystem
        >>> [element.label for element in CreditNetworksSystem().prepare().system_.elements]
        ['firms', 'banks']
    """

    field = 'economics/macroeconomics'
    topics = ('credit', 'banks', 'agents', 'market', 'network')
    references = ('Caiani et al. 2016 ch. 2', 'Delli Gatti et al. 2010')
    until = 100
    observables = (
        Observable('herfindahl', group='banks', variable='lent', reduce=_get_herfindahl),
        Observable('largest_share', group='banks', variable='lent', reduce=_get_largest_share),
        Observable('bank_equity', group='banks', variable='equity', reduce='sum'),
        Observable('firm_worth', group='firms', variable='net_worth', reduce='sum'),
        Observable('returns', group='firms', variable='gross_return', reduce='sum'),
        Observable('injected', group='firms', variable='injected', reduce='sum'),
        Observable('defaults', group='firms', variable='defaulted', reduce='sum'),
        Observable('bad_debt', group='banks', variable='losses', reduce='sum'),
    )

    def __init__(
        self,
        firms: int = 100,
        banks: int = 10,
        *,
        sample: int = 3,
        intensity: float = 4.0,
        leverage: float = 1.0,
        mean_return: float = 0.03,
        volatility: float = 0.3,
        net_worth: float = 10.0,
        equity: float = 100.0,
        alpha: float = 0.1,
        capital_ratio: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.firms = firms
        self.banks = banks
        self.sample = sample
        self.intensity = intensity
        self.leverage = leverage
        self.mean_return = mean_return
        self.volatility = volatility
        self.net_worth = net_worth
        self.equity = equity
        self.alpha = alpha
        self.capital_ratio = capital_ratio
        self.random_state = random_state

    @staticmethod
    def _lend(trade: MatchSpec, system: SystemView) -> None:
        """Record a loan on the books of the firm and of the bank."""
        del system
        firm, bank = trade.requester, trade.provider
        if not isinstance(firm, FirmView) or not isinstance(bank, LenderView):
            error_msg = 'A loan is a trade of a firm with a bank.'
            raise TypeError(error_msg)
        firm.loan += trade.quantity
        firm.rate = trade.value
        bank.lent += trade.quantity

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        lenders = LenderElement(
            'banks',
            equity=self.equity * rng.uniform(0.5, 1.5, self.banks),
            alpha=self.alpha,
            capital_ratio=self.capital_ratio,
            size=self.banks,
        )
        borrowers = FirmElement(
            'firms',
            net_worth=self.net_worth,
            leverage=self.leverage,
            mean_return=self.mean_return,
            volatility=self.volatility,
            entry=self.net_worth,
            size=self.firms,
        )
        market = CreditMarketInteraction(
            'credit',
            requesters='firms',
            providers='banks',
            rule=PartnerSwitchSpec(sample=self.sample, intensity=self.intensity),
            reserve=('bid', 'ask'),
            quantity=('demand', 'supply'),
            phase='credit',
            on_match=self._lend,
        )
        self.system_ = System(
            self.label,
            elements=[borrowers, lenders],
            interactions=[BalanceSheetsInteraction('sheets', firms='firms', banks='banks'), market],
            solver=MesaSolver(phases=[PhaseSpec('plan'), PhaseSpec('credit'), PhaseSpec('settle')]),
        )
