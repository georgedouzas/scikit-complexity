"""Econophysics systems: kinetic wealth exchange.

The exchange has a saving propensity or a debt limit.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import PartnerSpec, UpdateLaw
from ....core.components.laws.utils import pairs, transfer
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class SavingTraderView(AgentViewProtocol, Protocol):
    """Trader as the trading rules see it.

    It holds its `wealth`, negative down to the debt limit.
    """

    wealth: float


type Pairs = list[tuple[SavingTraderView, SavingTraderView]]


class SavingTraderElement(Element):
    """Agent of the kinetic wealth exchange systems.

    Its wealth changes only in pairwise trades.

    Args:
        label:
            The label of the agent or group.

        wealth:
            The initial wealth, negative down to the debt limit.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import SavingTraderElement
        >>> group = SavingTraderElement('traders', wealth=2.0, size=10)
        >>> group.label, group.size
        ('traders', 10)
    """

    def __init__(self, label: str, *, wealth: float = 1.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.wealth = wealth


class DebtExchangeInteraction(Interaction):
    """Random pairwise trades of Dragulescu and Yakovenko of a fixed amount.

    Debt is allowed down to a limit. In every trade an agent pays a fixed amount to another agent drawn at random,
    unless its wealth would fall below minus the debt limit. Every step, every agent pays once, in random order. The
    stationary wealth distribution is the exponential `P(w) = exp(-(w + d) / T) / T` shifted to the debt limit `d`, with
    temperature `T = <w> + d`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of traders.

        debt:
            The debt limit `d`, the largest debt an agent can have.

        amount:
            The amount paid in every trade.

        phase:
            The phase the trades run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import DebtExchangeInteraction
        >>> DebtExchangeInteraction('trades', target='traders', debt=5.0).debt
        5.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        debt: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        amount: Annotated[float, RealSet.open(0, oo)] = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.debt = debt
        self.amount = amount
        self.phase = phase

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the trades of every step as the rule of the system.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context, self.trade, phase=self.phase)]

    def trade(self, system: SystemViewProtocol) -> None:
        """Let every trader pay the amount to a random trader, in random order, while it stays above the limit.

        Args:
            system:
                The system view of the trades.
        """
        traders = system.get_agents('target')
        found = cast(Pairs, list(pairs(traders, traders, system.rng, PartnerSpec('all', 'uniform'))))
        for index in system.rng.permutation(len(found)):
            payer, payee = found[int(index)]
            if payer.alive and payee.alive and payer.wealth - self.amount >= -self.debt:
                transfer(payer, payee, self.amount, 'wealth')


class SavingExchangeInteraction(Interaction):
    """Random pairwise trades of Chakraborti and Chakrabarti.

    Every agent saves a fraction of its wealth. A trade between agents `i` and `j` keeps total wealth and gives
    `w_i' = lambda w_i + eps (1 - lambda) (w_i + w_j)` and `w_j' = lambda w_j + (1 - eps) (1 - lambda) (w_i + w_j)` with
    `eps` uniform on `[0, 1]`, so `i` transfers `(1 - lambda) (w_i - eps (w_i + w_j))` to `j`, an amount of either sign.
    Every step, every agent trades once with another agent drawn at random, in random order. Without saving the
    stationary wealth distribution is exponential, and with saving propensity `lambda` it is close to a Gamma
    distribution of shape `n = 1 + 3 lambda / (1 - lambda)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of traders.

        saving:
            The saving propensity `lambda`, the fraction of its wealth that an agent keeps out of a trade.

        phase:
            The phase the trades run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import SavingExchangeInteraction
        >>> SavingExchangeInteraction('trades', target='traders').saving
        0.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        saving: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.saving = saving
        self.phase = phase

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the trades of every step as the rule of the system.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context, self.trade, phase=self.phase)]

    def trade(self, system: SystemViewProtocol) -> None:
        """Let every trader trade a random share of the pooled wealth with a random trader, in random order.

        Args:
            system:
                The system view of the trades.
        """
        traders = system.get_agents('target')
        found = cast(Pairs, list(pairs(traders, traders, system.rng, PartnerSpec('all', 'uniform'))))
        for index in system.rng.permutation(len(found)):
            first, second = found[int(index)]
            if first.alive and second.alive:
                transfer(first, second, self._get_amount(first, second, system), 'wealth')

    def _get_amount(self, first: SavingTraderView, second: SavingTraderView, system: SystemViewProtocol) -> float:
        """Return the wealth the first trader transfers to the second, of either sign."""
        share = float(system.rng.random())
        return (1 - self.saving) * (first.wealth - share * (first.wealth + second.wealth))


class KineticWealthExchangeSystem(BaseSystem):
    """A kinetic wealth exchange system with a saving propensity or a debt limit.

    Without a debt limit, agents trade random shares of their pooled wealth after saving a fraction `saving` of it, as
    in Chakraborti and Chakrabarti. With a debt limit, they pay a fixed amount in every trade and may go into debt down
    to the limit, as in Dragulescu and Yakovenko. The system has the group `traders`, the interaction `trades`, and the
    variance of wealth `variance`.

    Args:
        n_agents:
            The number of agents.

        saving:
            The saving propensity `lambda`, between 0 and 1, of the trades without a debt limit.

        debt:
            The debt limit of the fixed-amount trades, or `None` for trades of random shares with saving.

        wealth:
            The initial wealth of every agent, the mean wealth.

        amount:
            The amount paid in every fixed-amount trade.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValueError:
            Both a saving propensity and a debt limit are given, or the saving propensity is not below 1.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import KineticWealthExchangeSystem
        >>> type(KineticWealthExchangeSystem(10, debt=2.0).prepare().system_.interactions[0]).__name__
        'DebtExchangeInteraction'
    """

    field = 'economics/econophysics'
    topics = ('wealth', 'agents', 'exchange', 'kinetic')
    references = ('Chakraborti and Chakrabarti 2000', 'Dragulescu and Yakovenko 2000', 'Yakovenko and Rosser 2009')
    until = 200
    observables = (Observable('variance', group='traders', variable='wealth', reduce='var'),)

    def __init__(
        self,
        n_agents: int = 500,
        *,
        saving: float = 0.0,
        debt: float | None = None,
        wealth: float = 1.0,
        amount: float = 0.1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.saving = saving
        self.debt = debt
        self.wealth = wealth
        self.amount = amount

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        if self.debt is not None and self.saving:
            error_msg = 'Give either a saving propensity or a debt limit, not both.'
            raise ValueError(error_msg)
        if not 0 <= self.saving < 1:
            error_msg = f'The saving propensity must lie in [0, 1), got {self.saving}.'
            raise ValueError(error_msg)
        trades = (
            SavingExchangeInteraction('trades', target='traders', saving=self.saving)
            if self.debt is None
            else DebtExchangeInteraction('trades', target='traders', debt=self.debt, amount=self.amount)
        )
        self.system_ = System(
            self.label,
            elements=[SavingTraderElement('traders', wealth=self.wealth, size=self.n_agents)],
            interactions=[trades],
        )
