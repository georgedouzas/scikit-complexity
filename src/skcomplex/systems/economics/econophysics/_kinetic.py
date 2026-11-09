"""Econophysics systems: kinetic wealth exchange.

The exchange has a saving propensity or a debt limit.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TransferInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class SavingTraderView(AgentView):
    """Trader as the trading rules see it.

    It holds its `wealth`, negative down to the debt limit.
    """

    wealth: float


class SavingTraderElement(Element):
    """Agent of the kinetic wealth exchange systems.

    Its wealth changes only in pairwise trades.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        wealth:
            The initial wealth, negative down to the debt limit.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.econophysics import SavingTraderElement
        >>> group = GroupElement('traders', element=SavingTraderElement, size=10, values={'wealth': 2.0})
        >>> group.label, group.element.__name__, group.size
        ('traders', 'SavingTraderElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth

    @classmethod
    def _get_view(cls) -> type[SavingTraderView]:
        """Return the view of a trader, with its wealth."""
        return SavingTraderView


class DebtExchangeInteraction(TransferInteraction):
    """Random pairwise trades of Dragulescu and Yakovenko of a fixed amount.

    Debt is allowed down to a limit. In every trade an agent pays a fixed amount to another agent drawn at random,
    unless its wealth would fall below minus the debt limit. Every step, every agent pays once, in random order. The
    stationary wealth distribution is the exponential `P(w) = exp(-(w + d) / T) / T` shifted to the debt limit `d`, with
    temperature `T = <w> + d`.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the group of traders.

        target:
            The path of the group of traders, the source again.

        debt:
            The debt limit `d`, the largest debt an agent can have.

        amount:
            The amount paid in every trade.

        phase:
            The phase the trades run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import DebtExchangeInteraction
        >>> DebtExchangeInteraction('trades', source='traders', target='traders', debt=5.0).debt
        5.0
    """

    def __init__(
        self,
        label: str,
        *,
        source: str,
        target: str,
        debt: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        amount: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            source=source,
            target=target,
            state='wealth',
            amount=amount,
            pairing='random',
            guard=self._is_solvent,
            form='agent',
            phase=phase,
        )
        self.debt = debt

    def _is_solvent(self, payer: SavingTraderView, payee: SavingTraderView, system: SystemViewProtocol) -> bool:
        """Return whether the payer stays above minus the debt limit after paying."""
        return bool(payer.wealth - system.parameter('amount') >= -system.parameter('debt'))


class SavingExchangeInteraction(TransferInteraction):
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

        source:
            The path of the group of traders.

        target:
            The path of the group of traders, the source again.

        saving:
            The saving propensity `lambda`, the fraction of its wealth that an agent keeps out of a trade.

        phase:
            The phase the trades run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import SavingExchangeInteraction
        >>> SavingExchangeInteraction('trades', source='traders', target='traders').saving
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        source: str,
        target: str,
        saving: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            source=source,
            target=target,
            state='wealth',
            amount=self._get_amount,
            pairing='random',
            form='agent',
            phase=phase,
        )
        self.saving = saving

    def _get_amount(self, first: SavingTraderView, second: SavingTraderView, system: SystemViewProtocol) -> float:
        """Return the wealth the first trader transfers to the second, of either sign."""
        saving = system.parameter('saving')
        share = float(system.rng.random())
        return (1 - saving) * (first.wealth - share * (first.wealth + second.wealth))


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
    observables = (ObservableSpec('variance', group='traders', variable='wealth', reduce='var'),)

    def __init__(
        self,
        n_agents: int = 500,
        *,
        saving: float = 0.0,
        debt: float | None = None,
        wealth: float = 1.0,
        amount: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.saving = saving
        self.debt = debt
        self.wealth = wealth
        self.amount = amount

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        if self.debt is not None and self.saving:
            error_msg = 'Give either a saving propensity or a debt limit, not both.'
            raise ValueError(error_msg)
        if not 0 <= self.saving < 1:
            error_msg = f'The saving propensity must lie in [0, 1), got {self.saving}.'
            raise ValueError(error_msg)
        trades = (
            SavingExchangeInteraction('trades', source='traders', target='traders', saving=self.saving)
            if self.debt is None
            else DebtExchangeInteraction(
                'trades',
                source='traders',
                target='traders',
                debt=self.debt,
                amount=self.amount,
            )
        )
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'traders',
                    element=SavingTraderElement,
                    size=self.n_agents,
                    values={'wealth': self.wealth},
                ),
            ],
            interactions=[trades],
        )
