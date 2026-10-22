"""Econophysics models: kinetic wealth exchange with saving propensity or a debt limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.backends import ModelView
from ....core.base import Domain, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.system import Observable, System


class Saver(Element):
    """Agent of the kinetic wealth exchange models, whose wealth changes only in pairwise trades.

    Args:
        label:
            The label of the agent or group.

        wealth:
            The initial wealth, negative down to the debt limit.

    Examples:
        >>> from skcomplex.models.economics.econophysics import Saver
        >>> Saver.group('savers', 10, wealth=2.0)
        Group('savers', element=Saver, size=10)
    """

    wealth: float = state(default=1.0, domain=Domain.REAL)


class SavingExchange(Interaction):
    """Random pairwise trades of Chakraborti and Chakrabarti, where every agent saves a fraction of its wealth.

    A trade between agents `i` and `j` keeps total wealth and gives `w_i' = lambda w_i + eps (1 - lambda) (w_i + w_j)`
    and `w_j' = lambda w_j + (1 - eps) (1 - lambda) (w_i + w_j)` with `eps` uniform on `[0, 1]`. Every step makes as
    many trades as there are agents. Without saving the stationary wealth distribution is exponential, and with
    saving propensity `lambda` it is close to a Gamma distribution of shape `n = 1 + 3 lambda / (1 - lambda)`.

    Args:
        label:
            The label of the interaction.

        traders:
            The path of the group of traders.

        saving:
            The saving propensity `lambda`, the fraction of its wealth that an agent keeps out of a trade.

    Examples:
        >>> from skcomplex.models.economics.econophysics import SavingExchange
        >>> SavingExchange('trades', traders='savers').saving
        0.0
    """

    traders: str = role()
    saving: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the trading rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.trade)]

    def trade(self, model: ModelView) -> None:
        """Make one random trade per agent.

        Args:
            model:
                The model.
        """
        agents = model.agents('traders')
        wealth = np.array([agent.wealth for agent in agents], dtype=float)
        saving = model.parameter('saving')
        pairs = model.rng.integers(len(agents), size=(len(agents), 2))
        shares = model.rng.random(len(agents))
        for (first, second), share in zip(pairs, shares, strict=True):
            if first == second:
                continue
            pool = (1 - saving) * (wealth[first] + wealth[second])
            wealth[first] = saving * wealth[first] + share * pool
            wealth[second] = saving * wealth[second] + (1 - share) * pool
        for agent, value in zip(agents, wealth, strict=True):
            agent.wealth = float(value)


class DebtExchange(Interaction):
    """Random pairwise trades of Dragulescu and Yakovenko of a fixed amount, allowing debt down to a limit.

    In every trade a random agent pays a fixed amount to another random agent, unless its wealth would fall below
    minus the debt limit. Every step makes as many trades as there are agents. The stationary wealth distribution is
    the exponential `P(w) = exp(-(w + d) / T) / T` shifted to the debt limit `d`, with temperature `T = <w> + d`.

    Args:
        label:
            The label of the interaction.

        traders:
            The path of the group of traders.

        debt:
            The debt limit `d`, the largest debt an agent can have.

        amount:
            The amount paid in every trade.

    Examples:
        >>> from skcomplex.models.economics.econophysics import DebtExchange
        >>> DebtExchange('trades', traders='savers', debt=5.0).debt
        5.0
    """

    traders: str = role()
    debt: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    amount: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the trading rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.trade)]

    def trade(self, model: ModelView) -> None:
        """Make one random trade per agent.

        Args:
            model:
                The model.
        """
        agents = model.agents('traders')
        wealth = np.array([agent.wealth for agent in agents], dtype=float)
        floor, amount = -model.parameter('debt'), model.parameter('amount')
        for payer, payee in model.rng.integers(len(agents), size=(len(agents), 2)):
            if payer != payee and wealth[payer] - amount >= floor:
                wealth[payer] -= amount
                wealth[payee] += amount
        for agent, value in zip(agents, wealth, strict=True):
            agent.wealth = float(value)


@register_model(
    field='economics/econophysics',
    tags=('wealth', 'agents', 'exchange', 'kinetic'),
    references=('Chakraborti and Chakrabarti 2000', 'Dragulescu and Yakovenko 2000', 'Yakovenko and Rosser 2009'),
    defaults={'until': 200},
    observables=lambda system: [Observable('variance', group='traders', variable='wealth', reduce='var')],
)
def kinetic_wealth_exchange(
    n_agents: int = 500,
    *,
    saving: float = 0.0,
    debt: float | None = None,
    wealth: float = 1.0,
    amount: float = 0.1,
) -> System:
    """Return a kinetic wealth exchange model with a saving propensity or a debt limit.

    Without a debt limit, agents trade random shares of their pooled wealth after saving a fraction `saving` of it,
    as in Chakraborti and Chakrabarti. With a debt limit, they pay a fixed amount in every trade and may go into debt
    down to the limit, as in Dragulescu and Yakovenko.

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

    Returns:
        system:
            The model, with the group `traders`, the interaction `trades`, and the variance of wealth `variance`.

    Raises:
        ValueError:
            Both a saving propensity and a debt limit are given, or the saving propensity is not below 1.

    Examples:
        >>> from skcomplex.models.economics.econophysics import kinetic_wealth_exchange
        >>> type(kinetic_wealth_exchange(10, debt=2.0).interactions[0]).__name__
        'DebtExchange'
    """
    if debt is not None and saving:
        error_msg = 'Give either a saving propensity or a debt limit, not both.'
        raise ValueError(error_msg)
    if not 0 <= saving < 1:
        error_msg = f'The saving propensity must lie in [0, 1), got {saving}.'
        raise ValueError(error_msg)
    trades = (
        SavingExchange('trades', traders='traders', saving=saving)
        if debt is None
        else DebtExchange('trades', traders='traders', debt=debt, amount=amount)
    )
    return System(
        components=[Saver.group('traders', n_agents, wealth=wealth)],
        interactions=[trades],
    )
