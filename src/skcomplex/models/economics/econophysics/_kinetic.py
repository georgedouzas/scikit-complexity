"""Econophysics models: kinetic wealth exchange with saving propensity or a debt limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.elements import GroupElement
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._debt_exchange import DebtExchangeInteraction
from ._saving_exchange import SavingExchangeInteraction
from ._saving_trader import SavingTraderElement


@register_model(
    field='economics/econophysics',
    tags=('wealth', 'agents', 'exchange', 'kinetic'),
    references=('Chakraborti and Chakrabarti 2000', 'Dragulescu and Yakovenko 2000', 'Yakovenko and Rosser 2009'),
    defaults={'until': 200},
    observables=lambda system: [ObservableSpec('variance', group='traders', variable='wealth', reduce='var')],
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
        'DebtExchangeInteraction'
    """
    if debt is not None and saving:
        error_msg = 'Give either a saving propensity or a debt limit, not both.'
        raise ValueError(error_msg)
    if not 0 <= saving < 1:
        error_msg = f'The saving propensity must lie in [0, 1), got {saving}.'
        raise ValueError(error_msg)
    trades = (
        SavingExchangeInteraction('trades', source='traders', target='traders', saving=saving)
        if debt is None
        else DebtExchangeInteraction('trades', source='traders', target='traders', debt=debt, amount=amount)
    )
    return System(
        elements=[GroupElement('traders', element=SavingTraderElement, size=n_agents, values={'wealth': wealth})],
        interactions=[trades],
    )
