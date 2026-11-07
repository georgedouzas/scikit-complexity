"""Finance systems: an order book of noise traders who follow gurus on a network that rewires towards wealth."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._guru_market import GuruMarketInteraction
from ._guru_trader import GuruTraderElement

HOLDINGS = ('cash', 'stocks')


@register_system(
    field='economics/finance',
    tags=('stock market', 'agents', 'order book', 'imitation', 'network', 'gurus', 'noise traders'),
    references=('Caiani et al. 2016 §3.2',),
    defaults={'until': 300, 'invariants': {f'total_{name}': 1e-6 for name in HOLDINGS}},
    observables=lambda system: [
        ObservableSpec(f'total_{name}', group='traders', variable=name, reduce='sum') for name in HOLDINGS
    ],
)
def guru_order_book(
    n_traders: int = 100,
    *,
    tau: int = 100,
    beta_step: float = 1.5,
    sigma: float = 0.01,
    cash: float = 1000.0,
    stocks: float = 100.0,
    opening: float = 10.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the order-driven market of Grilli and Tedeschi, whose noise traders imitate gurus they choose by wealth.

    Noise traders expect random returns of their own volatility, but every day each adopts the expectation of the
    trader it follows on a directed communication network. Every day each trader moves its link to a random trader with
    a logit probability of the difference of their wealths, whose intensity grows every day the guru, the most followed
    trader, keeps its place, so followers concentrate on successful traders and their expectations synchronize. The
    traders then place market and limit orders around a geometric Brownian forecast in an order book with price and
    time priority, within the day. This is the model of Caiani et al. (2016, Chapter 3, Sections 3.2 and 3.4). Imitation
    produces coordination periods in which most traders follow one guru, and the returns get fat tails and volatility
    clusters. The source leaves two choices open: the volatilities are uniform on `(0, sigma)`, and the intensity of
    choice grows by `beta_step` every day the guru survives. The cash and the stocks of the traders are conserved,
    which the default run checks as invariants.

    Args:
        n_traders:
            The number of traders.

        tau:
            The intraday operations of a day.

        beta_step:
            The growth of the intensity of choice every day the guru survives.

        sigma:
            The largest volatility of the expected returns of a trader.

        cash:
            The initial cash of every trader.

        stocks:
            The initial stocks of every trader.

        opening:
            The price before the first day.

        random_state:
            The seed or generator of the volatilities and the initial links.

    Returns:
        system:
            The system, with the group `traders` and the market `market`, whose reporters are `market.price`,
            `market.log_return`, `market.volume`, `market.guru`, `market.beta`, and `market.guru_followers`. The
            observables of the system are the totals `total_cash` and `total_stocks`.

    Examples:
        >>> from skcomplex.systems.economics.finance import guru_order_book
        >>> guru_order_book(10).elements[0].prepare().size_
        10
    """
    rng = np.random.default_rng(random_state)
    links = rng.integers(n_traders - 1, size=n_traders)
    links += links >= np.arange(n_traders)
    traders = GroupElement(
        'traders',
        element=GuruTraderElement,
        size=n_traders,
        values={
            'cash': cash,
            'stocks': stocks,
            'wealth': cash + opening * stocks,
            'follows': links.astype(float),
            'sigma': rng.uniform(0.0, sigma, n_traders),
        },
    )
    market = GuruMarketInteraction('market', traders='traders', tau=tau, beta_step=beta_step, opening=opening)
    return System(elements=[traders], interactions=[market])
