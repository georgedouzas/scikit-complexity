"""Finance models: wealth that grows by random returns and flows between connected agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx

from ....core.base import Domain, parameter, state
from ....core.model import Element, Field, NetworkSpace, Noise, Observable, Rate, System, laplacian
from ....core.runtime import register_model

Saver = Element.define(
    'Saver',
    states={'W': state(default=1.0, domain=Domain.POSITIVE)},
    parameters={
        'J': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'sigma': parameter(default=0.5, domain=Domain.NONNEGATIVE),
        'degree': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [Rate(s.W, s.J / s.degree * laplacian(s.W)), Noise(s.W, s.sigma * s.W)],
)


@register_model(
    field='economics/finance',
    tags=('wealth', 'sde', 'network', 'pareto'),
    references=('Yakovenko and Rosser 2009', 'Bouchaud and Mezard 2000'),
    defaults={'until': 1.0, 'dt': 0.1},
)
def bouchaud_mezard_wealth(
    n_agents: int = 20,
    *,
    graph: object = None,
    J: float = 1.0,
    sigma: float = 0.5,
    wealth: float | Sequence[float] = 1.0,
) -> System:
    """Return Bouchaud and Mezard's wealth dynamics, random multiplicative returns with exchange between neighbours.

    Agent `i` follows the Stratonovich equation `dW_i = sigma W_i dB_i + (J / c) sum_j (W_j - W_i) dt` over its
    neighbours `j` on a network of mean degree `c`. Exchange conserves the total wealth, and on the complete graph the
    relative wealth `W_i / <W>` approaches a stationary distribution with the Pareto tail of exponent
    `1 + J / sigma^2`.

    Args:
        n_agents:
            The number of agents, used when no graph is given.

        graph:
            The NetworkX graph of the exchanges, or `None` for the complete graph of `n_agents` nodes.

        J:
            The exchange rate between agents.

        sigma:
            The volatility of the returns.

        wealth:
            The initial wealth of every agent, or one value per agent.

    Returns:
        system:
            The model, with the wealth field `agents.W` and the mean wealth `mean_wealth`.

    Examples:
        >>> from skcomplex.models.economics.finance import bouchaud_mezard_wealth
        >>> bouchaud_mezard_wealth(10).elements[0].values['J']
        1.0
    """
    graph = nx.complete_graph(n_agents) if graph is None else graph
    size = nx.number_of_nodes(graph)
    degree = 2 * nx.number_of_edges(graph) / size
    agents = Field('agents', Saver, NetworkSpace(graph), values={'W': wealth, 'J': J, 'sigma': sigma, 'degree': degree})
    return System(
        elements=[agents],
        observables=[Observable('mean_wealth', group='agents', variable='W')],
        noise='stratonovich',
    )
