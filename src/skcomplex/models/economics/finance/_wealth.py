"""Finance models: wealth that grows by random returns and flows between connected agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx

from ....core.base import Expression, NumericDomain, parameter, state
from ....core.components import Element, GroupElement
from ....core.elements import FieldElement
from ....core.laws import NoiseLaw, RateLaw, Symbols, count, mean
from ....core.laws.utils import laplacian
from ....core.spaces import NetworkSpace
from ....core.specs import ObservableSpec
from ....core.system import System
from ....engine.runtime import register_model
from ...core.base import BaseLaw


class SaverElement(Element):
    """Saver element."""

    W = state(default=1.0, domain=NumericDomain.POSITIVE)
    J = parameter(default=1.0, domain=NumericDomain.NONNEGATIVE)
    sigma = parameter(default=0.5, domain=NumericDomain.NONNEGATIVE)
    degree = parameter(default=1.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the saver element."""
        return [
            RateLaw(s.W, s.J / s.degree * laplacian(s.W)),
            NoiseLaw(s.W, s.sigma * s.W, convention='stratonovich'),
        ]


def _mean_exchange(s: Symbols) -> Expression:
    """Return the exchange `(J / (N - 1)) sum_j (W_j - W_i)` with every other agent, written with the group mean."""
    return s.J * count() / (count() - 1) * (mean(s.W) - s.W)


class MeanFieldSaverElement(Element):
    """Mean field saver element."""

    W = state(default=1.0, domain=NumericDomain.POSITIVE)
    J = parameter(default=1.0, domain=NumericDomain.NONNEGATIVE)
    sigma = parameter(default=0.5, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the mean field saver element."""
        return [RateLaw(s.W, _mean_exchange(s)), NoiseLaw(s.W, s.sigma * s.W, convention='stratonovich')]


@register_model(
    field='economics/finance',
    tags=('wealth', 'sde', 'network', 'pareto'),
    references=('Yakovenko and Rosser 2009', 'Bouchaud and Mezard 2000'),
    defaults={'until': 1.0, 'dt': 0.1},
    observables=lambda system: [ObservableSpec('mean_wealth', group='agents', variable='W')],
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
    neighbours `j` on a network of mean degree `c`. Without a graph, the agents are the members of a group and every
    agent exchanges with the group mean, `(J N / (N - 1)) (<W> - W_i)`, the complete graph at a cost linear in the
    number of agents. A given graph keeps the agents on a field with its Laplacian. Exchange conserves the total wealth,
    and on the complete graph the relative wealth `W_i / <W>` approaches a stationary distribution with the Pareto tail
    `P(W_i / <W> > w) ~ w^-mu` of exponent `mu = 1 + 2 J / sigma^2`.

    Args:
        n_agents:
            The number of agents, used when no graph is given.

        graph:
            The NetworkX graph of the exchanges, or `None` for exchange with the mean of `n_agents` agents.

        J:
            The exchange rate between agents.

        sigma:
            The volatility of the returns.

        wealth:
            The initial wealth of every agent, or one value per agent.

    Returns:
        system:
            The model, with the wealth `agents.W` and the mean wealth `mean_wealth`.

    Examples:
        >>> from skcomplex.models.economics.finance import bouchaud_mezard_wealth
        >>> bouchaud_mezard_wealth(10).elements[0].values['J']
        1.0
    """
    if graph is None:
        agents: FieldElement | GroupElement = GroupElement(
            'agents',
            element=MeanFieldSaverElement,
            size=n_agents,
            values={'W': wealth, 'J': J, 'sigma': sigma},
        )
    else:
        degree = 2 * nx.number_of_edges(graph) / nx.number_of_nodes(graph)
        values = {'W': wealth, 'J': J, 'sigma': sigma, 'degree': degree}
        agents = FieldElement('agents', SaverElement, NetworkSpace(graph), values=values)
    return System(elements=[agents])
