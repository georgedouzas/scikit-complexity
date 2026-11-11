"""Finance systems: wealth that grows by random returns and flows between agents.

The agents are connected.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement
from ....core.components.interactions import CouplingInteraction, RateInteraction
from ....core.components.spaces import NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class NetworkSaverElement(Element):
    """Saver of Bouchaud and Mezard, whose wealth earns random returns.

    The wealth flows to and from its exchange partners.

    Args:
        label:
            The label of the saver.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        W:
            The wealth.

        J:
            The exchange rate with the partners.

        sigma:
            The volatility of the returns.

        degree:
            The mean number of exchange partners `c`, which divides the exchange rate, `N - 1` for exchange with every
            other saver.

    Examples:
        >>> from skcomplex.systems.economics.finance import NetworkSaverElement
        >>> NetworkSaverElement('saver', W=2.0).W
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        W: Annotated[object, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        J: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        sigma: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        degree: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.W = W
        self.J = J
        self.sigma = sigma
        self.degree = degree


class BouchaudMezardWealthSystem(BaseSystem):
    """Bouchaud and Mezard's wealth dynamics.

    Returns are random and multiplicative, with exchange between neighbours. Agent `i` follows the Stratonovich equation
    `dW_i = sigma W_i dB_i + (J / c) sum_j (W_j - W_i) dt` over its neighbours `j` on a network of mean degree `c`.
    Without a graph, the agents are the members of a group and every agent exchanges with every other, `c = N - 1`, the
    complete graph at a cost linear in the number of agents. A given graph keeps the agents on a field over its edges.
    Exchange conserves the total wealth, and on the complete graph the relative wealth `W_i / <W>` approaches a
    stationary distribution with the Pareto tail `P(W_i / <W> > w) ~ w^-mu` of exponent `mu = 1 + 2 J / sigma^2`.

    The system has the wealth `agents.W` and the mean wealth `mean_wealth`.

    Args:
        n_agents:
            The number of agents, used when no graph is given.

        graph:
            The NetworkX graph of the exchanges, or `None` for exchange with every other of `n_agents` agents.

        J:
            The exchange rate between agents.

        sigma:
            The volatility of the returns.

        wealth:
            The initial wealth of every agent, or one value per agent.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import BouchaudMezardWealthSystem
        >>> BouchaudMezardWealthSystem(10).prepare().system_.elements[0].values['J']
        1.0
    """

    field = 'economics/finance'
    topics = ('wealth', 'sde', 'network', 'pareto')
    references = ('Yakovenko and Rosser 2009', 'Bouchaud and Mezard 2000')
    until = 1.0
    dt = 0.1
    observables = (ObservableSpec('mean_wealth', group='agents', variable='W'),)

    def __init__(
        self,
        n_agents: int = 20,
        *,
        graph: object = None,
        J: float = 1.0,
        sigma: float = 0.5,
        wealth: float | Sequence[float] = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.graph = graph
        self.J = J
        self.sigma = sigma
        self.wealth = wealth

    @staticmethod
    def _exchange(s: SimpleNamespace) -> Expression:
        """Return the flow `(J / c) (W_j - W_i)` from saver `j` to saver `i`."""
        return s.member.J / s.member.degree * (s.other.W - s.member.W)

    @staticmethod
    def _returns(s: SimpleNamespace) -> Expression:
        """Return the amplitude `sigma W` of the random returns."""
        return s.target.sigma * s.target.W

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        returns = RateInteraction('returns', target='agents', state='W', noise=self._returns, convention='stratonovich')
        if self.graph is None:
            values = {'W': self.wealth, 'J': self.J, 'sigma': self.sigma, 'degree': self.n_agents - 1}
            agents: FieldElement | GroupElement = GroupElement(
                'agents',
                element=NetworkSaverElement,
                size=self.n_agents,
                values=values,
            )
            mode = 'group'
        else:
            degree = 2 * nx.number_of_edges(self.graph) / nx.number_of_nodes(self.graph)
            values = {'W': self.wealth, 'J': self.J, 'sigma': self.sigma, 'degree': degree}
            agents = FieldElement('agents', element=NetworkSaverElement, space=NetworkSpace(self.graph), values=values)
            mode = 'edges'
        exchange = CouplingInteraction(
            'exchange',
            target='agents',
            read='W',
            write='W',
            function=self._exchange,
            mode=mode,
        )
        return CompositeSystem(self.label, elements=[agents], interactions=[exchange, returns])
