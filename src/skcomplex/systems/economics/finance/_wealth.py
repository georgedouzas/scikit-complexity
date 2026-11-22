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

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import total
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System


class NetworkSaverElement(Element):
    """Saver of Bouchaud and Mezard, whose wealth earns random returns.

    The wealth flows to and from its exchange partners.

    Args:
        label:
            The label of the saver.

        W:
            The wealth.

        J:
            The exchange rate with the partners.

        sigma:
            The volatility of the returns.

        degree:
            The mean number of exchange partners `c`, which divides the exchange rate, `N - 1` for exchange with every
            other saver.

        size:
            The number of savers, or `None` for one or one per node of the space.

        space:
            The network whose nodes hold the savers, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.finance import NetworkSaverElement
        >>> NetworkSaverElement('saver', W=2.0).W
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        W: Annotated[object, RealSet.open(0, oo)] = 1.0,
        J: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        sigma: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        degree: Annotated[object, RealSet.open(0, oo)] = 1.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.finance import BouchaudMezardWealthSystem
        >>> BouchaudMezardWealthSystem(10).prepare().system_.elements[0].J
        1.0
    """

    field = 'economics/finance'
    topics = ('wealth', 'sde', 'network', 'pareto')
    references = ('Yakovenko and Rosser 2009', 'Bouchaud and Mezard 2000')
    until = 1.0
    dt = 0.1
    observables = (Observable('mean_wealth', group='agents', variable='W'),)

    def __init__(
        self,
        n_agents: int = 20,
        *,
        graph: object = None,
        J: float = 1.0,
        sigma: float = 0.5,
        wealth: float | Sequence[float] = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.graph = graph
        self.J = J
        self.sigma = sigma
        self.wealth = wealth

    @staticmethod
    def _get_flow(member: SimpleNamespace, other: SimpleNamespace) -> Expression:
        """Return the flow `(J / c) (W_j - W_i)` from saver `j` to saver `i`."""
        return member.J / member.degree * (other.W - member.W)

    @staticmethod
    def _get_returns(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the random returns `sigma W dB` of every saver, in the Stratonovich sense."""
        return [NoiseLaw(s.target.W, s.target.sigma * s.target.W, convention='stratonovich')]

    @staticmethod
    def _get_exchange(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the exchange of every saver with every other saver."""
        saver = s.target
        return [RateLaw(saver.W, total(BouchaudMezardWealthSystem._get_flow(saver, saver.other), over=saver.other))]

    @staticmethod
    def _get_edge_exchange(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the exchange of both ends of a link."""
        flow = BouchaudMezardWealthSystem._get_flow
        return [RateLaw(s.target.W, flow(s.target, s.source)), RateLaw(s.source.W, flow(s.source, s.target))]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        returns = Interaction('returns', roles={'target': 'agents'}, laws=self._get_returns)
        if self.graph is None:
            agents = NetworkSaverElement(
                'agents',
                W=self.wealth,
                J=self.J,
                sigma=self.sigma,
                degree=self.n_agents - 1,
                size=self.n_agents,
            )
            exchange = Interaction('exchange', roles={'target': 'agents'}, laws=self._get_exchange)
        else:
            degree = 2 * nx.number_of_edges(self.graph) / nx.number_of_nodes(self.graph)
            agents = NetworkSaverElement(
                'agents',
                W=self.wealth,
                J=self.J,
                sigma=self.sigma,
                degree=degree,
                space=NetworkSpace(self.graph),
            )
            exchange = Interaction('exchange', among='agents', along=True, laws=self._get_edge_exchange)
        self.system_ = System(self.label, elements=[agents], interactions=[exchange, returns])
