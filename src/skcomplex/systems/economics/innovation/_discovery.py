"""Innovation systems: discovery processes of explorers whose urns grow.

The urns share their adjacent possible.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import AgentView, BaseSolver, MesaSolver, SystemView
from ....core.specs import Role
from ....core.systems import BaseSystem, System

DRAW = 'draw'
GROW = 'grow'
BALLS = 'balls'


class BallView(AgentView):
    """Balls of one colour as the urn sees them.

    They hold their `color`, their `count`, and whether the node `drawn` it.
    """

    color: float
    count: float
    drawn: float


class BallElement(Element):
    """Balls of one colour in the urn of a node.

    The element holds how many there are and whether the node has drawn the colour.

    A ball agent stands for every ball of its colour in the urn of the node it sits on, so the set of ball agents on a
    node is the set of colours of its urn, and their multiplicities make the urn a multiset.

    Args:
        label:
            The label of the agent or group.

        color:
            The colour, a number shared by the balls of that colour in every urn.

        count:
            The number of balls of the colour in the urn.

        drawn:
            1 when the node has drawn the colour, and 0 when it has not.

        values:
            Further values by name, such as the node `node` of each member on the network.

        size:
            The number of balls, or `None` for one.

        space:
            The network the balls live on, whose axis `node` places them, or `None`.

    Examples:
        >>> from skcomplex.systems.economics.innovation import BallElement
        >>> BallElement('balls', color=3, count=2).count
        2
    """

    def __init__(
        self,
        label: str,
        *,
        color: Annotated[object, NN] = 0,
        count: Annotated[object, NN] = 1,
        drawn: Annotated[object, NN] = 0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.color = color
        self.count = count
        self.drawn = drawn

    @classmethod
    def _get_view(cls) -> type[BallView]:
        """Return the view of the agents of the element."""
        return BallView


class UrnView(AgentView):
    """Urn as its rules see it.

    It holds its `novelties`, the `color` of its last draw, whether that draw was `novel`, and its `rho` and `nu`.
    """

    novelties: float
    color: float
    novel: float
    rho: float
    nu: float


class UrnElement(Element):
    """Urn with triggering on a node of a network.

    It draws from its own balls and those of its neighbours by `TriggeringInteraction`. The urn of the interacting
    discovery processes of Iacopini et al. (2020). The urn of a node is the set of ball agents on its node.

    Args:
        label:
            The label of the agent or group.

        novelties:
            The number of distinct colours the node has drawn, `D_i(t)`.

        color:
            The colour of the last draw.

        novel:
            1 when the last draw was a novelty, and 0 when it was not.

        rho:
            The reinforcement, the balls of the drawn colour added after every draw.

        nu:
            The triggering, one less than the colours added after a novelty.

        values:
            Further values by name, such as the node `node` of each member on the network.

        size:
            The number of urns, or `None` for one.

        space:
            The network the urns live on, whose axis `node` places them, or `None`.

    Examples:
        >>> from skcomplex.systems.economics.innovation import UrnElement
        >>> UrnElement('urns', rho=4, nu=2).nu
        2
    """

    def __init__(
        self,
        label: str,
        *,
        novelties: Annotated[object, NN] = 0,
        color: object = -1,
        novel: Annotated[object, NN] = 0,
        rho: Annotated[object, NN] = 4,
        nu: Annotated[object, NN] = 1,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.novelties = novelties
        self.color = color
        self.novel = novel
        self.rho = rho
        self.nu = nu

    @classmethod
    def _get_view(cls) -> type[UrnView]:
        """Return the view of the agents of the element."""
        return UrnView


class TriggeringInteraction(Interaction):
    """Draws of urns with triggering from their social urns, and the growth of their own urns.

    Every step every node draws one ball from its social urn, its own balls with their multiplicities and one ball of
    every colour in the urns of its neighbours, in the phase `'draw'`. In the phase `'grow'` it adds `rho` balls of the
    drawn colour to its own urn, and when it draws the colour for the first time, a novelty, it adds one ball of each
    of `nu + 1` colours that no urn held before, the adjacent possible.

    Args:
        label:
            The label of the interaction.

        target:
            The urns that draw.

        balls:
            The balls of every urn.

    Examples:
        >>> from skcomplex.systems.economics.innovation import TriggeringInteraction
        >>> TriggeringInteraction('triggering').balls
        'balls'
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'urns', balls: Role = BALLS) -> None:
        super().__init__(label)
        self.target = target
        self.balls = balls

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the draw and the growth of every urn, each in its phase."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [
                UpdateLaw(context.target, self.draw, phase=DRAW),
                UpdateLaw(context.target, self.grow, phase=GROW),
            ]

    @staticmethod
    def draw(agent: UrnView, system: SystemView) -> None:
        """Draw a ball from the social urn of the node.

        It records its colour and whether it is a novelty.

        Args:
            agent:
                The view of the urn.

            system:
                The view of the system.
        """
        own = [ball for ball in agent.get_agents_here(BALLS) if isinstance(ball, BallView)]
        social = [ball for ball in agent.get_neighbors(group=BALLS) if isinstance(ball, BallView)]
        colors = [ball.color for ball in own] + [ball.color for ball in social]
        weights = np.asarray([ball.count for ball in own] + [1.0] * len(social), dtype=float)
        if not colors:
            return
        chosen = colors[int(system.rng.choice(len(colors), p=weights / weights.sum()))]
        agent.color = chosen
        agent.novel = float(not any(ball.color == chosen and ball.drawn for ball in own))

    @staticmethod
    def grow(agent: UrnView, system: SystemView) -> None:
        """Reinforce the drawn colour in the own urn.

        After a novelty, it adds the colours of the adjacent possible.

        Args:
            agent:
                The view of the urn.

            system:
                The view of the system.
        """
        if agent.color < 0:
            return
        rho, nu = agent.rho, agent.nu
        own = [ball for ball in agent.get_agents_here(BALLS) if isinstance(ball, BallView)]
        ball = next((ball for ball in own if ball.color == agent.color), None)
        if ball is None:
            agent.create(BALLS, color=agent.color, count=rho, drawn=1)
        else:
            ball.count += rho
            ball.drawn = 1
        if not agent.novel:
            return
        agent.novelties += 1
        stored = system.storage.get('colors')
        first = (
            stored
            if isinstance(stored, int)
            else int(max(item.color for item in system.get_agents(BALLS, BallView))) + 1
        )
        for color in range(first, first + int(nu) + 1):
            agent.create(BALLS, color=color, count=1, drawn=0)
        system.storage['colors'] = first + int(nu) + 1


class InteractingUrnsNetworkSystem(BaseSystem):
    """One urn with triggering per node of a network.

    Every node draws from its own urn and those of its neighbours. The discovery process of Iacopini et al. (2020, Eqs.
    1 and 2). Every node holds an urn of `m0` balls of colours that no other urn holds. Every step every node draws one
    ball from its social urn, its own urn and one ball of every colour of the urns of its neighbours, all from the urns
    at the start of the step. It then adds `rho` balls of the drawn colour to its own urn, and when it draws a colour
    for the first time, a novelty, `nu + 1` balls of colours that no urn held before. The number of novelties of a node
    grows as Heaps' law `D_i(t) ~ t^beta_i`, with `beta = nu / rho` for an isolated urn and higher exponents for nodes
    whose neighbours enlarge their social urns.

    The urn of a node is the set of ball agents of the variable group `balls` on its node, one agent per colour with its
    multiplicity, which the node reads with `agents_here` and its neighbours with `neighbors(group='balls')`. The system
    has the urns `urns` and their novelties `urns.novelties`, and the balls `balls`.

    Args:
        graph:
            The NetworkX graph of the explorers, or `None` for Zachary's karate club.

        rho:
            The reinforcement, the balls of the drawn colour added after every draw.

        nu:
            The triggering, one less than the colours added after a novelty.

        m0:
            The initial number of balls of every urn, or `None` for `nu + 1`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.economics.innovation import InteractingUrnsNetworkSystem
        >>> system = InteractingUrnsNetworkSystem(nx.path_graph(3), rho=2, nu=1).prepare().system_
        >>> [element.size for element in system.elements]
        [3, 6]
    """

    field = 'economics/innovation'
    topics = ('innovation', 'adjacent possible', 'urn', 'network', 'agents', 'heaps law')
    references = ('Iacopini et al. 2020',)
    until = 100

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        rho: int = 4,
        nu: int = 1,
        m0: int | None = None,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.rho = rho
        self.nu = nu
        self.m0 = m0

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        graph = self.graph
        graph = nx.karate_club_graph() if graph is None else graph
        nodes = list(graph.nodes)
        initial = self.nu + 1 if self.m0 is None else self.m0
        places = [node for node in nodes for _ in range(initial)]
        colors = list(range(len(places)))
        space = NetworkSpace(graph)
        self.system_ = System(
            self.label,
            elements=[
                UrnElement(
                    'urns',
                    rho=self.rho,
                    nu=self.nu,
                    values={'node': nodes},
                    size=len(nodes),
                    space=space,
                ),
                BallElement(BALLS, color=colors, values={'node': places}, size=len(places), space=space),
            ],
            interactions=[TriggeringInteraction('triggering')],
            space=space,
            solver=MesaSolver(phases=(PhaseSpec(DRAW, activation='simultaneous'), PhaseSpec(GROW))),
        )
