"""Innovation systems: discovery processes of explorers whose urns grow and share their adjacent possible."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec


class BallView(AgentView):
    """Balls of one colour as the urn sees them: their `color`, their `count`, and whether the node `drawn` it."""

    color: float
    count: float
    drawn: float


class BallElement(Element):
    """Balls of one colour in the urn of a node, with how many there are and whether the node has drawn the colour.

    A ball agent stands for every ball of its colour in the urn of the node it sits on, so the set of ball agents on a
    node is the set of colours of its urn, and their multiplicities make the urn a multiset.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        color:
            The colour, a number shared by the balls of that colour in every urn.

        count:
            The number of balls of the colour in the urn.

        drawn:
            1 when the node has drawn the colour, and 0 when it has not.

    Examples:
        >>> from skcomplex.systems.economics.innovation import BallElement
        >>> BallElement('balls', color=3, count=2).count
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        color: Annotated[float, StateSpec(domain=NN)] = 0,
        count: Annotated[float, StateSpec(domain=NN)] = 1,
        drawn: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.color = color
        self.count = count
        self.drawn = drawn

    @classmethod
    def _get_view(cls) -> type[BallView]:
        """Return the view of the agents of the element."""
        return BallView


class UrnView(AgentView):
    """Urn as its rules see it: its `novelties`, the `color` of its last draw, and whether that draw was `novel`."""

    novelties: float
    color: float
    novel: float


DRAW = 'draw'
GROW = 'grow'
BALLS = 'balls'


class UrnElement(Element):
    """Urn with triggering on a node of a network, which draws from its own balls and those of its neighbours.

    The urn of the interacting discovery processes of Iacopini et al. (2020). The urn of a node is the set of ball
    agents on its node. Every step the node draws one ball from its social urn, its own balls with their multiplicities
    and one ball of every colour in the urns of its neighbours, in the phase `'draw'`. In the phase `'grow'` it adds
    `rho` balls of the drawn colour to its own urn, and when it draws the colour for the first time, a novelty, it adds
    one ball of each of `nu + 1` colours that no urn held before, the adjacent possible.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.systems.economics.innovation import UrnElement
        >>> UrnElement('urns', rho=4, nu=2).nu
        2
    """

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        novelties: Annotated[float, StateSpec(domain=NN)] = 0,
        color: Annotated[float, StateSpec()] = -1,
        novel: Annotated[float, StateSpec(domain=NN)] = 0,
        rho: Annotated[float, ParameterSpec(domain=NN)] = 4,
        nu: Annotated[float, ParameterSpec(domain=NN)] = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.novelties = novelties
        self.color = color
        self.novel = novel
        self.rho = rho
        self.nu = nu

    @classmethod
    def _get_view(cls) -> type[UrnView]:
        """Return the view of the agents of the element."""
        return UrnView

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the draw from the social urn and the growth of the own urn, in their phases.

        Args:
            namespace:
                The namespace of the symbols of the element.

        Returns:
            laws:
                The rules of the two phases.
        """
        return [RuleLaw(self.draw, phase=DRAW), RuleLaw(self.grow, phase=GROW)]

    @staticmethod
    def draw(agent: UrnView, system: SystemView) -> None:
        """Draw a ball from the social urn of the node, and record its colour and whether it is a novelty.

        Args:
            agent:
                The view of the urn.

            system:
                The view of the system.
        """
        own = [ball for ball in agent.agents_here(BALLS) if isinstance(ball, BallView)]
        social = [ball for ball in agent.neighbors(group=BALLS) if isinstance(ball, BallView)]
        colors = [ball.color for ball in own] + [ball.color for ball in social]
        weights = np.asarray([ball.count for ball in own] + [1.0] * len(social), dtype=float)
        if not colors:
            return
        chosen = colors[int(system.rng.choice(len(colors), p=weights / weights.sum()))]
        agent.color = chosen
        agent.novel = float(not any(ball.color == chosen and ball.drawn for ball in own))

    @staticmethod
    def grow(agent: UrnView, system: SystemView) -> None:
        """Reinforce the drawn colour in the own urn and, after a novelty, add the colours of the adjacent possible.

        Args:
            agent:
                The view of the urn.

            system:
                The view of the system.
        """
        if agent.color < 0:
            return
        rho, nu = system.parameter('rho'), system.parameter('nu')
        own = [ball for ball in agent.agents_here(BALLS) if isinstance(ball, BallView)]
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
            stored if isinstance(stored, int) else int(max(item.color for item in system.agents(BALLS, BallView))) + 1
        )
        for color in range(first, first + int(nu) + 1):
            agent.create(BALLS, color=color, count=1, drawn=0)
        system.storage['colors'] = first + int(nu) + 1


class InteractingUrnsNetworkSystem(BaseSystem):
    """One urn with triggering per node of a network, which draws from its own urn and those of its neighbours.

    The discovery process of Iacopini et al. (2020, Eqs. 1 and 2). Every node holds an urn of `m0` balls of colours
    that no other urn holds. Every step every node draws one ball from its social urn, its own urn and one ball of every
    colour of the urns of its neighbours, all from the urns at the start of the step. It then adds `rho` balls of the
    drawn colour to its own urn, and when it draws a colour for the first time, a novelty, `nu + 1` balls of colours
    that no urn held before. The number of novelties of a node grows as Heaps' law `D_i(t) ~ t^beta_i`, with
    `beta = nu / rho` for an isolated urn and higher exponents for nodes whose neighbours enlarge their social urns.

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

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.economics.innovation import InteractingUrnsNetworkSystem
        >>> system = InteractingUrnsNetworkSystem(nx.path_graph(3), rho=2, nu=1).prepare().system_
        >>> [element.prepare().size_ for element in system.elements]
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
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.rho = rho
        self.nu = nu
        self.m0 = m0

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        graph = self.graph
        graph = nx.karate_club_graph() if graph is None else graph
        nodes = list(graph.nodes)
        initial = self.nu + 1 if self.m0 is None else self.m0
        places = [node for node in nodes for _ in range(initial)]
        colors = list(range(len(places)))
        return System(
            self.label,
            elements=[
                GroupElement(
                    'urns',
                    element=UrnElement,
                    size=len(nodes),
                    location=nodes,
                    values={'rho': self.rho, 'nu': self.nu},
                ),
                GroupElement(
                    'balls',
                    element=BallElement,
                    size=len(places),
                    variable=True,
                    location=places,
                    values={'color': colors},
                ),
            ],
            space=NetworkSpace(graph),
            time=DiscreteTime(
                schedule=PhasesScheduleSpec((PhaseSpec(DRAW, activation='simultaneous'), PhaseSpec(GROW))),
            ),
        )
