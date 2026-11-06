"""Define an urn with triggering on a node of a network, which draws from its own urn and the urns of its neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._ball_view import BallView
from ._urn_view import UrnView

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
