"""Learning on a spin-glass landscape.

The module holds the SK game of reinforcement learners on quenched random couplings.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import total, uniform, where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ....core.types import Array


class LearningAgentElement(Element):
    """Agents of the SK game, each playing an action of ±1 from a softmax policy over its incentive to play +1.

    `SKReinforcementInteraction` updates the incentive `Q` with the memory `alpha` and draws the action `sigma` at the
    temperature `T`.

    Args:
        label:
            The label of the agent or the group.

        sigma:
            The initial action, -1 or 1, or one action per agent.

        Q:
            The initial incentive.

        alpha:
            The memory parameter, the weight of the latest reward.

        temperature:
            The exploration temperature `T` of the softmax policy.

        size:
            The number of agents, or `None` for one agent.

    Examples:
        >>> from skcomplex.systems.physics.criticality import LearningAgentElement
        >>> LearningAgentElement('agent').alpha
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        sigma: object = 1.0,
        Q: object = 0.0,
        alpha: Annotated[float, RealSet.open(0, oo)] = 0.5,
        temperature: Annotated[float, RealSet.open(0, oo)] = 0.5,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.sigma = sigma
        self.Q = Q
        self.alpha = alpha
        self.temperature = temperature


class SKReinforcementInteraction(Interaction):
    """Reinforcement learning of the agents of the SK game.

    Every step the incentive of an agent becomes `(1 - alpha) Q + alpha sum_j J_ij sigma_j`, with its memory `alpha`
    and the quenched couplings `J`, and the agent plays `sigma = 1` with the softmax probability `1 / (1 + exp(-2 Q /
    T))` at its temperature `T`, and `-1` otherwise. Both run in parallel. It is a market of traders who follow their
    past gains, players of a repeated game, or neurons that learn from their inputs.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the agents, whose values are `sigma`, `Q`, `alpha`, and `temperature`.

        couplings:
            The coupling matrix `J`, one row and one column per agent, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SKReinforcementInteraction
        >>> SKReinforcementInteraction('learning', target='agents').target
        'agents'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, couplings: npt.ArrayLike | None = None) -> None:
        super().__init__(label)
        self.target = target
        self.couplings = couplings

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the reinforcement of the incentive and the softmax draw of the action.

        Args:
            context:
                The namespace of the agents and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            reward = total(s.sigma, weights=self.couplings)
            up = 1 / (1 + exp(-2 * s.Q / s.temperature))
            self.laws_ = [
                UpdateLaw(s.Q, (1 - s.alpha) * s.Q + s.alpha * reward),
                UpdateLaw(s.sigma, where(uniform('u') < up, 1, -1)),
            ]


class SKGameSystem(BaseSystem):
    """The SK game of reinforcement learners.

    The rewards come from quenched Sherrington-Kirkpatrick couplings.

    Every agent plays `sigma_i = ±1` and earns `sigma_i sum_j J_ij sigma_j`, with symmetric Gaussian couplings of mean
    0, variance `1 / N`, and zero diagonal. It discounts its rewards with the memory `alpha` into the incentive `Q_i`
    and plays from a softmax policy at the temperature `T` (Lesmana et al., Eqs. 1-3). All agents start with zero
    incentives, so their first actions are random, and update in parallel. At low temperature the agents freeze into one
    of many ordered states with a high collective reward, and at high temperature they explore noisily with a low one.
    The system has the actions `agents.sigma`, the incentives `agents.Q`, and the coupling matrix `couplings` of its
    interaction `learning`. Its standard observable `reward` is the mean reward per agent.

    Args:
        n_agents:
            The number of agents `N`.

        alpha:
            The memory parameter.

        temperature:
            The exploration temperature of every agent.

        random_state:
            The seed or generator of the couplings and of the initial actions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SKGameSystem
        >>> couplings = SKGameSystem(10).prepare().system_.interactions[0].couplings
        >>> bool((couplings == couplings.T).all())
        True
    """

    field = 'physics/criticality'
    topics = ('spin glass', 'reinforcement learning', 'game', 'disorder', 'discrete map')
    references = ('Lesmana et al. 2026', 'Sherrington and Kirkpatrick 1975', 'Watkins and Dayan 1992')
    until = 200
    dt = 1

    @staticmethod
    def _get_reward(actions: Array, couplings: Array) -> float:
        """Return the mean reward per agent, `sum_ij J_ij sigma_i sigma_j / N`."""
        actions = np.asarray(actions, dtype=float)
        return float(actions @ couplings @ actions / actions.size)

    def __init__(
        self,
        n_agents: int = 256,
        *,
        alpha: float = 0.5,
        temperature: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.alpha = alpha
        self.temperature = temperature
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the agents and their learning, and the observable of the mean reward per agent.

        The reward reads the couplings, so the observable is built with them unless observables are given.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        upper = np.triu(rng.normal(0.0, 1.0 / np.sqrt(self.n_agents), (self.n_agents, self.n_agents)), 1)
        couplings = upper + upper.T
        agents = LearningAgentElement(
            'agents',
            sigma=rng.choice([-1.0, 1.0], size=self.n_agents),
            Q=0.0,
            alpha=self.alpha,
            temperature=self.temperature,
            size=self.n_agents,
        )
        learning = SKReinforcementInteraction('learning', target='agents', couplings=couplings)
        self.system_ = System(self.label, elements=[agents], interactions=[learning])
        if not self.observables_:
            reward = partial(SKGameSystem._get_reward, couplings=couplings)
            self.observables_ = (Observable('reward', group='agents', variable='sigma', reduce=reward),)
