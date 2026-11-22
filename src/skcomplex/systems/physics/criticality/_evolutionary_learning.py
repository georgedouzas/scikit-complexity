"""Evolutionary learning on a rugged landscape.

The module holds the evolutionary Sherrington-Kirkpatrick game of Lesmana et al.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo, tanh
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import both, minimum, mod, total, uniform, where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role


class LearnerElement(Element):
    """Agents of the evolutionary Sherrington-Kirkpatrick game.

    Every agent keeps an incentive `Q`, the value of action +1, an action `s = ±1`, a temperature `T`, a fitness `s Q`,
    and the reward `s h` of its last action.

    Args:
        label:
            The label of the agent or the group.

        incentive:
            The initial incentive `Q`, or one per agent.

        spin:
            The initial action, -1 or 1, or one per agent.

        temperature:
            The initial temperature `T` of the softmax policy, or one per agent.

        fitness:
            The initial fitness `s Q`, or one per agent.

        reward:
            The initial reward `s h`, or one per agent.

        size:
            The number of agents, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.criticality import LearnerElement
        >>> LearnerElement('agent').temperature
        1.0
    """

    visual = VisualSpec('histograms', {'color_by': 'temperature', 'histogram_first': True})

    def __init__(
        self,
        label: str,
        *,
        incentive: object = 0.0,
        spin: object = 1.0,
        temperature: object = 1.0,
        fitness: object = 0.0,
        reward: object = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.incentive = incentive
        self.spin = spin
        self.temperature = temperature
        self.fitness = fitness
        self.reward = reward


class SKLearningInteraction(Interaction):
    """Learning of the agents of the Sherrington-Kirkpatrick game, with the replacement of the least fit.

    Every step the incentive of an agent becomes `(1 - alpha) Q + alpha h` for the local field `h = Σ_j J_ij s_j`, the
    agent plays the action `s = ±1` by the softmax policy at its temperature `T`, and its fitness is `s Q`. When the
    agents evolve, the least fit one is replaced by a newcomer with no incentive and a temperature drawn uniformly up to
    `max_temperature`, who learns from the field at once. The reward `s h` of the last actions is kept.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the agents.

        memory:
            The weight `alpha` of the latest field in the incentive, so small values keep a long memory.

        max_temperature:
            The largest temperature of a newcomer.

        couplings:
            The coupling matrix `J`, one row and one column per agent.

        evolving:
            Whether the least fit agent is replaced.

        settling:
            The number of steps of learning alone after every replacement, so replacements happen when `t` is a
            multiple of `settling + 1`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SKLearningInteraction
        >>> SKLearningInteraction('learning', target='agents').memory
        0.75
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        memory: Annotated[float, RealSet.closed_open(0, oo)] = 0.75,
        max_temperature: Annotated[float, RealSet.open(0, oo)] = 2.0,
        couplings: npt.ArrayLike | None = None,
        evolving: bool = True,
        settling: int = 2,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.memory = memory
        self.max_temperature = max_temperature
        self.couplings = couplings
        self.evolving = evolving
        self.settling = settling

    @staticmethod
    def _get_softmax_action(incentive: Expression, temperature: Expression) -> Expression:
        """Return the action ±1 drawn from the softmax policy.

        The probability of the action is `exp(±Q / T) / (exp(Q / T) + exp(-Q / T))`.
        """
        return where(uniform('action') < (1 + tanh(incentive / temperature)) / 2, 1, -1)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the replacement of the least fit agent, the learning, and the action.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, agents = context, context.target
        field = total(agents.spin, weights=self.couplings)
        incentive, temperature = agents.incentive, agents.temperature
        if self.evolving:
            least = both(agents.fitness <= minimum(agents.fitness), mod(s.t, self.settling + 1) < 1 / 2)
            incentive = where(least, 0, agents.incentive)
            temperature = where(least, uniform('newcomer', 0, s.max_temperature), agents.temperature)
        learned = (1 - s.memory) * incentive + s.memory * field
        action = self._get_softmax_action(learned, temperature)
        self.laws_ = [
            UpdateLaw(agents.incentive, learned),
            UpdateLaw(agents.temperature, temperature),
            UpdateLaw(agents.spin, action),
            UpdateLaw(agents.fitness, action * learned),
            UpdateLaw(agents.reward, agents.spin * field),
        ]


class EvolutionarySKGameSystem(BaseSystem):
    """The evolutionary Sherrington-Kirkpatrick game.

    The game self-organises to the edge of ergodicity breaking.

    After Lesmana et al. (2026, arXiv 2604.15669), `N` agents play ±1 on a Sherrington-Kirkpatrick landscape with
    symmetric Gaussian couplings of variance `1 / N`. Each learns the incentive of its actions with memory `alpha` and
    plays by a softmax policy at its own temperature, and the least fit agent is replaced, as in the Bak-Sneppen system,
    by a newcomer whose temperature is drawn uniformly on `[0, max_temperature]`. The temperatures evolve to the edge
    between ergodic and frozen learning, the replacement cascades have sizes distributed as `S^(-3/2)`, and the mean
    reward exceeds that of the game at any fixed temperature. With a fixed temperature, no agent is replaced and the
    system is the Sherrington-Kirkpatrick game of Galla and Farmer. The system has the group `agents` of states
    `incentive`, `spin`, `temperature`, `fitness`, and `reward`, and the observables `mean_reward`, `mean_temperature`,
    and `least_fitness`.

    Args:
        n_agents:
            The number of agents `N`.

        memory:
            The weight `alpha` of the latest field in the incentive.

        temperature:
            The temperature of every agent, or one per agent, of the game without replacement, or `None` for the
            evolving game.

        max_temperature:
            The largest temperature of a newcomer, and of the initial temperatures drawn uniformly.

        random_state:
            The seed or generator of the couplings, the initial actions, and the initial temperatures.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.criticality import EvolutionarySKGameSystem
        >>> system = EvolutionarySKGameSystem(16).prepare().system_
        >>> couplings = system.interactions[0].couplings
        >>> system.elements[0].size, bool((couplings == couplings.T).all()), float(couplings.trace())
        (16, True, 0.0)
    """

    field = 'physics/criticality'
    topics = ('self-organized criticality', 'reinforcement learning', 'spin glass', 'extremal dynamics', 'discrete map')
    references = ('Lesmana, Feng, Chen, and Lai 2026', 'Bak and Sneppen 1993', 'Sherrington and Kirkpatrick 1975')
    until = 3000
    dt = 1
    observables = (
        Observable('mean_reward', group='agents', variable='reward'),
        Observable('mean_temperature', group='agents', variable='temperature'),
        Observable('least_fitness', group='agents', variable='fitness', reduce='min'),
    )

    def __init__(
        self,
        n_agents: int = 256,
        *,
        memory: float = 0.75,
        temperature: npt.ArrayLike | None = None,
        max_temperature: float = 2.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.memory = memory
        self.temperature = temperature
        self.max_temperature = max_temperature
        self.random_state = random_state

    @staticmethod
    def _get_sk_couplings(n_agents: int, rng: np.random.Generator) -> Array:
        """Return symmetric Gaussian couplings of variance `1 / N` with a zero diagonal."""
        upper = np.triu(rng.normal(0.0, 1.0 / np.sqrt(n_agents), (n_agents, n_agents)), 1)
        return upper + upper.T

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the agents and their learning, built from the model parameters.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        couplings = self._get_sk_couplings(self.n_agents, rng)
        spins = rng.choice([-1.0, 1.0], size=self.n_agents)
        evolving = self.temperature is None
        if self.temperature is None:
            temperatures = rng.uniform(0.0, self.max_temperature, self.n_agents)
        else:
            temperatures = np.broadcast_to(np.asarray(self.temperature, dtype=float), (self.n_agents,)).copy()
        agents = LearnerElement('agents', spin=spins, temperature=temperatures, size=self.n_agents)
        learning = SKLearningInteraction(
            'learning',
            target='agents',
            memory=self.memory,
            max_temperature=self.max_temperature,
            couplings=couplings,
            evolving=evolving,
        )
        self.system_ = System(self.label, elements=[agents], interactions=[learning])
