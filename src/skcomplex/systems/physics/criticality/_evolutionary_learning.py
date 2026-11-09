"""Evolutionary learning on a rugged landscape.

The module holds the evolutionary Sherrington-Kirkpatrick game of Lesmana et al.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, Expression, oo, tanh
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import both, minimum, mod, total, uniform, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array

TEMPERATURE_VISUAL = VisualSpec('histograms', {'color_by': 'temperature', 'histogram_first': True})


def _softmax_action(incentive: Expression, temperature: Expression) -> Expression:
    """Return the action ±1 drawn from the softmax policy.

    The probability of the action is `exp(±Q / T) / (exp(Q / T) + exp(-Q / T))`.
    """
    return where(uniform('action') < (1 + tanh(incentive / temperature)) / 2, 1, -1)


class LearnerElement(Element):
    """Agent of the evolutionary Sherrington-Kirkpatrick game.

    The agent learns a binary action and may be replaced.

    The agent keeps an incentive `Q`, the value of action +1, and plays the action `s = ±1` by the softmax policy at
    its temperature `T`. Every step the incentive becomes `(1 - alpha) Q + alpha h` for the local field
    `h = Σ_j J_ij s_j`, and the fitness is `s Q`. When the agent evolves and its fitness is the least of the group,
    it is replaced by a newcomer with no incentive and a temperature drawn uniformly up to `max_temperature`, who
    learns from the field at once. The reward `s h` of the last actions is kept.

    Args:
        label:
            The label of the agent or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        incentive:
            The initial incentive `Q`.

        spin:
            The initial action, -1 or 1.

        temperature:
            The initial temperature `T` of the softmax policy.

        fitness:
            The initial fitness `s Q`.

        reward:
            The initial reward `s h`.

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
        >>> from skcomplex.systems.physics.criticality import LearnerElement
        >>> LearnerElement('agent').memory
        0.75
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        incentive: Annotated[float, StateSpec()] = 0.0,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        temperature: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        fitness: Annotated[float, StateSpec()] = 0.0,
        reward: Annotated[float, StateSpec()] = 0.0,
        memory: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
        max_temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        couplings: npt.ArrayLike | None = None,
        evolving: bool = True,
        settling: int = 2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.incentive = incentive
        self.spin = spin
        self.temperature = temperature
        self.fitness = fitness
        self.reward = reward
        self.memory = memory
        self.max_temperature = max_temperature
        self.couplings = couplings
        self.evolving = evolving
        self.settling = settling

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the replacement of the least fit agent, the learning, and the action.

        The learning is that of the incentive, and the action follows the softmax policy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        field = total(s.spin, weights=self.couplings)
        incentive, temperature = s.incentive, s.temperature
        if self.evolving:
            least = both(s.fitness <= minimum(s.fitness), mod(s.t, self.settling + 1) < 1 / 2)
            incentive = where(least, 0, s.incentive)
            temperature = where(least, uniform('newcomer', 0, s.max_temperature), s.temperature)
        learned = (1 - s.memory) * incentive + s.memory * field
        action = _softmax_action(learned, temperature)
        return [
            UpdateLaw(s.incentive, learned),
            UpdateLaw(s.temperature, temperature),
            UpdateLaw(s.spin, action),
            UpdateLaw(s.fitness, action * learned),
            UpdateLaw(s.reward, s.spin * field),
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

    Examples:
        >>> from skcomplex.systems.physics.criticality import EvolutionarySKGameSystem
        >>> agents = EvolutionarySKGameSystem(16).prepare().system_.elements[0]
        >>> couplings = agents.values['couplings']
        >>> agents.size, bool((couplings == couplings.T).all()), float(couplings.trace())
        (16, True, 0.0)
    """

    field = 'physics/criticality'
    topics = ('self-organized criticality', 'reinforcement learning', 'spin glass', 'extremal dynamics', 'discrete map')
    references = ('Lesmana, Feng, Chen, and Lai 2026', 'Bak and Sneppen 1993', 'Sherrington and Kirkpatrick 1975')
    until = 3000
    dt = 1
    observables = (
        ObservableSpec('mean_reward', group='agents', variable='reward'),
        ObservableSpec('mean_temperature', group='agents', variable='temperature'),
        ObservableSpec('least_fitness', group='agents', variable='fitness', reduce='min'),
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
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.memory = memory
        self.temperature = temperature
        self.max_temperature = max_temperature
        self.random_state = random_state

    @staticmethod
    def _sk_couplings(n_agents: int, rng: np.random.Generator) -> Array:
        """Return symmetric Gaussian couplings of variance `1 / N` with a zero diagonal."""
        upper = np.triu(rng.normal(0.0, 1.0 / np.sqrt(n_agents), (n_agents, n_agents)), 1)
        return upper + upper.T

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        couplings = self._sk_couplings(self.n_agents, rng)
        spins = rng.choice([-1.0, 1.0], size=self.n_agents)
        evolving = self.temperature is None
        if self.temperature is None:
            temperatures = rng.uniform(0.0, self.max_temperature, self.n_agents)
        else:
            temperatures = np.broadcast_to(np.asarray(self.temperature, dtype=float), (self.n_agents,)).copy()
        agents = GroupElement(
            'agents',
            element=LearnerElement,
            size=self.n_agents,
            values={
                'spin': spins,
                'temperature': temperatures,
                'memory': self.memory,
                'max_temperature': self.max_temperature,
                'couplings': couplings,
                'evolving': evolving,
            },
            visual=TEMPERATURE_VISUAL,
        )
        return CompositeSystem(self.label, elements=[agents])
