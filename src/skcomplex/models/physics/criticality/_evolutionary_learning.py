"""Evolutionary learning on a rugged landscape: the evolutionary Sherrington-Kirkpatrick game of Lesmana et al."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import tanh

from ....core.base import Domain, Expression, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Law, Symbols, Update, both, minimum, mod, total, uniform, where
from ....core.runtime import register_model
from ....core.system import Observable, System

type Array = npt.NDArray[np.float64]

TEMPERATURE_VISUAL = VisualSpec('traits', {'color_by': 'temperature', 'histogram_first': True})


def _softmax_action(incentive: Expression, temperature: Expression) -> Expression:
    """Return the action ±1 drawn with probability `exp(±Q / T) / (exp(Q / T) + exp(-Q / T))` of the softmax policy."""
    return where(uniform('action') < (1 + tanh(incentive / temperature)) / 2, 1, -1)


def _sk_couplings(n_agents: int, rng: np.random.Generator) -> Array:
    """Return symmetric Gaussian couplings of variance `1 / N` with a zero diagonal."""
    upper = np.triu(rng.normal(0.0, 1.0 / np.sqrt(n_agents), (n_agents, n_agents)), 1)
    return upper + upper.T


class Learner(Element):
    """Agent of the evolutionary Sherrington-Kirkpatrick game, who learns a binary action and may be replaced.

    The agent keeps an incentive `Q`, the value of action +1, and plays the action `s = ±1` by the softmax policy at
    its temperature `T`. Every step the incentive becomes `(1 - alpha) Q + alpha h` for the local field
    `h = Σ_j J_ij s_j`, and the fitness is `s Q`. When the agent evolves and its fitness is the least of the group,
    it is replaced by a newcomer with no incentive and a temperature drawn uniformly up to `max_temperature`, who
    learns from the field at once. The reward `s h` of the last actions is kept.

    Args:
        label:
            The label of the agent or the group.

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
        >>> from skcomplex.models.physics.criticality import Learner
        >>> Learner('agent').memory
        0.75
    """

    incentive: float = state(default=0.0)
    spin: float = state(default=1.0, domain=Domain.INTEGER)
    temperature: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    fitness: float = state(default=0.0)
    reward: float = state(default=0.0)
    memory: float = parameter(default=0.75, domain=Domain.NONNEGATIVE)
    max_temperature: float = parameter(default=2.0, domain=Domain.POSITIVE)
    couplings: Any = None
    evolving: bool = True
    settling: int = 2

    def laws(self, s: Symbols) -> list[Law]:
        """Return the replacement of the least fit agent, the learning of the incentive, and the softmax action.

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
            Update(s.incentive, learned),
            Update(s.temperature, temperature),
            Update(s.spin, action),
            Update(s.fitness, action * learned),
            Update(s.reward, s.spin * field),
        ]


def _observables(system: System) -> list[Observable]:
    """Return the mean reward, the mean temperature, and the least fitness of the agents."""
    del system
    return [
        Observable('mean_reward', group='agents', variable='reward'),
        Observable('mean_temperature', group='agents', variable='temperature'),
        Observable('least_fitness', group='agents', variable='fitness', reduce='min'),
    ]


@register_model(
    field='physics/criticality',
    tags=('self-organized criticality', 'reinforcement learning', 'spin glass', 'extremal dynamics', 'discrete map'),
    references=('Lesmana, Feng, Chen, and Lai 2026', 'Bak and Sneppen 1993', 'Sherrington and Kirkpatrick 1975'),
    defaults={'until': 3000, 'dt': 1},
    observables=_observables,
)
def evolutionary_sk_game(
    n_agents: int = 256,
    *,
    memory: float = 0.75,
    temperature: npt.ArrayLike | None = None,
    max_temperature: float = 2.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the evolutionary Sherrington-Kirkpatrick game, which self-organises to the edge of ergodicity breaking.

    After Lesmana et al. (2026, arXiv 2604.15669), `N` agents play ±1 on a Sherrington-Kirkpatrick landscape with
    symmetric Gaussian couplings of variance `1 / N`. Each learns the incentive of its actions with memory `alpha` and
    plays by a softmax policy at its own temperature, and the least fit agent is replaced, as in the Bak-Sneppen
    model, by a newcomer whose temperature is drawn uniformly on `[0, max_temperature]`. The temperatures evolve to the
    edge between ergodic and frozen learning, the replacement cascades have sizes distributed as `S^(-3/2)`, and the
    mean reward exceeds that of the game at any fixed temperature. With a fixed temperature, no agent is replaced and
    the model is the Sherrington-Kirkpatrick game of Galla and Farmer.

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

    Returns:
        system:
            The model, with the group `agents` of states `incentive`, `spin`, `temperature`, `fitness`, and `reward`,
            and the observables `mean_reward`, `mean_temperature`, and `least_fitness`.

    Examples:
        >>> from skcomplex.models.physics.criticality import evolutionary_sk_game
        >>> agents = evolutionary_sk_game(16).elements[0]
        >>> couplings = agents.values['couplings']
        >>> agents.size, bool((couplings == couplings.T).all()), float(couplings.trace())
        (16, True, 0.0)
    """
    rng = np.random.default_rng(random_state)
    couplings = _sk_couplings(n_agents, rng)
    spins = rng.choice([-1.0, 1.0], size=n_agents)
    evolving = temperature is None
    if temperature is None:
        temperatures = rng.uniform(0.0, max_temperature, n_agents)
    else:
        temperatures = np.broadcast_to(np.asarray(temperature, dtype=float), (n_agents,)).copy()
    agents = Learner.group(
        'agents',
        n_agents,
        visual=TEMPERATURE_VISUAL,
        spin=spins,
        temperature=temperatures,
        memory=memory,
        max_temperature=max_temperature,
        couplings=couplings,
        evolving=evolving,
    )
    return System(components=[agents])
