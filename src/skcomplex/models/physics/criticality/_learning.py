"""Learning on a spin-glass landscape: the SK game of reinforcement learners on quenched random couplings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from functools import partial
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Law, Symbols, Update, total, uniform, where
from ....core.runtime import register_model
from ....core.system import Observable, System

type Array = npt.NDArray[np.float64]


def _reward(actions: Array, couplings: Array) -> float:
    """Return the mean reward per agent, `sum_ij J_ij sigma_i sigma_j / N`."""
    actions = np.asarray(actions, dtype=float)
    return float(actions @ couplings @ actions / actions.size)


def _observables(system: System) -> list[Observable]:
    """Return the observable of the mean reward per agent of the SK game."""
    values = cast('Mapping[str, Any]', system.elements[0].values)
    couplings = np.asarray(values['couplings'], dtype=float)
    return [Observable('reward', group='agents', variable='sigma', reduce=partial(_reward, couplings=couplings))]


class LearningAgent(Element):
    """Agent of the SK game that plays an action of ±1 from a softmax policy over its incentive to play +1.

    The incentive follows the reinforcement rule `Q(t + 1) = (1 - alpha) Q(t) + alpha sum_j J_ij sigma_j(t)`, the
    discounted reward of playing +1, and the agent plays `sigma = ±1` with probability `1 / (1 + exp(-+2 Q / T))`.

    Args:
        label:
            The label of the agent or the group.

        sigma:
            The initial action, -1 or 1.

        Q:
            The initial incentive.

        alpha:
            The memory parameter, the weight of the latest reward.

        temperature:
            The exploration temperature `T` of the softmax policy.

        couplings:
            The coupling matrix `J`, one row and one column per agent.

    Examples:
        >>> from skcomplex.models.physics.criticality import LearningAgent
        >>> LearningAgent('agent').alpha
        0.5
    """

    sigma: float = state(default=1.0, domain=Domain.INTEGER)
    Q: float = state(default=0.0)
    alpha: float = parameter(default=0.5, domain=Domain.POSITIVE)
    temperature: float = parameter(default=0.5, domain=Domain.POSITIVE)
    couplings: Any = None

    def laws(self, s: Symbols) -> list[Law]:
        """Return the reinforcement of the incentive and the softmax draw of the action, both in parallel.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        reward = total(s.sigma, weights=self.couplings)
        up = 1 / (1 + exp(-2 * s.Q / s.temperature))
        return [
            Update(s.Q, (1 - s.alpha) * s.Q + s.alpha * reward),
            Update(s.sigma, where(uniform('u') < up, 1, -1)),
        ]


@register_model(
    field='physics/criticality',
    tags=('spin glass', 'reinforcement learning', 'game', 'disorder', 'discrete map'),
    references=('Lesmana et al. 2026', 'Sherrington and Kirkpatrick 1975', 'Watkins and Dayan 1992'),
    defaults={'until': 200, 'dt': 1},
    observables=_observables,
)
def sk_game(
    n_agents: int = 256,
    *,
    alpha: float = 0.5,
    temperature: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the SK game, reinforcement learners whose rewards come from quenched Sherrington-Kirkpatrick couplings.

    Every agent plays `sigma_i = ±1` and earns `sigma_i sum_j J_ij sigma_j`, with symmetric Gaussian couplings of mean
    0, variance `1 / N`, and zero diagonal. It discounts its rewards with the memory `alpha` into the incentive `Q_i`
    and plays from a softmax policy at the temperature `T` (Lesmana et al., Eqs. 1-3). All agents start with zero
    incentives, so their first actions are random, and update in parallel. At low temperature the agents freeze into
    one of many ordered states with a high collective reward, and at high temperature they explore noisily with a low
    one.

    Args:
        n_agents:
            The number of agents `N`.

        alpha:
            The memory parameter.

        temperature:
            The exploration temperature of every agent.

        random_state:
            The seed or generator of the couplings and of the initial actions.

    Returns:
        system:
            The model, with the actions `agents.sigma`, the incentives `agents.Q`, and the coupling matrix in the values
            of the group. Its standard observable `reward` is the mean reward per agent.

    Examples:
        >>> from skcomplex.models.physics.criticality import sk_game
        >>> couplings = sk_game(10).elements[0].values['couplings']
        >>> bool((couplings == couplings.T).all())
        True
    """
    rng = np.random.default_rng(random_state)
    upper = np.triu(rng.normal(0.0, 1.0 / np.sqrt(n_agents), (n_agents, n_agents)), 1)
    agents = LearningAgent.group(
        'agents',
        n_agents,
        sigma=rng.choice([-1.0, 1.0], size=n_agents),
        Q=0.0,
        alpha=alpha,
        temperature=temperature,
        couplings=upper + upper.T,
    )
    return System(components=[agents])
