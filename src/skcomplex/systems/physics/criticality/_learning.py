"""Learning on a spin-glass landscape.

The module holds the SK game of reinforcement learners on quenched random couplings.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import total, uniform, where
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array


class LearningAgentElement(Element):
    """Agent of the SK game that plays an action of ±1 from a softmax policy.

    The policy runs over its incentive to play +1.

    The incentive follows the reinforcement rule `Q(t + 1) = (1 - alpha) Q(t) + alpha sum_j J_ij sigma_j(t)`, the
    discounted reward of playing +1, and the agent plays `sigma = ±1` with probability `1 / (1 + exp(-+2 Q / T))`.

    Args:
        label:
            The label of the agent or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.physics.criticality import LearningAgentElement
        >>> LearningAgentElement('agent').alpha
        0.5
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        sigma: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        Q: Annotated[float, StateSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        couplings: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sigma = sigma
        self.Q = Q
        self.alpha = alpha
        self.temperature = temperature
        self.couplings = couplings

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reinforcement of the incentive and the softmax draw of the action.

        Both run in parallel.

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
    The system has the actions `agents.sigma`, the incentives `agents.Q`, and the coupling matrix in the values of the
    group. Its standard observable `reward` is the mean reward per agent.

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

    Examples:
        >>> from skcomplex.systems.physics.criticality import SKGameSystem
        >>> couplings = SKGameSystem(10).prepare().system_.elements[0].values['couplings']
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
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.alpha = alpha
        self.temperature = temperature
        self.random_state = random_state

    @staticmethod
    def _get_observables(system: CompositeSystem) -> list[ObservableSpec]:
        """Return the observable of the mean reward per agent of the SK game."""
        agents = system.elements[0]
        if not isinstance(agents, GroupElement) or agents.values is None:
            error_msg = 'The SK game needs its first element to be the group of agents with their couplings.'
            raise ValidationError(error_msg)
        couplings = np.asarray(agents.values['couplings'], dtype=float)
        return [
            ObservableSpec(
                'reward',
                group='agents',
                variable='sigma',
                reduce=partial(SKGameSystem._get_reward, couplings=couplings),
            ),
        ]

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the components are built from the model parameters.
        """
        settings = super().get_run_settings()
        settings['observables'] = tuple(self._get_observables(self._build_system()))
        return settings

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        upper = np.triu(rng.normal(0.0, 1.0 / np.sqrt(self.n_agents), (self.n_agents, self.n_agents)), 1)
        agents = GroupElement(
            'agents',
            element=LearningAgentElement,
            size=self.n_agents,
            values={
                'sigma': rng.choice([-1.0, 1.0], size=self.n_agents),
                'Q': 0.0,
                'alpha': self.alpha,
                'temperature': self.temperature,
                'couplings': upper + upper.T,
            },
        )
        return CompositeSystem(self.label, elements=[agents])
