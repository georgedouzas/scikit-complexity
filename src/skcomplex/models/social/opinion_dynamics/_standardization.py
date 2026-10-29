"""Opinion dynamics models: complex agents with many internal configurations whose interactions standardize them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, oo
from sage.sets.real_set import RealSet
from scipy.special import expit

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model

MARKED = 2


def _standardization(spins: Array) -> float:
    """Return the share `q` of agents that show the characteristic, the mean of `|S_i|`."""
    return float(np.mean(np.abs(spins)))


def _spins_of(configurations: Array) -> Array:
    """Return the spin of each configuration: 1 for configuration 0, -1 for configuration 1, and 0 otherwise."""
    return np.select([configurations == 0, configurations == 1], [1.0, -1.0], 0.0)


class ComplexAgentElement(Element):
    """Agent with many internal configurations, of which two show a characteristic with the spin `1` or `-1`.

    Configuration 0 has the spin `S = 1`, configuration 1 the spin `S = -1`, and the configurations from 2 to `H + 1`
    the spin 0.

    Args:
        label:
            The label of the agent or group.

        configuration:
            The internal configuration, from 0 to `H + 1`.

        spin:
            The spin of the configuration.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import ComplexAgentElement
        >>> GroupElement(
        ...     'agents', element=ComplexAgentElement, size=10, values={'configuration': 5.0, 'spin': 0.0}
        ... ).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        configuration: Annotated[float, StateSpec(domain=NN)] = 2.0,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.configuration = configuration
        self.spin = spin

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the standardization moves every agent.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class StandardizationInteraction(Interaction):
    """Changes of configuration of random agents, accepted with the logistic probability of their gain in utility.

    The utility of agent `i` is `u_i = K / N sum_j |S_i| |S_j| + J / N sum_j S_i S_j` over the other agents. A random
    agent proposes a configuration drawn uniformly from the `H + 2` configurations and takes it with the probability
    `1 / (1 + exp(-(u_i' - u_i) / T))`, so the agents sample `P ~ exp(E / T)`. One unit of time is `N` proposals.

    Args:
        label:
            The label of the interaction.

        H:
            The number of configurations without the characteristic.

        **values:
            The values of the declared variables, by name:

            agents:
                The path of the group of agents.

            K:
                The coupling of the characteristics `|S_i|`, which standardizes the agents.

            J:
                The coupling of the spins `S_i`, which orders the standardized agents.

            T:
                The temperature, the randomness of the choices.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import StandardizationInteraction
        >>> StandardizationInteraction('standardization', agents='agents', H=100).H
        100
    """

    def __init__(
        self,
        label: str,
        *,
        agents: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        H: int = 50,
    ) -> None:
        super().__init__(label)
        self.agents = agents
        self.K = K
        self.J = J
        self.T = T
        self.H = H

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the configuration changes.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.sweep)]

    def sweep(self, model: ModelView) -> None:
        """Propose one change of configuration per agent on average, and accept each with the logistic probability.

        Args:
            model:
                The model.
        """
        K, J, T = (model.parameter(name) for name in ('K', 'J', 'T'))
        agents = model.agents(self.agents, AgentView)
        size = len(agents)
        configurations = np.array([agent.configuration for agent in agents], dtype=float)
        spins = _spins_of(configurations)
        marked, magnetization = float(np.abs(spins).sum()), float(spins.sum())
        rng = model.rng
        chosen, proposals = rng.integers(size, size=size), rng.integers(self.H + MARKED, size=size).astype(float)
        draws = rng.random(size)
        for index, proposal, new, draw in zip(chosen, proposals, _spins_of(proposals), draws, strict=True):
            old = spins[index]
            standardizing = K * (abs(new) - abs(old)) * (marked - abs(old))
            ordering = J * (new - old) * (magnetization - old)
            if draw >= expit((standardizing + ordering) / (size * T)):
                continue
            configurations[index], spins[index] = proposal, new
            marked += abs(new) - abs(old)
            magnetization += new - old
        for agent, configuration, spin in zip(agents, configurations, spins, strict=True):
            agent.configuration, agent.spin = float(configuration), float(spin)


@register_model(
    field='social/opinion_dynamics',
    tags=('agents', 'standardization', 'phase transition', 'large deviations', 'mean field'),
    references=('Bertin 2021', 'Bertin and Jensen 2019'),
    defaults={'until': 1000},
    observables=lambda system: [
        ObservableSpec('q', group='agents', variable='spin', reduce=_standardization),
        ObservableSpec('m', group='agents', variable='spin', reduce='mean'),
    ],
)
def standardized_agents(
    agents: int = 100,
    *,
    K: float = 10.0,
    J: float = 1.0,
    T: float = 0.5,
    H: int = 50,
    standardized: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return complex agents whose interactions make them standardized, all showing one characteristic, and ordered.

    Below the temperature `T_S = K / (2 ln(H / 2))` the most probable share of agents with the characteristic jumps
    from `q ~ 0` to `q ~ 1`, a discontinuous transition, and below `T_M = J` the standardized agents order, with a
    magnetization `m` of either sign. Both states are long-lived on either side of `T_S` in large groups, so the
    initial share `standardized` selects the branch that the run follows.

    Args:
        agents:
            The number of agents `N`.

        K:
            The coupling of the characteristics `|S_i|`, which standardizes the agents.

        J:
            The coupling of the spins `S_i`, which orders the standardized agents.

        T:
            The temperature, the randomness of the choices.

        H:
            The number of configurations without the characteristic.

        standardized:
            The initial share of agents with the characteristic, each with a random spin. The others start in random
            configurations without it.

        random_state:
            The seed or generator of the initial configurations.

    Returns:
        system:
            The model, with the group `agents`, the share `q` of agents with the characteristic, and the magnetization
            `m`, the mean spin.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import standardized_agents
        >>> standardized_agents(10, standardized=0.0).elements[0].values['spin'].tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """
    rng = np.random.default_rng(random_state)
    marked = rng.random(agents) < standardized
    configurations = np.where(marked, rng.integers(MARKED, size=agents), rng.integers(MARKED, H + MARKED, size=agents))
    configurations = configurations.astype(float)
    return System(
        elements=[
            GroupElement(
                'agents',
                element=ComplexAgentElement,
                size=agents,
                values={'configuration': configurations, 'spin': _spins_of(configurations)},
            ),
        ],
        interactions=[StandardizationInteraction('standardization', agents='agents', K=K, J=J, T=T, H=H)],
    )
