"""Opinion dynamics models: complex agents with many internal configurations whose interactions standardize them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from ....core.backends import ModelView
from ....core.base import Domain, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.system import Observable, System

type Array = npt.NDArray[np.float64]

MARKED = 2


def _standardization(spins: Array) -> float:
    """Return the share `q` of agents that show the characteristic, the mean of `|S_i|`."""
    return float(np.mean(np.abs(spins)))


def _spins_of(configurations: Array) -> Array:
    """Return the spin of each configuration: 1 for configuration 0, -1 for configuration 1, and 0 otherwise."""
    return np.select([configurations == 0, configurations == 1], [1.0, -1.0], 0.0)


class ComplexAgent(Element):
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
        >>> from skcomplex.models.social.opinion_dynamics import ComplexAgent
        >>> ComplexAgent.group('agents', 10, configuration=5.0, spin=0.0).size
        10
    """

    configuration: float = state(default=2.0, domain=Domain.NONNEGATIVE_INTEGER)
    spin: float = state(default=0.0, domain=Domain.INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return no laws, since the standardization moves every agent.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class Standardization(Interaction):
    """Changes of configuration of random agents, accepted with the logistic probability of their gain in utility.

    The utility of agent `i` is `u_i = K / N sum_j |S_i| |S_j| + J / N sum_j S_i S_j` over the other agents. A random
    agent proposes a configuration drawn uniformly from the `H + 2` configurations and takes it with the probability
    `1 / (1 + exp(-(u_i' - u_i) / T))`, so the agents sample `P ~ exp(E / T)`. One unit of time is `N` proposals.

    Args:
        label:
            The label of the interaction.

        agents:
            The path of the group of agents.

        K:
            The coupling of the characteristics `|S_i|`, which standardizes the agents.

        J:
            The coupling of the spins `S_i`, which orders the standardized agents.

        T:
            The temperature, the randomness of the choices.

        H:
            The number of configurations without the characteristic.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Standardization
        >>> Standardization('standardization', agents='agents', H=100).H
        100
    """

    agents: str = role()
    K: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    J: float = parameter(default=1.0, domain=Domain.REAL)
    T: float = parameter(default=0.5, domain=Domain.POSITIVE)
    H: int = 50

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of the configuration changes.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.sweep)]

    def sweep(self, model: ModelView) -> None:
        """Propose one change of configuration per agent on average, and accept each with the logistic probability.

        Args:
            model:
                The model.
        """
        K, J, T = (model.parameter(name) for name in ('K', 'J', 'T'))
        agents = model.agents(self.agents)
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
        Observable('q', group='agents', variable='spin', reduce=_standardization),
        Observable('m', group='agents', variable='spin', reduce='mean'),
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
        components=[ComplexAgent.group('agents', agents, configuration=configurations, spin=_spins_of(configurations))],
        interactions=[Standardization('standardization', agents='agents', K=K, J=J, T=T, H=H)],
    )
