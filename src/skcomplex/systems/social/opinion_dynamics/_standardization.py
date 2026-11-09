"""Opinion dynamics systems: complex agents standardized by their interactions.

The agents have many internal configurations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import ChoiceInteraction, MapInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import AnyArray, Array


class ComplexAgentView(AgentView):
    """View of an agent of the standardization system.

    Its states and parameters are attributes.
    """

    configuration: float
    spin: float


class ComplexAgentElement(Element):
    """Agent with many internal configurations.

    Two of them show a characteristic with the spin `1` or `-1`. Configuration 0 has the spin `S = 1`, configuration 1
    the spin `S = -1`, and the configurations from 2 to `H + 1` the spin 0. In `StandardizedAgentsSystem`, a
    `MapInteraction` sets the spin after every change of configuration.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        configuration:
            The internal configuration, from 0 to `H + 1`.

        spin:
            The spin of the configuration.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import ComplexAgentElement
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

    @classmethod
    def _get_view(cls) -> type[ComplexAgentView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return ComplexAgentView


MARKED = 2


PHASE = 'default'


def _standardization(spins: AnyArray) -> float:
    """Return the share `q` of agents that show the characteristic.

    It is the mean of `|S_i|`.
    """
    return float(np.mean(np.abs(spins)))


class StandardizationInteraction(ChoiceInteraction):
    """Changes of configuration of random agents.

    They are accepted with the logistic probability of their gain in utility. The utility of agent `i` is
    `u_i = K / N sum_j |S_i| |S_j| + J / N sum_j S_i S_j` over the other agents. An agent proposes a configuration drawn
    uniformly from the `H + 2` configurations, its own included, and takes it with the probability
    `1 / (1 + exp(-beta (u_i' - u_i)))` of the rule `'fermi'`, so the agents sample `P ~ exp(beta E)`. The utilities
    read the counts of the configurations of the group, so every proposal sees the changes accepted before it. In
    `StandardizedAgentsSystem`, a phase of visits draws `N` agents with replacement every unit of time.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents.

        K:
            The coupling of the characteristics `|S_i|`, which standardizes the agents.

        J:
            The coupling of the spins `S_i`, which orders the standardized agents.

        H:
            The number of configurations without the characteristic.

        intensity:
            The intensity of choice `beta`, the inverse of the temperature.

        activity:
            The probability that a visited agent proposes a configuration.

        phase:
            The phase of the step the choice runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import StandardizationInteraction
        >>> StandardizationInteraction('standardization', target='agents', H=100).options[-1]
        101.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        H: int = 50,
        intensity: float = 2.0,
        activity: float = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state='configuration',
            options=tuple(float(option) for option in range(H + MARKED)),
            utility=self._get_utility,
            rule='fermi',
            intensity=intensity,
            schedule='member',
            activity=activity,
            form='agent',
            phase=phase,
            proposal='all',
        )
        self.K = K
        self.J = J
        self.H = H

    def _get_utility(self, agent: ComplexAgentView, system: SystemViewProtocol, option: float) -> float:
        """Return the utility of a configuration for an agent.

        It is read from the counts of the configurations of the others.
        """
        counts = system.composition('target', 'configuration')
        up, down = counts.get(0.0, 0), counts.get(1.0, 0)
        spin, new = float(StandardizationInteraction._spins_of(np.array(agent.configuration))), float(
            StandardizationInteraction._spins_of(np.array(option)),
        )
        standardizing = system.parameter('K') * abs(new) * (up + down - abs(spin))
        ordering = system.parameter('J') * new * (up - down - spin)
        return (standardizing + ordering) / sum(counts.values())

    @staticmethod
    def _spins_of(configurations: Array) -> Array:
        """Return the spin of each configuration.

        It is 1 for configuration 0, -1 for configuration 1, and 0 otherwise.
        """
        return np.select([configurations == 0, configurations == 1], [1.0, -1.0], 0.0)


class StandardizedAgentsSystem(BaseSystem):
    """Complex agents whose interactions make them standardized.

    They all show one characteristic, and are ordered. Below the temperature `T_S = K / (2 ln(H / 2))` the most probable
    share of agents with the characteristic jumps from `q ~ 0` to `q ~ 1`, a discontinuous transition, and below
    `T_M = J` the standardized agents order, with a magnetization `m` of either sign. Both states are long-lived on
    either side of `T_S` in large groups, so the initial share `standardized` selects the branch that the run follows.
    The system has the group `agents`, the share `q` of agents with the characteristic, and the magnetization `m`, the
    mean spin.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import StandardizedAgentsSystem
        >>> StandardizedAgentsSystem(10, standardized=0.0).prepare().system_.elements[0].values['spin'].tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """

    field = 'social/opinion_dynamics'
    topics = ('agents', 'standardization', 'phase transition', 'large deviations', 'mean field')
    references = ('Bertin 2021', 'Bertin and Jensen 2019')
    until = 1000
    observables = (
        ObservableSpec('q', group='agents', variable='spin', reduce=_standardization),
        ObservableSpec('m', group='agents', variable='spin', reduce='mean'),
    )

    def __init__(
        self,
        agents: int = 100,
        *,
        K: float = 10.0,
        J: float = 1.0,
        T: float = 0.5,
        H: int = 50,
        standardized: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.agents = agents
        self.K = K
        self.J = J
        self.T = T
        self.H = H
        self.standardized = standardized
        self.random_state = random_state

    @staticmethod
    def _spin_of(agent: ComplexAgentView, system: SystemViewProtocol) -> float:
        """Return the spin of the configuration of an agent."""
        return float(StandardizationInteraction._spins_of(np.array(agent.configuration)))

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        marked = rng.random(self.agents) < self.standardized
        configurations = np.where(
            marked,
            rng.integers(MARKED, size=self.agents),
            rng.integers(MARKED, self.H + MARKED, size=self.agents),
        )
        configurations = configurations.astype(float)
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'agents',
                    element=ComplexAgentElement,
                    size=self.agents,
                    values={
                        'configuration': configurations,
                        'spin': StandardizationInteraction._spins_of(configurations),
                    },
                ),
            ],
            interactions=[
                StandardizationInteraction(
                    'standardization',
                    target='agents',
                    K=self.K,
                    J=self.J,
                    H=self.H,
                    intensity=1 / self.T,
                    phase=PHASE,
                ),
                MapInteraction(
                    'spin',
                    target='agents',
                    state='spin',
                    function=self._spin_of,
                    schedule='member',
                    form='agent',
                    phase=PHASE,
                ),
            ],
            time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec(PHASE, activation='visits', groups=('agents',))])),
        )
