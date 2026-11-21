"""Diffusion systems: beliefs and innovations that spread between neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.agents import MovementInteraction

MODES = ('infection', 'influence')
N_LEVELS = 4
PASSIONATE = 3
INNOVATORS = 0.025


@runtime_checkable
class BelieverView(AgentViewProtocol, Protocol):
    """View of an agent of the influentials system.

    Its states and parameters are attributes.
    """

    belief: Array
    caring: float
    influence: float
    learning: float


class BelieverElement(Element):
    """Agent of the belief diffusion system.

    It holds the weights it gives to four levels of caring about an issue. The levels are not caring at all, caring
    less, caring a bit, and caring passionately. An agent cares at the level of its largest weight. In
    `InfluentialsSystem`, an agent rule adds every step `learning` times the influence of every neighbour to the weight
    of the level the neighbour cares at, a tie broken at random, and a second one sets the level of caring.

    Args:
        label:
            The label of the agent or group.

        belief:
            The initial weights of the four levels.

        caring:
            The initial level of caring, the index of the largest weight.

        influence:
            The weight of the agent's own level in the beliefs of its neighbours.

        learning:
            The increment per neighbour and step.

        values:
            Further values by name, such as the cell `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The grid whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import BelieverElement
        >>> group = BelieverElement('people', belief=[1.0, 0.0, 0.0, 0.0], size=4)
        >>> group.label, type(group).__name__, group.size
        ('people', 'BelieverElement', 4)
    """

    visual = VisualSpec('agents', {'color_by': 'caring', 'color_range': (0.0, 3.0)})

    def __init__(
        self,
        label: str,
        *,
        belief: npt.ArrayLike = 0.0,
        caring: Annotated[object, NN] = 0.0,
        influence: Annotated[object, RealSet.open(0, oo)] = 1.0,
        learning: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.belief = belief
        self.caring = caring
        self.influence = influence
        self.learning = learning


@runtime_checkable
class AdopterView(AgentViewProtocol, Protocol):
    """View of an agent of the threshold adoption system.

    Its states and parameters are attributes.
    """

    adopted: float
    threshold: float
    reach: float


class AdopterElement(Element):
    """Agent of the threshold adoption system, who adopts an innovation for good.

    Args:
        label:
            The label of the agent or group.

        adopted:
            Whether the agent starts as an adopter.

        threshold:
            The percentage of adopters in the agent's social circle above which it adopts in the influence mode.

        reach:
            The distance within which other agents form an agent's social circle.

        values:
            Further values by name, such as the position `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The plane whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import AdopterElement
        >>> group = AdopterElement('people', threshold=50.0, size=10)
        >>> group.label, type(group).__name__, group.size
        ('people', 'AdopterElement', 10)
    """

    visual = VisualSpec('agents', {'color_by': 'adopted', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        adopted: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        threshold: Annotated[object, RealSet.closed_open(0, oo)] = 50.0,
        reach: Annotated[object, RealSet.open(0, oo)] = 15.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.adopted = adopted
        self.threshold = threshold
        self.reach = reach


class AdoptionInteraction(Interaction):
    """Synchronous adoption of the threshold system.

    Every agent adopts for good from its social circle. In the infection mode an agent adopts once anyone in its social
    circle has adopted. In the influence mode it adopts once the percentage of adopters in its circle exceeds its
    threshold, or in society when its circle is empty. Every agent decides from the adoptions at the start of the step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of adopters.

        mode:
            `'infection'` or `'influence'`.

    Raises:
        ValidationError:
            The mode is unknown.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import AdoptionInteraction
        >>> AdoptionInteraction('adoption', target='people', mode='influence').mode
        'influence'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        mode: str = 'infection',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mode = mode

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The mode is unknown.
        """
        super()._check_arguments()
        if self.mode not in MODES:
            error_msg = f'Unknown adoption mode {self.mode!r}. Use one of {", ".join(MODES)}.'
            raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the adoptions of every step as the rule of the system.

        Args:
            context:
                The namespace of the adopters.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context, self.adopt)]

    def adopt(self, system: SystemViewProtocol) -> None:
        """Let every agent that has not adopted adopt, from the adoptions at the start of the step.

        Args:
            system:
                The system view of the adoptions.
        """
        agents = [agent for agent in system.get_agents('target') if isinstance(agent, AdopterView)]
        adopting = [agent for agent in agents if float(agent.adopted) == 0.0 and self._adopts(agent, system)]
        for agent in adopting:
            agent.adopted = 1.0

    def _adopts(self, agent: AdopterView, system: SystemViewProtocol) -> bool:
        """Return whether an agent adopts from the adopters in its social circle.

        Without a circle, the adopters are those in society.
        """
        circle = [item for item in agent.get_neighbors(agent.reach) if isinstance(item, AdopterView)]
        adopters = sum(item.adopted for item in circle)
        if self.mode == 'infection':
            return adopters > 0
        if circle:
            return 100 * adopters / len(circle) > agent.threshold
        society = [item for item in system.get_agents('target') if isinstance(item, AdopterView)]
        return 100 * sum(item.adopted for item in society) / len(society) > agent.threshold


class InfluentialsSystem(BaseSystem):
    """The belief diffusion system with influentials.

    The influentials care passionately and weigh more with their neighbours. Every cell of a torus holds one agent whose
    weights start uniform on 0 to 4. Every agent in turn, in a random order, learns from its neighbours and then sets
    its level of caring. The system has the group `people`.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        n_influentials:
            The number of influentials, chosen at random.

        influence:
            The influence of an influential, the others having influence 1.

        news_report:
            Whether a news report adds one to every agent's weight of caring passionately at the start.

        random_state:
            The seed or generator of the initial beliefs and of the influentials.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import InfluentialsSystem
        >>> InfluentialsSystem(5, 4, n_influentials=2, random_state=0).prepare().system_.elements[0].size
        20
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'grid', 'influentials')
    references = ('Chesney 2021',)
    until = 20

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        n_influentials: int = 10,
        influence: float = 3.0,
        news_report: bool = False,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.width = width
        self.height = height
        self.n_influentials = n_influentials
        self.influence = influence
        self.news_report = news_report
        self.random_state = random_state

    @staticmethod
    def _learn(agent: BelieverView, system: SystemViewProtocol) -> Array:
        """Return the weights of an agent plus the influence of every neighbour.

        The influence is on the level the neighbour cares at.
        """
        belief = np.array(agent.belief, dtype=np.float64)
        for neighbor in (item for item in agent.get_neighbors() if isinstance(item, BelieverView)):
            weights = np.asarray(neighbor.belief, dtype=np.float64)
            levels = np.flatnonzero(weights == weights.max())
            belief[levels[int(system.rng.integers(len(levels)))]] += agent.learning * neighbor.influence
        return belief

    @staticmethod
    def _update_belief(agent: BelieverView, system: SystemViewProtocol) -> None:
        """Add to the weights of an agent the influence of every neighbour."""
        agent.belief = InfluentialsSystem._learn(agent, system)

    @staticmethod
    def _update_care(agent: BelieverView, system: SystemViewProtocol) -> None:
        """Set the level an agent cares at, the index of its largest weight."""
        del system
        agent.caring = float(np.argmax(agent.belief))

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        size = self.width * self.height
        belief = rng.integers(0, 5, (size, N_LEVELS)).astype(float)
        belief[:, PASSIONATE] += float(self.news_report)
        chosen = rng.choice(size, self.n_influentials, replace=False)
        belief[chosen, PASSIONATE] = belief[chosen].max(axis=1) + 1
        weights = np.ones(size)
        weights[chosen] = self.influence
        space = LatticeSpace((self.width, self.height), boundary='periodic', capacity=1)
        x, y = np.array([(column, row) for column in range(self.width) for row in range(self.height)]).T
        people = BelieverElement(
            'people',
            belief=belief,
            caring=belief.argmax(axis=1).astype(float),
            influence=weights,
            values={'x': x, 'y': y},
            size=size,
            space=space,
        )
        learning = Interaction(
            'learning',
            roles={'target': 'people'},
            laws=lambda s: [UpdateLaw(s.target, self._update_belief)],
        )
        caring = Interaction(
            'caring',
            roles={'target': 'people'},
            laws=lambda s: [UpdateLaw(s.target, self._update_care)],
        )
        self.system_ = System(self.label, elements=[people], interactions=[learning, caring], space=space)


class ThresholdAdoptionSystem(BaseSystem):
    """The threshold adoption system in social circles.

    It is started by the 2.5% of agents with the lowest thresholds. The system has the group `people`.

    Args:
        n_agents:
            The number of agents, placed at random on a square torus.

        size:
            The side of the torus.

        reach:
            The distance within which other agents form an agent's social circle.

        mode:
            `'infection'` to adopt after one adopter in the circle, or `'influence'` to adopt above a threshold.

        thresholds:
            `'uniform'` for thresholds uniform on 1 to 100 percent, or `'normal'` for a normal distribution of mean 50
            and standard deviation 50 clipped to 1 to 100.

        shifting:
            The share of agents that move one unit in a random direction every step.

        random_state:
            The seed or generator of the positions and thresholds.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ThresholdAdoptionSystem
        >>> ThresholdAdoptionSystem(40, size=60.0, random_state=0).prepare().system_.elements[0].size
        40
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'innovation', 'threshold')
    references = ('Granovetter 1978', 'Hamill and Gilbert 2016')
    until = 25

    def __init__(
        self,
        n_agents: int = 300,
        *,
        size: float = 175.0,
        reach: float = 15.0,
        mode: str = 'infection',
        thresholds: str = 'uniform',
        shifting: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.size = size
        self.reach = reach
        self.mode = mode
        self.thresholds = thresholds
        self.shifting = shifting
        self.random_state = random_state

    @staticmethod
    def _shift(agent: AgentViewProtocol, system: SystemViewProtocol, size: float) -> Array:
        """Return the position one unit away from an agent in a random direction.

        It is wrapped on a square torus.
        """
        angle = system.rng.uniform(0, 2 * np.pi)
        position = np.asarray(agent.location, dtype=float) + np.array([np.cos(angle), np.sin(angle)])
        return np.mod(position, size)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        drawn = (
            rng.uniform(1, 100, self.n_agents)
            if self.thresholds == 'uniform'
            else np.clip(rng.normal(50, 50, self.n_agents), 1, 100)
        )
        innovators = np.argsort(drawn)[: max(1, round(INNOVATORS * self.n_agents))]
        adopted = np.zeros(self.n_agents)
        adopted[innovators] = 1.0
        space = ContinuousSpace(extent=(self.size, self.size), boundary='periodic')
        x, y = rng.uniform(0, self.size, (self.n_agents, 2)).T
        people = AdopterElement(
            'people',
            adopted=adopted,
            threshold=drawn,
            reach=self.reach,
            values={'x': x, 'y': y},
            size=self.n_agents,
            space=space,
        )
        self.system_ = System(
            self.label,
            elements=[people],
            interactions=[
                MovementInteraction(
                    'shift',
                    target='people',
                    step=lambda agent, system: self._shift(agent, system, self.size),
                    probability=self.shifting,
                ),
                AdoptionInteraction('adoption', target='people', mode=self.mode),
            ],
            space=space,
        )
