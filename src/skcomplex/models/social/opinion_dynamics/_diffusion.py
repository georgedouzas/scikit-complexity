"""Diffusion models: beliefs and innovations that spread between neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MapInteraction, MovementInteraction, TransitionInteraction
from ....core.spaces import ContinuousSpace, LatticeSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....errors import ValidationError
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._adopter_view import AdopterView
from ._believer_view import BelieverView

N_LEVELS = 4
PASSIONATE = 3
INNOVATORS = 0.025
MODES = ('infection', 'influence')


def _shift(agent: AgentViewProtocol, model: ModelViewProtocol, size: float) -> Array:
    """Return the position one unit away from an agent in a random direction, wrapped on a square torus."""
    angle = model.rng.uniform(0, 2 * np.pi)
    position = np.asarray(agent.location, dtype=float) + np.array([np.cos(angle), np.sin(angle)])
    return np.mod(position, size)


def _learn(agent: BelieverView, model: ModelViewProtocol) -> Array:
    """Return the weights of an agent plus the influence of every neighbour on the level the neighbour cares at."""
    belief = np.array(agent.belief, dtype=np.float64)
    for neighbor in (item for item in agent.neighbors() if isinstance(item, BelieverView)):
        weights = np.asarray(neighbor.belief, dtype=np.float64)
        levels = np.flatnonzero(weights == weights.max())
        belief[levels[int(model.rng.integers(len(levels)))]] += agent.learning * neighbor.influence
    return belief


def _care(agent: BelieverView, model: ModelViewProtocol) -> float:
    """Return the level an agent cares at, the index of its largest weight."""
    return float(np.argmax(agent.belief))


class BelieverElement(Element):
    """Agent of the belief diffusion model, with the weights it gives to four levels of caring about an issue.

    The levels are not caring at all, caring less, caring a bit, and caring passionately. An agent cares at the level
    of its largest weight. In `influentials`, a `MapInteraction` of the vector of weights adds every step `learning`
    times the influence of every neighbour to the weight of the level the neighbour cares at, a tie broken at random,
    and a second one sets the level of caring.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        belief:
            The initial weights of the four levels.

        caring:
            The initial level of caring, the index of the largest weight.

        influence:
            The weight of the agent's own level in the beliefs of its neighbours.

        learning:
            The increment per neighbour and step.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import BelieverElement
        >>> group = GroupElement('people', element=BelieverElement, size=4, values={'belief': [1.0, 0.0, 0.0, 0.0]})
        >>> group.label, group.element.__name__, group.size
        ('people', 'BelieverElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'caring', 'color_range': (0.0, 3.0)}),
        *,
        belief: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        caring: Annotated[float, StateSpec(domain=NN)] = 0.0,
        influence: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        learning: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.belief = belief
        self.caring = caring
        self.influence = influence
        self.learning = learning

    @classmethod
    def _get_view(cls) -> type[BelieverView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return BelieverView


class AdopterElement(Element):
    """Agent of the threshold adoption model, who adopts an innovation for good.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        adopted:
            Whether the agent starts as an adopter.

        threshold:
            The percentage of adopters in the agent's social circle above which it adopts in the influence mode.

        reach:
            The distance within which other agents form an agent's social circle.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import AdopterElement
        >>> group = GroupElement('people', element=AdopterElement, size=10, values={'threshold': 50.0})
        >>> group.label, group.element.__name__, group.size
        ('people', 'AdopterElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'adopted', 'color_range': (0.0, 1.0)}),
        *,
        adopted: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        threshold: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        reach: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.adopted = adopted
        self.threshold = threshold
        self.reach = reach

    @classmethod
    def _get_view(cls) -> type[AdopterView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return AdopterView


class AdoptionInteraction(TransitionInteraction):
    """Synchronous adoption of the threshold model, where every agent adopts for good from its social circle.

    In the infection mode an agent adopts once anyone in its social circle has adopted. In the influence mode it
    adopts once the percentage of adopters in its circle exceeds its threshold, or in society when its circle is empty.
    Every agent decides from the adoptions at the start of the step.

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
        >>> from skcomplex.models.social.opinion_dynamics import AdoptionInteraction
        >>> AdoptionInteraction('adoption', target='people', mode='influence').check().mode
        'influence'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        mode: str = 'infection',
    ) -> None:
        super().__init__(
            label,
            target=target,
            state='adopted',
            source=0.0,
            destination=1.0,
            condition=self._adopts,
            form='agent',
        )
        self.mode = mode

    def _check(self) -> None:
        """Raise an error for an unknown mode.

        Raises:
            ValidationError:
                The mode is unknown.
        """
        super()._check()
        if self.mode not in MODES:
            error_msg = f'Unknown adoption mode {self.mode!r}. Use one of {", ".join(MODES)}.'
            raise ValidationError(error_msg)

    def _adopts(self, agent: AdopterView, model: ModelViewProtocol) -> bool:
        """Return whether an agent adopts from the adopters in its social circle, or in society without a circle."""
        circle = [item for item in agent.neighbors(agent.reach) if isinstance(item, AdopterView)]
        adopters = sum(item.adopted for item in circle)
        if self.mode == 'infection':
            return adopters > 0
        if circle:
            return 100 * adopters / len(circle) > agent.threshold
        society = [item for item in model.agents('target') if isinstance(item, AdopterView)]
        return 100 * sum(item.adopted for item in society) / len(society) > agent.threshold


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'grid', 'influentials'),
    references=('Chesney 2021',),
    defaults={'until': 20},
)
def influentials(
    width: int = 20,
    height: int = 20,
    *,
    n_influentials: int = 10,
    influence: float = 3.0,
    news_report: bool = False,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the belief diffusion model with influentials who care passionately and weigh more with their neighbours.

    Every cell of a torus holds one agent whose weights start uniform on 0 to 4. Every agent in turn, in a random
    order, learns from its neighbours and then sets its level of caring.

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

    Returns:
        system:
            The model, with the group `people`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import influentials
        >>> influentials(5, 4, n_influentials=2, random_state=0).elements[0].check().prepare().size_
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    belief = rng.integers(0, 5, (size, N_LEVELS)).astype(float)
    belief[:, PASSIONATE] += float(news_report)
    chosen = rng.choice(size, n_influentials, replace=False)
    belief[chosen, PASSIONATE] = belief[chosen].max(axis=1) + 1
    weights = np.ones(size)
    weights[chosen] = influence
    people = GroupElement(
        'people',
        element=BelieverElement,
        size=size,
        values={'belief': belief, 'caring': belief.argmax(axis=1).astype(float), 'influence': weights},
        location=[(column, row) for column in range(width) for row in range(height)],
    )
    learning = MapInteraction(
        'learning',
        target='people',
        state='belief',
        function=_learn,
        schedule='member',
        form='agent',
    )
    caring = MapInteraction('caring', target='people', state='caring', function=_care, schedule='member', form='agent')
    return System(
        elements=[people],
        interactions=[learning, caring],
        space=LatticeSpace((width, height), boundary='periodic', capacity=1),
    )


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'innovation', 'threshold'),
    references=('Granovetter 1978', 'Hamill and Gilbert 2016'),
    defaults={'until': 25},
)
def threshold_adoption(
    n_agents: int = 300,
    *,
    size: float = 175.0,
    reach: float = 15.0,
    mode: str = 'infection',
    thresholds: str = 'uniform',
    shifting: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the threshold adoption model in social circles, started by the 2.5% of agents with the lowest thresholds.

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

    Returns:
        system:
            The model, with the group `people`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import threshold_adoption
        >>> threshold_adoption(40, size=60.0, random_state=0).elements[0].check().prepare().size_
        40
    """
    rng = np.random.default_rng(random_state)
    drawn = rng.uniform(1, 100, n_agents) if thresholds == 'uniform' else np.clip(rng.normal(50, 50, n_agents), 1, 100)
    innovators = np.argsort(drawn)[: max(1, round(INNOVATORS * n_agents))]
    adopted = np.zeros(n_agents)
    adopted[innovators] = 1.0
    people = GroupElement(
        'people',
        element=AdopterElement,
        size=n_agents,
        values={'adopted': adopted, 'threshold': drawn, 'reach': reach},
        location=rng.uniform(0, size, (n_agents, 2)),
    )
    return System(
        elements=[people],
        interactions=[
            MovementInteraction(
                'shift',
                target='people',
                step=lambda agent, model: _shift(agent, model, size),
                probability=shifting,
            ),
            AdoptionInteraction('adoption', target='people', mode=mode),
        ],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )
