"""Test the `_payoff_learning` module of `skcomplex.systems.economics.common.learning`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import AgentView, SystemView
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.economics.common.learning import PayoffLearningInteraction
from skcomplex.systems.general.agents import ChoiceInteraction


class LearnerView(AgentView):
    """View of a learner, with its choice, the scores of its two options, and its value of option 1."""

    choice: float
    first: float
    second: float
    value: float


class LearnerElement(Element):
    """Member with a choice, the scores of two options, and a value of option 1."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        choice: Annotated[float, StateSpec()] = 0.0,
        first: Annotated[float, StateSpec()] = 0.0,
        second: Annotated[float, StateSpec()] = 0.0,
        value: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.choice = choice
        self.first = first
        self.second = second
        self.value = value

    @classmethod
    def _get_view(cls) -> type[LearnerView]:
        """Return the view of a learner, which types its states."""
        return LearnerView


class PoolElement(Element):
    """Member whose scores of several options are the components of one vector state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        scores: Annotated[object, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.scores = scores


def _run(interactions: list, values: dict, size: int, until: int) -> Trajectory:
    """Return the trajectory of a group of learners bound to interactions."""
    group = GroupElement('learners', element=LearnerElement, size=size, values=values)
    system = CompositeSystem(elements=[group], interactions=interactions, time=DiscreteTime())
    return simulate(system, until=until, random_state=0)


def _agent_payoff(agent: LearnerView, system: SystemView, index: int) -> float:
    """Return the value of option 1 for option 1 and one for option 0."""
    return agent.value if index == 1 else 1.0


def test_reinforcement_interaction_moves_scores_as_a_weighted_mean_of_payoffs():
    """Test that scores follow `s <- s + a (P - s)` towards constant payoffs."""
    reinforcement = PayoffLearningInteraction(
        'reinforcement',
        target='learners',
        scores=('first', 'second'),
        payoff=lambda s, index: s.target.value * index,
        rate=0.25,
    )
    trajectory = _run([reinforcement], {'value': np.array([2.0, 4.0])}, 2, 5)
    expected = 1 - 0.75 ** np.arange(6)
    assert np.allclose(trajectory.values('learners.first'), 0.0)
    assert np.allclose(trajectory.values('learners.second'), np.outer(expected, [2.0, 4.0]))


def test_reinforcement_interaction_agent_scores_steer_a_best_choice():
    """Test that agents choosing the option of best score learn to take the option of larger payoff."""
    reinforcement = PayoffLearningInteraction(
        'reinforcement',
        target='learners',
        scores=('first', 'second'),
        payoff=_agent_payoff,
        rate=0.5,
        form='agent',
    )
    choice = ChoiceInteraction(
        'choice',
        target='learners',
        state='choice',
        options=(0, 1),
        utility=lambda agent, system, option: agent.second if option == 1 else agent.first,
        rule='best',
        form='agent',
    )
    trajectory = _run([reinforcement, choice], {'value': np.array([0.5, 3.0])}, 2, 4)
    frame = trajectory.get_agents('learners', 4).values
    assert frame['choice'].tolist() == [0.0, 1.0]
    assert frame['second'].tolist() == pytest.approx([0.5 * (1 - 0.5**4), 3.0 * (1 - 0.5**4)])


def test_reinforcement_interaction_cumulative_rule_adds_the_payoffs():
    """Test that the cumulative rule adds the payoff of every option to its score in every step."""
    reinforcement = PayoffLearningInteraction(
        'reinforcement',
        target='learners',
        scores=('first', 'second'),
        payoff=lambda s, index: s.target.value * (index + 1),
        rule='cumulative',
    )
    trajectory = _run([reinforcement], {'value': np.array([2.0, 4.0])}, 2, 3)
    assert np.allclose(trajectory.values('learners.second')[-1], [12.0, 24.0])


def test_reinforcement_interaction_reinforces_the_components_of_a_vector_state():
    """Test that the agent form reinforces every component of a vector state of scores, one per option."""
    reinforcement = PayoffLearningInteraction(
        'reinforcement',
        target='pool',
        scores='scores',
        payoff=lambda agent, system, index: float(index),
        rate=0.5,
        form='agent',
    )
    group = GroupElement('pool', element=PoolElement, size=1, values={'scores': np.zeros((1, 4))})
    system = CompositeSystem(elements=[group], interactions=[reinforcement], time=DiscreteTime())
    trajectory = simulate(system, until=2, random_state=0)
    scores = np.array(trajectory.get_agents('pool', 2).values['scores'].tolist(), dtype=float)
    assert scores.ravel().tolist() == pytest.approx([0.0, 0.75, 1.5, 2.25])


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'form': 'table'}, 'The form of map'),
        ({'rule': 'median'}, 'Unknown rule'),
        ({'scores': 'first'}, 'sequence of state names'),
        ({'scores': ()}, 'sequence of state names'),
        ({'payoff': 1.0}, 'must be callable'),
        ({'schedule': 'parallel'}, 'The schedule of map'),
        ({'schedule': 'member'}, 'only the agent form'),
        ({'schedule': 'sequential'}, 'runs agent rules'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that an unknown form, missing scores, or a payoff that is not callable is rejected."""
    values = {'target': 'learners', 'scores': ('first', 'second'), 'payoff': lambda s, k: k, **arguments}
    with pytest.raises(ValidationError, match=message):
        PayoffLearningInteraction('reinforcement', **values).prepare()


def test_reinforcement_interaction_moves_only_the_score_of_the_chosen_option():
    """Test that with a chosen state only the score of the option a member chose moves towards its payoff."""
    reinforcement = PayoffLearningInteraction(
        'reinforcement',
        target='learners',
        scores=('first', 'second'),
        payoff=lambda s, index: s.target.value,
        rate=0.5,
        chosen='choice',
    )
    trajectory = _run([reinforcement], {'value': 2.0, 'choice': np.array([0.0, 1.0])}, 2, 1)
    assert trajectory.values('learners.first')[-1].tolist() == [1.0, 0.0]
    assert trajectory.values('learners.second')[-1].tolist() == [0.0, 1.0]


def _scarce_payoff(agent: LearnerView, system: SystemView, index: int) -> float:
    """Return a payoff of option 1 while no learner has scored it yet, and zero otherwise."""
    scored = sum(learner.second for learner in system.get_agents('target', LearnerView))
    return float(index == 1 and scored == 0)


@pytest.mark.parametrize(('schedule', 'expected'), [('synchronous', [1.0, 1.0]), ('sequential', [1.0, 0.0])])
def test_reinforcement_interaction_sequential_schedule_reads_the_scores_set_before(schedule, expected):
    """Test that sequential reinforcement reads the scores earlier agents set, while synchronous does not."""
    reinforcement = PayoffLearningInteraction(
        'reinforcement',
        target='learners',
        scores=('first', 'second'),
        payoff=_scarce_payoff,
        rule='cumulative',
        schedule=schedule,
        form='agent',
    )
    assert _run([reinforcement], {}, 2, 1).get_agents('learners', 1).values['second'].tolist() == expected


def test_reinforcement_interaction_members_reinforce_their_chosen_option_in_their_activation():
    """Test that under the member schedule every agent moves the score of its chosen option after its choice."""
    choice = ChoiceInteraction(
        'choice',
        target='learners',
        state='choice',
        options=(0, 1),
        utility=lambda agent, system, option: float(option),
        rule='best',
        schedule='member',
        form='agent',
    )
    reinforcement = PayoffLearningInteraction(
        'reinforcement',
        target='learners',
        scores=('first', 'second'),
        payoff=_agent_payoff,
        rate=0.5,
        chosen='choice',
        schedule='member',
        form='agent',
    )
    frame = _run([choice, reinforcement], {'value': 4.0}, 3, 2).get_agents('learners', 2).values
    assert frame['choice'].tolist() == [1.0, 1.0, 1.0]
    assert frame['first'].tolist() == [0.0, 0.0, 0.0]
    assert frame['second'].tolist() == [3.0, 3.0, 3.0]
