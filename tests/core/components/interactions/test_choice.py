"""Test the `_choice` module of `skcomplex.core.components.interactions`."""

from typing import Annotated, ClassVar

import numpy as np
import pytest
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import ChoiceInteraction, FeedbackInteraction
from skcomplex.core.components.laws.utils import mean
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import AgentView, SystemView
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec

GLAUBER_ROOT = 0.9575040240772687


class ChooserView(AgentView):
    """View of a chooser, with its option."""

    option: float


class ChooserElement(Element):
    """Member with one discrete state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        option: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.option = option

    @classmethod
    def _get_view(cls) -> type[ChooserView]:
        """Return the view of a chooser, which types its option."""
        return ChooserView


class PayoffFeedback(FeedbackInteraction[SystemView]):
    """Reporter of a constant payoff of option 1 after every choice."""

    reports: ClassVar[tuple[str, ...]] = ('payoff',)

    def act(self, system: SystemView) -> None:
        """Report the payoff of option 1."""
        system.report('payoff', 5.0)


def _run_choice(choice: ChoiceInteraction, initial: np.ndarray, until: int, seed: int = 0) -> Trajectory:
    """Return the trajectory of a group of choosers bound to a choice."""
    group = GroupElement('members', element=ChooserElement, size=len(initial), values={'option': initial})
    system = System(elements=[group], interactions=[choice], time=DiscreteTime())
    return simulate(system, until=until, random_state=seed)


def _fixed(namespace: object, option: float) -> float:
    """Return a utility equal to the option."""
    return option


def _fixed_agent(agent: object, system: object, option: float) -> float:
    """Return a utility equal to the option, for agents."""
    return option


def _agent_field(agent: ChooserView, system: SystemView, option: float) -> float:
    """Return the option times the mean option of the group in the last step, read once per step."""
    cached = system.storage.get('mean')
    if not isinstance(cached, tuple) or cached[0] != system.time:
        cached = (system.time, float(np.mean([member.option for member in system.agents('target', ChooserView)])))
        system.storage['mean'] = cached
    return option * cached[1]


def _last_payoff(agent: object, system: SystemView, option: float) -> float:
    """Return the last payoff of option 1 that the feedback reported, zero for option 0 and before any report."""
    history = system.history('feedback.payoff', 1)
    return option * float(history[-1]) if len(history) else 0.0


def _frequencies(values: np.ndarray, count: int) -> np.ndarray:
    """Return the frequency of every option among integer values."""
    return np.bincount(values.astype(int).ravel(), minlength=count) / values.size


def test_choice_interaction_logit_reaches_mean_field():
    """Test that fully connected spins under the logit rule at intensity 2 reach the root of `m = tanh(2 m)`."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(-1, 1),
        utility=lambda s, option: option * mean(s.target.option),
        rule='logit',
        intensity=2.0,
    )
    initial = np.where(np.random.default_rng(0).random(1000) < 0.7, 1.0, -1.0)
    magnetization = _run_choice(choice, initial, 40).values('members.option').mean(axis=1)
    assert abs(magnetization[-10:].mean() - GLAUBER_ROOT) < 0.02
    assert abs(np.tanh(2 * GLAUBER_ROOT) - GLAUBER_ROOT) < 1e-12


def test_choice_interaction_agent_logit_reaches_mean_field():
    """Test that fully connected agents under the logit rule at intensity 2 reach the root of `m = tanh(2 m)`."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(-1, 1),
        utility=_agent_field,
        intensity=2.0,
        form='agent',
    )
    initial = np.where(np.random.default_rng(0).random(400) < 0.7, 1.0, -1.0)
    trajectory = _run_choice(choice, initial, 30)
    magnetization = [trajectory.agents('members', frame).values['option'].mean() for frame in range(20, 31)]
    assert abs(np.mean(magnetization) - GLAUBER_ROOT) < 0.03


def test_choice_interaction_logit_draws_softmax_probabilities():
    """Test that the logit rule draws every option with probability `exp(beta u) / sum`."""
    choice = ChoiceInteraction('choice', target='members', state='option', options=(0, 1, 2), utility=_fixed)
    values = _run_choice(choice, np.zeros(20000), 10).values('members.option')[1:]
    expected = np.exp([0.0, 1.0, 2.0]) / np.exp([0.0, 1.0, 2.0]).sum()
    assert np.allclose(_frequencies(values, 3), expected, atol=0.01)


@pytest.mark.parametrize('rule', ['metropolis', 'fermi'])
def test_choice_interaction_proposal_rules_reach_boltzmann_distribution(rule):
    """Test that the Metropolis and Fermi rules converge to the distribution `exp(beta u) / sum`."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1, 2),
        utility=_fixed,
        rule=rule,
        intensity=0.5,
    )
    values = _run_choice(choice, np.zeros(5000), 60).values('members.option')[-20:]
    expected = np.exp([0.0, 0.5, 1.0]) / np.exp([0.0, 0.5, 1.0]).sum()
    assert np.allclose(_frequencies(values, 3), expected, atol=0.01)


@pytest.mark.parametrize('rule', ['metropolis', 'fermi'])
def test_choice_interaction_agent_proposal_rules_reach_boltzmann_distribution(rule):
    """Test that agents under the Metropolis and Fermi rules reach the distribution `exp(beta u) / sum`."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1, 2),
        utility=_fixed_agent,
        rule=rule,
        intensity=0.5,
        form='agent',
    )
    trajectory = _run_choice(choice, np.zeros(2000), 40)
    values = np.concatenate([trajectory.agents('members', frame).values['option'] for frame in range(30, 41)])
    expected = np.exp([0.0, 0.5, 1.0]) / np.exp([0.0, 0.5, 1.0]).sum()
    assert np.allclose(_frequencies(values, 3), expected, atol=0.015)


@pytest.mark.parametrize('form', ['symbolic', 'agent'])
def test_choice_interaction_heat_bath_proposals_reach_boltzmann_distribution(form):
    """Test that Fermi acceptance of proposals of every option reaches the distribution `exp(beta u) / sum`."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1, 2),
        utility=_fixed if form == 'symbolic' else _fixed_agent,
        rule='fermi',
        intensity=0.5,
        form=form,
        proposal='all',
    )
    trajectory = _run_choice(choice, np.zeros(2000), 60)
    if form == 'symbolic':
        values = trajectory.values('members.option')[-20:]
    else:
        values = np.concatenate([trajectory.agents('members', frame).values['option'] for frame in range(40, 61)])
    expected = np.exp([0.0, 0.5, 1.0]) / np.exp([0.0, 0.5, 1.0]).sum()
    assert np.allclose(_frequencies(values, 3), expected, atol=0.015)


def test_choice_interaction_heat_bath_proposals_keep_the_option_at_equal_utilities():
    """Test that proposals of every option keep the current one in a third of the draws of three equal options."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1, 2),
        utility=_fixed_agent,
        rule='metropolis',
        intensity=0.0,
        form='agent',
        proposal='all',
    )
    values = _run_choice(choice, np.zeros(3000), 1).agents('members', 1).values['option']
    assert np.mean(values == 0) == pytest.approx(1 / 3, abs=0.03)


def test_choice_interaction_best_draws_among_the_largest_utilities():
    """Test that the best rule draws uniformly among the options of largest utility and never the others."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1, 2),
        utility=lambda s, option: min(option, 1),
        rule='best',
    )
    values = _run_choice(choice, np.zeros(10000), 3).values('members.option')[1:]
    assert np.allclose(_frequencies(values, 3), [0.0, 0.5, 0.5], atol=0.01)


def test_choice_interaction_agent_best_takes_the_largest_utility():
    """Test that agents under the best rule all take the option of largest utility."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1, 2),
        utility=_fixed_agent,
        rule='best',
        form='agent',
    )
    assert _run_choice(choice, np.zeros(50), 1).agents('members', 1).values['option'].tolist() == [2.0] * 50


def test_choice_interaction_activity_revises_a_share_of_members():
    """Test that with activity `p` a share `p` of the members revise in a step."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1),
        utility=_fixed,
        rule='best',
        activity=0.3,
    )
    values = _run_choice(choice, np.zeros(20000), 1).values('members.option')[1]
    assert abs(values.mean() - 0.3) < 0.015


def test_choice_interaction_agent_activity_revises_a_share_of_agents():
    """Test that with activity `p` a share `p` of the agents revise in a step."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1),
        utility=_fixed_agent,
        rule='best',
        activity=0.3,
        form='agent',
    )
    values = _run_choice(choice, np.zeros(5000), 1).agents('members', 1).values['option']
    assert abs(values.mean() - 0.3) < 0.03


def test_choice_interaction_random_sequential_sees_earlier_choices():
    """Test that under the random-sequential schedule every agent sees the choices made before it in the step."""

    def attend(agent: ChooserView, system: SystemView, option: float) -> float:
        """Return a utility of option 1 that is larger than that of option 0 while fewer than two agents hold it."""
        ones = sum(member.option for member in system.agents('target', ChooserView))
        return float(ones < 2) if option == 1 else 0.5

    sequential = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1),
        utility=attend,
        rule='best',
        schedule='random-sequential',
        form='agent',
    )
    synchronous = ChoiceInteraction(**{**sequential.get_params(), 'schedule': 'synchronous'})
    finals = [_run_choice(sequential, np.zeros(4), 1, seed).agents('members', 1).values['option'] for seed in range(5)]
    assert all(final.sum() == 2.0 for final in finals)
    assert _run_choice(synchronous, np.zeros(4), 1).agents('members', 1).values['option'].tolist() == [1.0] * 4


def test_choice_interaction_agent_utility_reads_the_history_of_the_system():
    """Test that an agent choice whose utility reads the payoffs reported in earlier steps takes the better option."""
    group = GroupElement('members', element=ChooserElement, size=30)
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1),
        utility=_last_payoff,
        intensity=4.0,
        form='agent',
        phase='choice',
    )
    schedule = PhasesScheduleSpec([PhaseSpec('choice'), PhaseSpec('report')])
    system = System(
        elements=[group],
        interactions=[choice, PayoffFeedback('feedback', phase='report')],
        time=DiscreteTime(schedule=schedule),
    )
    trajectory = simulate(system, until=5, random_state=0)
    assert 0.0 < trajectory.agents('members', 1).values['option'].mean() < 1.0
    assert trajectory.agents('members', 5).values['option'].mean() > 0.9


def test_choice_interaction_keeps_values_among_options():
    """Test that a proposal rule only ever sets the state to one of the options."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(-1, 1),
        utility=_fixed,
        rule='metropolis',
    )
    values = _run_choice(choice, np.ones(200), 10).values('members.option')
    assert set(np.unique(values).tolist()) <= {-1.0, 1.0}


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'rule': 'glauber'}, 'Unknown rule'),
        ({'schedule': 'ordered'}, 'Unknown schedule'),
        ({'form': 'table'}, 'Unknown form'),
        ({'schedule': 'random-sequential'}, 'only the agent form'),
        ({'schedule': 'member'}, 'only the agent form'),
        ({'options': (1,)}, 'at least 2 options'),
        ({'utility': 1.0}, 'must be callable'),
        ({'proposal': 'current'}, 'Unknown proposal'),
        ({'ties': 'nearest'}, 'Unknown ties'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that an unknown variant, a single option, or a utility that is not callable is rejected."""
    values = {'target': 'members', 'state': 'option', 'options': (0, 1), 'utility': _fixed, **arguments}
    with pytest.raises(ValidationError, match=message):
        ChoiceInteraction('choice', **values).prepare()


def _flat(namespace: object, option: float) -> float:
    """Return the same utility for every option."""
    return 0.0


def _flat_agent(agent: object, system: object, option: float) -> float:
    """Return the same utility for every option, for agents."""
    return 0.0


def test_choice_best_rule_takes_the_first_of_tied_options_by_the_first_ties():
    """Test that the best rule with the ties `'first'` takes the first of the options of equal utility in both forms."""
    symbolic = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(2, 0, 1),
        utility=_flat,
        rule='best',
        ties='first',
    )
    agent = ChoiceInteraction(**{**symbolic.get_params(), 'utility': _flat_agent, 'form': 'agent'})
    assert set(_run_choice(symbolic, np.zeros(200), 1).values('members.option')[1].tolist()) == {2.0}
    assert set(_run_choice(agent, np.zeros(200), 1).agents('members', 1).values['option'].tolist()) == {2.0}


@pytest.mark.parametrize('form', ['symbolic', 'agent'])
def test_choice_best_rule_draws_tied_options_uniformly_by_the_random_ties(form):
    """Test that the best rule with the ties `'random'` draws every option of equal utility equally often."""
    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1, 2),
        utility=_flat if form == 'symbolic' else _flat_agent,
        rule='best',
        form=form,
    )
    trajectory = _run_choice(choice, np.zeros(6000), 1)
    values = trajectory.values('members.option')[1] if form == 'symbolic' else trajectory.agents('members', 1)
    options = values if form == 'symbolic' else values.values['option']
    np.testing.assert_allclose(_frequencies(np.asarray(options), 3), [1 / 3] * 3, atol=0.03)


def test_choice_sequential_schedule_reads_the_choices_of_the_members_before():
    """Test that under the sequential schedule each member reads the choices of the members before it, in order."""

    def scarce(agent: ChooserView, system: SystemView, option: float) -> float:
        """Return a utility of option 1 that is positive while no agent holds it."""
        ones = sum(member.option for member in system.agents('target', ChooserView))
        return option * (0.5 - ones)

    choice = ChoiceInteraction(
        'choice',
        target='members',
        state='option',
        options=(0, 1),
        utility=scarce,
        rule='best',
        form='agent',
        schedule='sequential',
    )
    assert _run_choice(choice, np.zeros(3), 1).agents('members', 1).values['option'].tolist() == [1.0, 0.0, 0.0]
