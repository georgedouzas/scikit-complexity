"""Test the `_transition` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import NN

from skcomplex.core.elements import Element, GroupElement, LatticeElement
from skcomplex.core.interactions import TransitionInteraction
from skcomplex.core.laws.utils import neighbors, where
from skcomplex.core.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

MEMBERS = 4000


class HolderElement(Element):
    """Holder of a discrete state `s` and an observed state `o`, both natural numbers."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[int, StateSpec(domain=NN)] = 0,
        o: Annotated[int, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.o = o


def _values(element, interactions, until, *, time=None, state='s', dt=1.0, backend=None):
    """Return the values of a state of the members at every unit of time."""
    system = System(elements=[element], interactions=interactions, time=time)
    simulation = Simulation(system, dt=dt, random_state=0, backend=backend).check().prepare()
    return simulation.run(until=until).trajectory_.values(f'members.{state}')


def _agent_values(element, interactions, until, state='s'):
    """Return the values of a state of the agents at the last step."""
    system = System(elements=[element], interactions=interactions, time=DiscreteTime())
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_
    return trajectory.agents('members', until).values[state]


def _group(initial=1.0, size=MEMBERS):
    """Return a group of holders."""
    return GroupElement('members', element=HolderElement, size=size, values={'s': initial})


def _ring(initial):
    """Return holders on the sites of a ring."""
    return LatticeElement('members', element=HolderElement, space=LatticeSpace(len(initial)), values={'s': initial})


def test_transition_at_a_rate_empties_the_source_exponentially():
    """Test that members leave the source value at rate 1, so its fraction decays as `exp(-t)`."""
    move = TransitionInteraction('decay', target='members', state='s', source=1, destination=0, rate=lambda s: 1.0)
    values = _values(_group(), [move], 2.0)
    assert set(np.unique(values)) == {0.0, 1.0}
    assert values[-1].mean() == pytest.approx(np.exp(-2.0), abs=0.02)


def test_transition_with_a_probability_moves_a_share_per_step():
    """Test that members move with probability 0.2 in every step of discrete time."""
    move = TransitionInteraction(
        'move',
        target='members',
        state='s',
        source=1,
        destination=2,
        probability=lambda s: 0.2,
    )
    values = _values(_group(), [move], 3, time=DiscreteTime())
    assert (values[1] == 2).mean() == pytest.approx(0.2, abs=0.02)
    assert (values[3] == 1).mean() == pytest.approx(0.8**3, abs=0.02)


def test_transition_over_neighbors_combines_their_probabilities():
    """Test that two active neighbours of probability 0.5 each move a member with probability 0.75."""
    initial = np.tile([1.0, 0.0], MEMBERS // 2)
    move = TransitionInteraction(
        'spread',
        target='members',
        state='s',
        probability=lambda s: where(s.target.other.s == 1, 0.5, 0),
        over='neighbors',
    )
    values = _values(_ring(initial), [move], 1, time=DiscreteTime())
    assert values[1][initial == 0].mean() == pytest.approx(0.75, abs=0.03)
    assert np.all(values[1][initial == 1] == 1)


def test_transition_over_neighbors_adds_their_rates():
    """Test that two active neighbours of rate 0.5 each move a member at rate 1."""
    initial = np.tile([1.0, 0.0], MEMBERS // 2)
    move = TransitionInteraction(
        'spread',
        target='members',
        state='s',
        rate=lambda s: where(s.target.other.s == 1, 0.5, 0),
        over='neighbors',
    )
    values = _values(_ring(initial), [move], 0.5, dt=0.5)
    assert values[-1][initial == 0].mean() == pytest.approx(1 - np.exp(-0.5), abs=0.03)


def test_transition_on_a_condition_spreads_a_threshold_cascade():
    """Test that inactive members with at least half of their neighbours active move in the next step."""
    initial = np.zeros(21)
    initial[10] = 1.0
    move = TransitionInteraction(
        'cascade',
        target='members',
        state='s',
        condition=lambda s: neighbors(s.target.other.s, reduce='mean') >= 0.5,
    )
    values = _values(_ring(initial), [move], 3, time=DiscreteTime())
    assert values.sum(axis=1).tolist() == [1, 3, 5, 7]


def test_transition_matrix_reaches_the_stationary_distribution():
    """Test that a two-state Markov chain reaches the stationary distribution of its transition matrix."""
    chain = TransitionInteraction('chain', target='members', state='s', matrix=lambda s: [[0.9, 0.1], [0.3, 0.7]])
    values = _values(_group(0.0), [chain], 40, time=DiscreteTime())
    assert values[-1].mean() == pytest.approx(0.25, abs=0.02)


def _sir(infected, after=None, form='symbolic'):
    """Return the infection and the recovery of a discrete SIR, which both move the state of the members."""
    probability = (lambda a, m: infected) if form == 'agent' else (lambda s: infected)
    recovery = (lambda a, m: 0.1) if form == 'agent' else (lambda s: 0.1)
    return [
        TransitionInteraction(
            'infection',
            target='members',
            state='s',
            source=0,
            destination=1,
            probability=probability,
            form=form,
        ),
        TransitionInteraction(
            'recovery',
            target='members',
            state='s',
            source=1,
            destination=2,
            probability=recovery,
            after=after,
            form=form,
        ),
    ]


def _sir_shares(steps, infected=0.2, recovered=0.1, *, in_turn=False):
    """Return the expected shares of susceptible, infected, and recovered members of the discrete SIR.

    In turn, the recovery reads the infections of the same step, as agent rules that run one after the other do.
    """
    susceptible, sick = 1.0, 0.0
    for _ in range(steps):
        newly = susceptible * infected
        susceptible = susceptible - newly
        sick = (sick + newly) * (1 - recovered) if in_turn else sick * (1 - recovered) + newly
    return np.asarray([susceptible, sick, 1 - susceptible - sick])


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()])
def test_two_transitions_of_one_state_run_a_discrete_sir(backend):
    """Test that an infection and a recovery of one state compose in every step into the shares of a discrete SIR."""
    values = _values(_group(0.0), _sir(0.2), 8, time=DiscreteTime(), backend=backend)
    shares = np.asarray([(values[-1] == value).mean() for value in (0, 1, 2)])
    np.testing.assert_allclose(shares, _sir_shares(8), atol=0.025)


def test_transition_after_a_residence_time_moves_no_member_earlier():
    """Test that members move from the source value only once they have held it for the residence time."""
    recovery = TransitionInteraction(
        'recovery',
        target='members',
        state='s',
        source=1,
        destination=2,
        probability=lambda s: 0.5,
        after=3,
    )
    values = _values(_group(1.0), [recovery], 6, time=DiscreteTime())
    assert np.all(values[:3] == 1)
    assert (values[3] == 2).mean() == pytest.approx(0.5, abs=0.03)
    assert (values[4] == 2).mean() == pytest.approx(0.75, abs=0.03)


def test_agent_transitions_run_a_discrete_sir():
    """Test that agent transitions of one state, which run one after the other, run the shares of a discrete SIR."""
    values = _agent_values(_group(0.0, size=2000), _sir(0.2, form='agent'), 8)
    shares = np.asarray([(values == value).mean() for value in (0, 1, 2)])
    np.testing.assert_allclose(shares, _sir_shares(8, in_turn=True), atol=0.035)


def test_agent_transition_after_a_residence_time_moves_no_agent_earlier():
    """Test that agents move from the source value only once they have held it for the residence time."""
    recovery = TransitionInteraction(
        'recovery',
        target='members',
        state='s',
        source=1,
        destination=2,
        condition=lambda a, m: True,
        after=2,
        form='agent',
    )
    group = _group(1.0, size=3)
    assert _agent_values(group, [recovery], 1).tolist() == [1.0, 1.0, 1.0]
    assert _agent_values(group, [recovery], 2).tolist() == [2.0, 2.0, 2.0]


def test_agent_transition_matrix_draws_rows_and_emissions():
    """Test that agents step by the rows of a matrix and draw their observed state from the emission matrix."""
    source = TransitionInteraction(
        'source',
        target='members',
        state='s',
        matrix=lambda a, m: [[0.0, 1.0], [0.0, 1.0]],
        emission=lambda a, m: [[1.0, 0.0], [0.2, 0.8]],
        observed='o',
        form='agent',
    )
    values = _agent_values(_group(0.0, size=2000), [source], 1, state='o')
    assert values.mean() == pytest.approx(0.8, abs=0.03)


def test_emission_matrix_draws_an_observation_given_the_moved_state():
    """Test that an emission matrix draws the observed state from the row the next value of the state selects."""
    source = TransitionInteraction(
        'source',
        target='members',
        state='s',
        matrix=lambda s: [[0.0, 1.0], [0.0, 1.0]],
        emission=lambda s: [[1.0, 0.0], [0.2, 0.8]],
        observed='o',
    )
    observed = _values(_group(0), [source], 1, time=DiscreteTime(), state='o')
    assert observed[-1].mean() == pytest.approx(0.8, abs=0.02)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({}, 'exactly one'),
        ({'rate': lambda s: 1.0, 'probability': lambda s: 0.5}, 'exactly one'),
        ({'rate': 1.0}, 'exactly one'),
        ({'rate': lambda s: 1.0, 'emission': lambda s: [[1.0]], 'observed': 'o'}, 'both with a matrix'),
        ({'matrix': lambda s: [[1.0]], 'emission': lambda s: [[1.0]]}, 'both with a matrix'),
        ({'rate': lambda s: 1.0, 'over': 'edges'}, 'Use one of'),
        ({'condition': lambda s: True, 'over': 'neighbors'}, 'only a rate or a probability'),
        ({'rate': lambda s: 1.0, 'source': 1, 'destination': 1}, 'distinct values'),
        ({'rate': lambda s: 1.0, 'after': 2}, 'residence time'),
        ({'probability': lambda s: 0.5, 'after': -1}, 'residence time'),
        ({'probability': lambda s: 0.5, 'form': 'mesa'}, 'form of transition'),
        ({'probability': lambda a, m: 0.5, 'over': 'neighbors', 'form': 'agent'}, 'symbolic form'),
        ({'probability': lambda s: 0.5, 'schedule': 'member'}, 'schedule of transition'),
        ({'probability': lambda a, m: 0.5, 'schedule': 'parallel', 'form': 'agent'}, 'schedule of transition'),
        ({'probability': lambda s: 0.5, 'over': 'cell'}, 'act on their cell'),
        ({'condition': lambda a, o, m: True, 'over': 'cell', 'after': 2, 'form': 'agent'}, 'act on their cell'),
    ],
)
def test_check_rejects_forms_it_cannot_take(arguments, match):
    """Test that a transition without one form, or with arguments its form ignores, raises an error."""
    with pytest.raises(ValidationError, match=match):
        TransitionInteraction('move', target='members', state='s', **arguments).check()


def _empty(agent, model):
    """Return whether no member holds the value 1."""
    return all(member.s != 1 for member in model.agents('target'))


@pytest.mark.parametrize(
    ('schedule', 'moved'),
    [('synchronous', 5), ('sequential', 1), ('random-sequential', 1), ('member', 1)],
)
def test_agent_transition_schedule_reads_the_values_set_before(schedule, moved):
    """Test that synchronous agents decide from the start of the step and the other schedules from earlier moves."""
    move = TransitionInteraction('move', target='members', state='s', condition=_empty, schedule=schedule, form='agent')
    assert _agent_values(_group(0.0, size=5), [move], 1).sum() == moved


def test_agent_transition_over_the_cell_moves_the_members_of_the_cell():
    """Test that a member holding 1 moves the members of its cell, and no member of another cell."""
    cells = [(0, 0), (0, 0), (0, 0), (2, 2)]
    group = GroupElement('members', element=HolderElement, size=4, values={'s': [1, 0, 0, 0]}, location=cells)
    spread = TransitionInteraction(
        'spread',
        target='members',
        state='s',
        condition=lambda agent, other, model: other.s == 1,
        over='cell',
        form='agent',
    )
    system = System(elements=[group], interactions=[spread], space=LatticeSpace((3, 3)), time=DiscreteTime())
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
    assert trajectory.agents('members', 1).values['s'].tolist() == [1, 1, 1, 0]


def test_agent_transition_member_schedule_lowers_to_a_member_rule():
    """Test that the member schedule lowers to a rule run by the members of the target."""
    move = TransitionInteraction(
        'move',
        target='members',
        state='s',
        probability=lambda a, m: 0.5,
        schedule='member',
        form='agent',
    )
    assert move.check().prepare().run(SimpleNamespace()).laws_[0].members == 'target'
