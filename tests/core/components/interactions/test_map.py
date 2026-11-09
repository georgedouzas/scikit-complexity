"""Test the `_map` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.laws.utils import mean, normal, sign
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec

RATE = 3.2


class ValueElement(Element):
    """Element with a value `x`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


def _values(elements, interactions, until, path='value.x'):
    """Return the values of a state at every step of a discrete-time system."""
    system = CompositeSystem(elements=elements, interactions=interactions, time=DiscreteTime())
    trajectory = simulate(system, until=until, random_state=0)
    return trajectory.values(path)


def _agent_values(group, interaction, until):
    """Return the values of the agents of a group at the last step."""
    system = CompositeSystem(elements=[group], interactions=[interaction], time=DiscreteTime())
    trajectory = simulate(system, until=until, random_state=0)
    return trajectory.get_agents(group.label, until).values['x']


def test_logistic_map_settles_on_the_closed_form_two_cycle():
    """Test that the logistic map at `r = 3.2` settles on its closed-form 2-cycle."""
    logistic = MapInteraction('map', target='value', state='x', function=lambda s: RATE * s.target.x * (1 - s.target.x))
    values = _values([ValueElement('value')], [logistic], 400)
    root = np.sqrt((RATE + 1) * (RATE - 3))
    cycle = sorted([(RATE + 1 - root) / (2 * RATE), (RATE + 1 + root) / (2 * RATE)])
    np.testing.assert_allclose(sorted(values[-2:]), cycle, rtol=1e-9)
    assert values[-1] != pytest.approx(values[-2])


def test_increment_mode_adds_the_function_to_every_member():
    """Test that the increment mode adds the function to the state of every member of a group."""
    group = GroupElement('value', element=ValueElement, size=3, values={'x': [0.0, 1.0, 2.0]})
    increment = MapInteraction('step', target='value', state='x', function=lambda s: 0.5, mode='increment')
    values = _values([group], [increment], 4)
    np.testing.assert_allclose(values[-1], [2.0, 3.0, 4.0])


def test_map_reads_its_sources():
    """Test that a map reads the states of the elements its sources bind."""
    copy = MapInteraction('copy', target='value', state='x', function=lambda s: 2 * s.sources[0].x, sources=['input'])
    values = _values([ValueElement('value', x=0.0), ValueElement('input', x=0.25)], [copy], 2)
    assert values.tolist() == [0.0, 0.5, 0.5]


def test_map_draws_one_value_per_member():
    """Test that a map with a draw gives every member of a group its own value."""
    group = GroupElement('value', element=ValueElement, size=200, values={'x': 0.0})
    noise = MapInteraction('noise', target='value', state='x', function=lambda s: normal('kick'), mode='increment')
    values = _values([group], [noise], 100)
    assert values[-1].var() == pytest.approx(100.0, rel=0.25)


def test_increment_maps_of_one_state_add():
    """Test that two increment maps of one state add their increments in every step."""
    group = GroupElement('value', element=ValueElement, size=2, values={'x': [0.0, 1.0]})
    first = MapInteraction('first', target='value', state='x', function=lambda s: 1.0, mode='increment')
    second = MapInteraction('second', target='value', state='x', function=lambda s: 2.0, mode='increment')
    assert _values([group], [first, second], 2)[-1].tolist() == [6.0, 7.0]


def _hopfield_field(s):
    """Return the Hebbian field of two stored patterns `a` and `b` on a neuron."""
    neuron = s.target
    return neuron.a * mean(neuron.a * neuron.x) + neuron.b * mean(neuron.b * neuron.x)


def _hopfield(schedule, backend, start, patterns, until):
    """Return the states of a Hopfield network of two patterns under a schedule."""
    group = GroupElement(
        'neurons',
        element=NeuronElement,
        size=start.size,
        values={'x': start, 'a': patterns[0], 'b': patterns[1]},
    )
    network = MapInteraction(
        'hopfield',
        target='neurons',
        state='x',
        function=lambda s: sign(_hopfield_field(s)),
        schedule=schedule,
    )
    system = CompositeSystem(elements=[group], interactions=[network], time=DiscreteTime())
    trajectory = simulate(system, until=until, random_state=0, backend=backend)
    return trajectory.values('neurons.x')


class NeuronElement(Element):
    """Neuron of a Hopfield network with a state `x` and two stored patterns `a` and `b`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec()] = 1.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.b = b


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()])
def test_sequential_hopfield_map_matches_the_asynchronous_update(backend):
    """Test that a sequential Hopfield map updates the neurons one after the other, each reading those before it."""
    rng = np.random.default_rng(3)
    patterns = rng.choice([-1.0, 1.0], (2, 40))
    start = rng.choice([-1.0, 1.0], 40)
    weights = patterns.T @ patterns / start.size
    expected = [start.copy()]
    for _ in range(4):
        state = expected[-1].copy()
        for index in range(state.size):
            state[index] = np.sign(weights[index] @ state)
        expected.append(state)
    sequential = _hopfield('sequential', backend, start, patterns, 4)
    synchronous = _hopfield('synchronous', backend, start, patterns, 4)
    np.testing.assert_array_equal(sequential, expected)
    np.testing.assert_array_equal(synchronous[1], np.sign(weights @ start))
    assert np.any(sequential[1] != synchronous[1])


def test_random_sequential_hopfield_map_reaches_a_stored_pattern():
    """Test that a random-sequential Hopfield map from a corrupted pattern recovers the pattern."""
    rng = np.random.default_rng(4)
    patterns = rng.choice([-1.0, 1.0], (2, 60))
    start = patterns[0].copy()
    start[rng.choice(60, 12, replace=False)] *= -1
    values = _hopfield('random-sequential', NumpyBackend(), start, patterns, 3)
    np.testing.assert_array_equal(values[-1], patterns[0])


@pytest.mark.parametrize(
    ('schedule', 'expected'),
    [
        ('synchronous', [4.0, 4.0, 4.0]),
        ('sequential', [4.0, 7.0, 13.0]),
        ('random-sequential', [4.0, 7.0, 13.0]),
        ('member', [4.0, 7.0, 13.0]),
    ],
)
def test_agent_map_updates_agents_by_the_schedule(schedule, expected):
    """Test that agents under a schedule read the values of the start of the step, or those set before them."""

    def following(agent, system):
        """Return one plus the sum of the values of the agents."""
        return 1.0 + sum(member.x for member in system.get_agents('target'))

    group = GroupElement('value', element=ValueElement, size=3, values={'x': 1.0})
    agents = MapInteraction('map', target='value', state='x', function=following, schedule=schedule, form='agent')
    assert sorted(_agent_values(group, agents, 1).tolist()) == expected


def test_agent_map_in_the_increment_mode_adds_the_value():
    """Test that an agent map in the increment mode adds the value of the function to the state."""
    group = GroupElement('value', element=ValueElement, size=2, values={'x': [0.0, 1.0]})
    agents = MapInteraction('map', target='value', state='x', function=lambda a, m: 0.5, mode='increment', form='agent')
    assert _agent_values(group, agents, 2).tolist() == [1.0, 2.0]


def test_agent_map_reads_a_whole_group_bound_to_its_sources():
    """Test that an entry of the sources naming a whole group binds its agents, which the function reads by role."""
    group = GroupElement('value', element=ValueElement, size=2, values={'x': [0.0, 1.0]})
    others = GroupElement('others', element=ValueElement, size=3, values={'x': [1.0, 2.0, 3.0]})
    agents = MapInteraction(
        'map',
        target='value',
        state='x',
        function=lambda a, m: sum(other.x for other in m.get_agents('sources')),
        sources=['others'],
        form='agent',
    )
    system = CompositeSystem(elements=[group, others], interactions=[agents], time=DiscreteTime())
    trajectory = simulate(system, until=1, random_state=0)
    assert trajectory.get_agents('value', 1).values['x'].tolist() == [6.0, 6.0]


class VectorElement(Element):
    """Element with a vector state `w`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        w: Annotated[object, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.w = w


@pytest.mark.parametrize(
    ('mode', 'expected'),
    [('replace', [[2.0, 4.0], [6.0, 8.0]]), ('increment', [[3.0, 6.0], [9.0, 12.0]])],
)
def test_agent_map_sets_a_vector_state(mode, expected):
    """Test that an agent map whose function returns an array replaces or increments a vector state."""
    group = GroupElement('vectors', element=VectorElement, size=2, values={'w': [[1.0, 2.0], [3.0, 4.0]]})
    doubling = MapInteraction(
        'map',
        target='vectors',
        state='w',
        function=lambda a, m: 2 * np.asarray(a.w),
        mode=mode,
        form='agent',
    )
    system = CompositeSystem(elements=[group], interactions=[doubling], time=DiscreteTime())
    trajectory = simulate(system, until=1, random_state=0)
    assert trajectory.get_agents('vectors', 1).values['w'].tolist() == expected


def test_agent_map_rejects_a_vector_of_another_shape():
    """Test that an agent map whose function returns an array of another shape than its vector state raises."""
    group = GroupElement('vectors', element=VectorElement, size=1, values={'w': [[1.0, 2.0]]})
    stretching = MapInteraction('map', target='vectors', state='w', function=lambda a, m: [1.0, 2.0, 3.0], form='agent')
    system = CompositeSystem(elements=[group], interactions=[stretching], time=DiscreteTime())
    with pytest.raises(ValidationError, match='shape of the state'):
        simulate(system, until=1, random_state=0)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'mode': 'assign', 'function': lambda s: 0.0}, 'Unknown mode'),
        ({'schedule': 'parallel', 'function': lambda s: 0.0}, 'schedule of map'),
        ({'form': 'mesa', 'function': lambda s: 0.0}, 'form of map'),
        ({'schedule': 'member', 'function': lambda s: 0.0}, 'only the agent form'),
        ({'function': 1.0}, 'callable'),
    ],
)
def test_check_rejects_unknown_variants_and_functions(arguments, match):
    """Test that an unknown mode, schedule, or form, or a function that is not callable raise errors."""
    with pytest.raises(ValidationError, match=match):
        MapInteraction('map', target='value', state='x', **arguments).prepare()
