"""Test the `_jump` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import JumpInteraction
from skcomplex.core.components.laws.utils import draw
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec

MEMBERS = 2000


class CounterElement(Element):
    """Counter `x` with a mark `y`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y


def _final(interactions, until, *, time=None, size=MEMBERS, dt=1.0, state='x'):
    """Return the final values of a state of every member of a group of counters."""
    group = GroupElement('counters', element=CounterElement, size=size)
    system = System(elements=[group], interactions=interactions, time=time)
    trajectory = simulate(system, until=until, dt=dt, random_state=0)
    return np.atleast_1d(trajectory.values(f'counters.{state}')[-1])


def test_jumps_at_a_rate_count_a_poisson_process():
    """Test that jumps of one unit at rate 2 count a Poisson process of mean and variance 20 at time 10."""
    counts = _final([JumpInteraction('count', target='counters', changes={'x': 1}, rate=lambda s: 2.0)], 10.0)
    assert counts.mean() == pytest.approx(20.0, rel=0.03)
    assert counts.var() == pytest.approx(20.0, rel=0.1)


def test_signed_jumps_walk_on_the_integers():
    """Test that jumps of plus and minus one walk on the integers with mean 0 and variance growing as `2 r t`."""
    walk = [
        JumpInteraction('up', target='counters', changes={'x': 1}, rate=lambda s: 1.0),
        JumpInteraction('down', target='counters', changes={'x': -1}, rate=lambda s: 1.0),
    ]
    positions = _final(walk, 10.0)
    assert np.any(positions < 0)
    assert abs(positions.mean()) < 0.3
    assert positions.var() == pytest.approx(20.0, rel=0.1)


def test_jumps_with_a_probability_fire_in_steps_of_discrete_time():
    """Test that a jump with probability 0.3 per step fires a binomial number of times in 10 steps."""
    step = JumpInteraction('step', target='counters', changes={'x': 1, 'y': -1}, probability=lambda s: 0.3)
    counts = _final([step], 10, time=DiscreteTime())
    marks = _final([step], 10, time=DiscreteTime(), state='y')
    assert counts.mean() == pytest.approx(3.0, rel=0.05)
    assert counts.var() == pytest.approx(2.1, rel=0.15)
    np.testing.assert_array_equal(marks, -counts)


def test_jumps_after_drawn_waits_renew_on_their_own_clocks():
    """Test that jumps after exponential waits of mean 0.5 count a Poisson process of rate 2."""
    renewal = JumpInteraction(
        'renewal',
        target='counters',
        changes={'x': 1},
        waiting=lambda s: draw('wait', 'expon', scale=0.5),
    )
    counts = _final([renewal], 10.0, size=500)
    assert counts.mean() == pytest.approx(20.0, rel=0.05)


def test_jumps_after_fixed_waits_reset_states():
    """Test that jumps after waits of 1.5 add their changes and set the reset states at every firing."""
    renewal = JumpInteraction(
        'renewal',
        target='counters',
        changes={'x': 1},
        waiting=lambda s: 1.5,
        reset={'y': lambda s: s.target.x + 10},
    )
    assert _final([renewal], 4.0, size=1, dt=0.5).tolist() == [2.0]
    assert _final([renewal], 4.0, size=1, dt=0.5, state='y').tolist() == [11.0]


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'changes': {'x': 1}}, 'exactly one'),
        ({'changes': {'x': 1}, 'rate': lambda s: 1.0, 'probability': lambda s: 0.5}, 'exactly one'),
        ({'changes': {'x': 1}, 'rate': 1.0}, 'function of the namespace'),
        ({'rate': lambda s: 1.0}, 'changes no state'),
        ({'changes': {'x': 1}, 'rate': lambda s: 1.0, 'reset': {'y': lambda s: 0}}, 'only a waiting time'),
        ({'waiting': lambda s: 1.0, 'reset': {'y': 0}}, 'Give functions'),
    ],
)
def test_check_rejects_clocks_and_changes_it_cannot_take(arguments, match):
    """Test that a jump without one clock, without changes, or with a reset it cannot take raises an error."""
    with pytest.raises(ValidationError, match=match):
        JumpInteraction('jump', target='counters', **arguments).prepare()


class PositionElement(Element):
    """Element with a vector position."""

    def __init__(self, label: str, *, x: Annotated[object, StateSpec(vector=True)] = 0.0) -> None:
        super().__init__(label)
        self.x = x


def test_jumps_after_waits_reset_every_component_of_a_vector_state():
    """Test that a reset of a vector state sets every component, here a step of `+1` along both axes per wait."""
    steps = JumpInteraction(
        'steps',
        target='walker',
        waiting=lambda s: 1.0,
        reset={'x': lambda s: [item + 1 for item in s.target.x]},
    )
    system = System(
        elements=[PositionElement('walker', x=[0.0, 0.0])],
        interactions=[steps],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = simulate(system, until=3.5, dt=1.0, random_state=0)
    np.testing.assert_allclose(trajectory.values('walker.x')[-1], [3.0, 3.0])
