"""Test the `_jump` module of `skcomplex.core.components.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.laws.utils import draw
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

N, K = var('n k')


def test_jump_targets_its_changes_or_its_reset():
    """Test that a jump at a rate targets and gives its changes, and one after a waiting time its reset."""
    law = JumpLaw(2 * N, {N: -1})
    assert (law._get_targets(), law._get_value({'n': 3.0})) == ((N,), {N: -1})
    waiting = JumpLaw(waiting=draw('wait', 'expon'), reset={N: K}, label='fires')
    assert (waiting._get_targets(), waiting._get_value({'k': 4.0})) == ((N,), {N: 4.0})


def test_jump_at_a_rate_takes_the_label_of_its_channel():
    """Test that a jump at a rate takes a label, which names its channel, as an infection of an epidemic."""
    law = JumpLaw(0.5 * N, {N: -1, K: 1}, label='infection')
    assert (law.label, law._get_targets()) == ('infection', (N, K))


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({}, 'exactly one of them'),
        ({'rate': 1.0, 'waiting': 1.0}, 'exactly one of them'),
        ({'waiting': draw('wait', 'expon'), 'changes': {N: 1}}, 'changes belong to the exponential clock'),
        ({'rate': 1.0, 'changes': {N: 1}, 'reset': {N: 0}}, 'resets to the waiting'),
        ({'rate': 1.0, 'changes': {N: K}}, 'holds a symbol'),
    ],
)
def test_jump_rejects_invalid_clocks_and_increments(arguments, message):
    """Test that a jump rejects both or no clock, an option of the clock it does not use, and a symbolic increment."""
    with pytest.raises(ValidationError, match=message):
        JumpLaw(**arguments)._check()


def test_jump_moving_a_quantity_between_two_stocks_conserves_their_total():
    """Test that a jump that takes one unit from a stock and gives it to another keeps their total."""
    stocks = Element('stocks', values={'a': 50.0, 'b': 0.0})
    move = Interaction(
        'move',
        roles={'target': 'stocks'},
        laws=lambda s: [JumpLaw(0.1 * s.target.a, {s.target.a: -1, s.target.b: 1}, label='move')],
    )
    trajectory = System(elements=[stocks], interactions=[move]).prepare().simulate(until=10, dt=1, random_state=0)
    assert np.all(trajectory.get_values('stocks.a') + trajectory.get_values('stocks.b') == 50.0)
    assert trajectory.get_values('stocks.b')[-1] > 0
