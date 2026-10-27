"""Test the `_jump` module of `skcomplex.core.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError
from skcomplex.core.laws import JumpLaw
from skcomplex.core.laws.utils import draw

N, K = var('n k')


def test_jump_targets_its_changes_or_its_reset():
    """Test that a jump at a rate targets and runs into its changes, and one after a waiting time its reset."""
    law = JumpLaw(2 * N, {N: -1}).check().prepare()
    assert (law.targets_, law.run({'n': 3.0})) == ((N,), {N: -1})
    waiting = JumpLaw(waiting=draw('wait', 'expon'), reset={N: K}, label='fires').check().prepare()
    assert (waiting.targets_, waiting.run({'k': 4.0})) == ((N,), {N: 4.0})


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({}, 'exactly one of them'),
        ({'rate': 1.0, 'waiting': 1.0}, 'exactly one of them'),
        ({'waiting': draw('wait', 'expon'), 'changes': {N: 1}}, 'changes belong to the exponential clock'),
        ({'rate': 1.0, 'changes': {N: 1}, 'reset': {N: 0}}, 'resets and labels to the waiting'),
        ({'rate': 1.0, 'changes': {N: 1}, 'label': 'fires'}, 'takes a label'),
        ({'rate': 1.0, 'changes': {N: K}}, 'holds a symbol'),
    ],
)
def test_jump_rejects_invalid_clocks_and_increments(arguments, message):
    """Test that a jump rejects both or no clock, an option of the clock it does not use, and a symbolic increment."""
    with pytest.raises(ValidationError, match=message):
        JumpLaw(**arguments).check()
