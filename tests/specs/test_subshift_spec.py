"""Test the subshift spec."""

import pytest

from skcomplex.core.spaces import LatticeNeighborhood
from skcomplex.errors import ValidationError
from skcomplex.specs import SubshiftSpec

PAIR = LatticeNeighborhood(((0,), (1,)))


def test_alphabet_holds_the_values_of_a_cell():
    """Test that the alphabet holds the values 0 to states - 1."""
    assert list(SubshiftSpec(PAIR, ((1, 1),), 3).alphabet) == [0, 1, 2]


def test_check_rejects_invalid_patterns():
    """Test that forbidden patterns of different lengths or with values outside the states are rejected."""
    with pytest.raises(ValidationError, match='differ in length'):
        SubshiftSpec(PAIR, ((1, 1), (1,))).prepare()
    with pytest.raises(ValidationError, match='outside 0 to 1'):
        SubshiftSpec(PAIR, ((2, 1),)).prepare()
