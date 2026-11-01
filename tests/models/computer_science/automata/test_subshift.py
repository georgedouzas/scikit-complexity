"""Test the `_subshift` module of `skcomplex.models.computer_science.automata`."""

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup

from skcomplex.core.neighborhoods import LatticeNeighborhood, MemoryNeighborhood
from skcomplex.core.spaces import CayleySpace, LatticeSpace
from skcomplex.errors import ValidationError
from skcomplex.models.computer_science.automata import Subshift


def test_golden_mean_shift_counts_follow_fibonacci():
    """Test that the words of length n of the golden mean shift number F(n + 2)."""
    golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
    counts = [golden.count(LatticeSpace(length, boundary='fixed')) for length in range(1, 11)]
    assert counts == [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


def test_violations_mark_the_forbidden_patterns():
    """Test that two adjacent ones are flagged at the first of them, and that the edge of a window is not checked."""
    golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
    space = LatticeSpace(5, boundary='fixed')
    np.testing.assert_array_equal(golden.violations([0, 1, 1, 0, 1], space), [False, True, False, False, False])
    assert golden.is_admissible([1, 0, 1, 0, 1], space)


def test_golden_mean_shift_samples_admissible_configurations():
    """Test that the Markov chain keeps configurations without two adjacent ones, and rejects a bad start."""
    golden = Subshift(MemoryNeighborhood(('1', 'a')), [(1, 1)])
    space = CayleySpace(PermutationGroup([[tuple(range(1, 61))]]))
    sample = golden.sample(space, 20, random_state=0)
    assert golden.is_admissible(sample, space)
    assert 0 < sample.mean() < 0.5
    with pytest.raises(ValidationError, match='forbidden pattern'):
        golden.sample(space, initial=np.ones(60))
