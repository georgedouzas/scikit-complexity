"""Test the `_subshift` module of `skcomplex.models.computer_science.automata`."""

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup

from skcomplex.core.spaces import CayleySpace, LatticeNeighborhood, LatticeSpace, MemoryNeighborhood
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


def test_reference_values_of_the_golden_mean_shift():
    """Test that the golden mean shift keeps its recorded counts, violations, and samples with fixed seeds."""
    golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
    assert golden.count(LatticeSpace(6, boundary='fixed')) == 21
    assert golden.count(LatticeSpace(6)) == 18
    assert golden.count(LatticeSpace(30, boundary='fixed')) == 2178309
    assert golden.violations([1, 0, 0, 0, 1], LatticeSpace(5)).tolist() == [False, False, False, False, True]
    assert not golden.is_admissible([1, 0, 1, 0, 1], LatticeSpace(5))
    assert golden.sample(LatticeSpace(12), 10, random_state=0).tolist() == [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    window = LatticeSpace(12, boundary='fixed')
    assert golden.sample(window, 5, random_state=7).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    initial = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    sample = golden.sample(LatticeSpace(10), 3, initial=initial, random_state=3)
    assert sample.tolist() == [0, 0, 1, 0, 0, 1, 0, 1, 0, 1]


def test_reference_values_on_a_group_and_with_three_values():
    """Test that subshifts on a cyclic group and over three values keep their recorded counts and samples."""
    golden = Subshift(MemoryNeighborhood(('1', 'a')), [(1, 1)])
    space = CayleySpace(PermutationGroup([[tuple(range(1, 13))]]))
    assert golden.count(space) == 322
    assert golden.sample(space, 20, random_state=0).tolist() == [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    violations = golden.violations([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], space)
    assert np.flatnonzero(violations).tolist() == [0]
    three = Subshift(LatticeNeighborhood(((0,), (1,))), [(0, 2), (2, 0)], states=3)
    assert three.count(LatticeSpace(5, boundary='fixed')) == 99
    assert three.sample(LatticeSpace(8), 5, random_state=1).tolist() == [0, 1, 1, 1, 1, 0, 0, 1]
