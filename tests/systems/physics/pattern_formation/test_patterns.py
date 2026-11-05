"""Test the `_patterns` module of `skcomplex.systems.physics.pattern_formation`."""

import numpy as np

from skcomplex.systems.physics.pattern_formation import (
    complex_ginzburg_landau,
)


def test_complex_ginzburg_landau_seeds_both_parts():
    """Test that the random state fixes both the real and the imaginary part of the initial amplitude."""
    first, second, other = (complex_ginzburg_landau((8, 8), random_state=seed) for seed in (1, 1, 2))
    for part in ('u', 'v'):
        values = [np.asarray(system.elements[0].values[part]) for system in (first, second, other)]
        np.testing.assert_array_equal(values[0], values[1])
        assert not np.array_equal(values[0], values[2])
