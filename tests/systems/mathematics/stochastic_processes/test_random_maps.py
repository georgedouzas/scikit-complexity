"""Test the `_random_maps` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import numpy as np
import pytest

from skcomplex.systems.mathematics.stochastic_processes import RandomBranchMapSystem

from ..conftest import _run


def test_random_branch_map_with_a_reset_branch_has_the_stationary_mean_of_its_renewals():
    """Test that with `mu = 0` the orbits settle on the stationary mean `p1 / (2 - p1)`."""
    x = _run(RandomBranchMapSystem(2000, p1=0.6, mu=0.0), 100).values('orbits.x')[50:]
    assert x.mean() == pytest.approx(0.6 / 1.4, abs=0.01)


def test_random_branch_map_keeps_the_orbits_in_the_unit_interval():
    """Test that both branches map `[0, 1]` into itself for `mu <= 1`, and that the orbits spread over it."""
    x = _run(RandomBranchMapSystem(500, p1=0.5, mu=1.0), 200).values('orbits.x')[20:]
    assert x.min() >= 0
    assert x.max() <= 1
    assert np.std(x) > 0.1


def test_random_branch_map_collapses_onto_one_above_half_for_unit_coefficient():
    """Test that for `mu = 1` the common fixed point 1 attracts the orbits for `p1 > 1/2` and repels them below."""
    above = _run(RandomBranchMapSystem(200, p1=0.7, mu=1.0), 300).values('orbits.x')[-1]
    below = _run(RandomBranchMapSystem(200, p1=0.3, mu=1.0), 300).values('orbits.x')[-1]
    np.testing.assert_allclose(above, 1.0, atol=1e-6)
    assert below.mean() < 0.5
