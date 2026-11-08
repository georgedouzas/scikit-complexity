"""Test the `_sandpiles` module of `skcomplex.systems.physics.criticality`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.criticality import (
    FixedEnergySandpileSystem,
    SelfOrganizedBranchingSystem,
    ZhangSandpileSystem,
)


@pytest.mark.parametrize(('density', 'active'), [(1.9, False), (2.5, True)])
def test_fixed_energy_sandpile_conserves_grains_and_crosses_its_absorbing_transition(density, active):
    """Test that grains are conserved, and that the activity dies below the critical density and lasts above it."""
    trajectory = simulate(FixedEnergySandpileSystem(density=density), random_state=0)
    heights = np.asarray(trajectory.values('pile.height'))
    totals = heights.reshape(len(heights), -1).sum(axis=1)
    assert np.all(totals == round(density * 32 * 32))
    activity = np.asarray(trajectory.values('activity'))[-50:]
    assert (activity.min() > 0.05) if active else (activity.max() == 0)


def test_zhang_sandpile_relaxes_below_the_threshold_with_avalanches_of_many_sizes():
    """Test that every site ends below the threshold and that avalanches range from single sites to large ones."""
    trajectory = simulate(ZhangSandpileSystem(), random_state=0)
    energies = np.asarray(trajectory.values('pile.energy'))
    sizes = np.array([event.size for event in trajectory.events])
    assert energies.max() < 1.0
    assert energies.min() >= 0.0
    assert sizes.min() == 1
    assert sizes.max() > 100


def test_self_organized_branching_tunes_itself_to_the_critical_probability():
    """Test that the branching probability organizes itself to `p_c = 1 / 2` from below and from above."""
    for start in (0.3, 0.7):
        p = np.asarray(simulate(SelfOrganizedBranchingSystem(p=start), random_state=0).values('tree.p'))
        assert p[-10000:].mean() == pytest.approx(0.5, abs=0.02)
