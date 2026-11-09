"""Test the `_multilevel_selection` module of `skcomplex.systems.social.cooperation`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.social.cooperation import (
    NonlinearPairwiseGroupConflictSystem,
    TrimorphicMultilevelSelectionSystem,
    TwoLevelReplicatorSystem,
)


def _final_density(system, until, dt=0.02):
    """Return the initial and final density of a system of group compositions."""
    trajectory = simulate(system, until=until, dt=until, solver=RK4Solver(dt=dt))
    values = trajectory.values('groups.x')
    return values[0], values[-1]


@pytest.mark.parametrize('victory', ['fermi', 'local', 'tullock'])
def test_nonlinear_pairwise_group_conflict_keeps_the_total(victory):
    """Test that every victory probability keeps the integral of the density at one."""
    start, final = _final_density(
        NonlinearPairwiseGroupConflictSystem(40, victory=victory, exponent=2.0).prepare().system_,
        3.0,
    )
    assert abs(final.sum() / 40 - 1.0) < 1e-10
    assert not np.allclose(final, start)


def test_nonlinear_pairwise_group_conflict_strong_conflict_sustains_punishers():
    """Test that the mean fraction of punishers ends higher with strong than with weak between-group conflict."""
    centres = (np.arange(50) + 0.5) / 50
    means = {}
    for strength in (0.1, 2.0):
        _, final = _final_density(
            NonlinearPairwiseGroupConflictSystem(50, group_strength=strength).prepare().system_,
            10.0,
        )
        means[strength] = float((final * centres).sum() / 50)
    assert means[2.0] > 0.8
    assert means[0.1] < 0.1


@pytest.mark.parametrize('victory', ['defectors', 'payoff'])
def test_trimorphic_multilevel_selection_lowers_defection_and_keeps_the_total(victory):
    """Test that the mean fraction of defectors falls on the simplex while the integral stays one."""
    system = TrimorphicMultilevelSelectionSystem(16, victory=victory).prepare().system_
    space = system.elements[0].space.prepare()
    weights = space.volumes_.ravel() / 16**2
    defectors = np.broadcast_to(space.coordinates_['y'][None, None, :], space.shape_).ravel()
    start, final = _final_density(system, 5.0)
    assert abs((final * weights).sum() - 1.0) < 1e-10
    assert (final * weights * defectors).sum() < 0.8 * (start * weights * defectors).sum()


def test_multilevel_selection_rejects_an_unknown_victory_probability():
    """Test that an unknown victory probability raises a named error listing the known ones."""
    with pytest.raises(ValidationError, match='fermi'):
        NonlinearPairwiseGroupConflictSystem(victory='majority').prepare()
    with pytest.raises(ValidationError, match='defectors'):
        TrimorphicMultilevelSelectionSystem(victory='fermi').prepare()


@pytest.mark.parametrize(('strength', 'expected'), [(2.0, 0.75), (1.0, 0.5)])
def test_two_level_replicator_reaches_the_mean_fraction_of_punishers(strength, expected):
    """Test that above `lambda* = c + q - p` the mean fraction of punishers tends to `1 - (c + q - p) / lambda`."""
    _, final = _final_density(TwoLevelReplicatorSystem(group_strength=strength).prepare().system_, 20.0, dt=0.01)
    final = final.ravel()
    centres = (np.arange(final.size) + 0.5) / final.size
    assert np.sum(centres * final) / np.sum(final) == pytest.approx(expected, abs=0.02)
    assert np.mean(final) == pytest.approx(1.0, rel=1e-3)


def test_two_level_replicator_loses_the_punishers_below_the_threshold():
    """Test that groups concentrate at all defectors below `lambda*`, and at all punishers if `p > c + q`."""
    centres = (np.arange(100) + 0.5) / 100
    _, weak = _final_density(TwoLevelReplicatorSystem(group_strength=0.3).prepare().system_, 20.0, dt=0.01)
    assert np.sum(centres * weak.ravel()) / np.sum(weak) < 0.05
    _, strong = _final_density(TwoLevelReplicatorSystem(punishment=1.5).prepare().system_, 20.0, dt=0.01)
    assert np.sum(centres * strong.ravel()) / np.sum(strong) > 0.95


def test_two_level_replicator_rejects_a_non_additive_victory():
    """Test that only additively separable victory probabilities reduce to a two-level replicator equation."""
    with pytest.raises(ValidationError, match='Unknown victory'):
        TwoLevelReplicatorSystem(victory='fermi').prepare()
