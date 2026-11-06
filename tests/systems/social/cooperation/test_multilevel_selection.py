"""Test the `_multilevel_selection` module of `skcomplex.systems.social.cooperation`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.social.cooperation import nonlinear_pairwise_group_conflict, trimorphic_multilevel_selection


def _final_density(system, until, dt=0.02):
    """Return the initial and final density of a system of group compositions."""
    simulation = Simulation(system, dt=until, solver=RK4Solver(dt=dt)).prepare().run(until=until)
    values = simulation.trajectory_.values('groups.f')
    return values[0], values[-1]


@pytest.mark.parametrize('victory', ['fermi', 'local', 'tullock'])
def test_nonlinear_pairwise_group_conflict_keeps_the_total(victory):
    """Test that every victory probability keeps the integral of the density at one."""
    start, final = _final_density(nonlinear_pairwise_group_conflict(40, victory=victory, exponent=2.0), 3.0)
    assert abs(final.sum() / 40 - 1.0) < 1e-10
    assert not np.allclose(final, start)


def test_nonlinear_pairwise_group_conflict_strong_conflict_sustains_punishers():
    """Test that the mean fraction of punishers ends higher with strong than with weak between-group conflict."""
    centres = (np.arange(50) + 0.5) / 50
    means = {}
    for strength in (0.1, 2.0):
        _, final = _final_density(nonlinear_pairwise_group_conflict(50, group_strength=strength), 10.0)
        means[strength] = float((final * centres).sum() / 50)
    assert means[2.0] > 0.8
    assert means[0.1] < 0.1


@pytest.mark.parametrize('victory', ['defectors', 'payoff'])
def test_trimorphic_multilevel_selection_lowers_defection_and_keeps_the_total(victory):
    """Test that the mean fraction of defectors falls on the simplex while the integral stays one."""
    system = trimorphic_multilevel_selection(16, victory=victory)
    space = system.elements[0].space.prepare()
    weights = space.volumes_.ravel() / 16**2
    defectors = np.broadcast_to(space.coordinates_['y'][None, None, :], space.shape_).ravel()
    start, final = _final_density(system, 5.0)
    assert abs((final * weights).sum() - 1.0) < 1e-10
    assert (final * weights * defectors).sum() < 0.8 * (start * weights * defectors).sum()


def test_multilevel_selection_rejects_an_unknown_victory_probability():
    """Test that an unknown victory probability raises a named error listing the known ones."""
    with pytest.raises(ValidationError, match='fermi'):
        nonlinear_pairwise_group_conflict(victory='majority')
    with pytest.raises(ValidationError, match='defectors'):
        trimorphic_multilevel_selection(victory='fermi')
