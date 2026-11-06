"""Test the `_communities` module of `skcomplex.systems.biology.ecology`."""

import pytest

from skcomplex.systems.biology.ecology import (
    competition,
    food_webs,
    mutualism,
)

from ..conftest import _run_until


def test_competition_settles_at_coexistence():
    """Test that weak competitors coexist at the equilibrium of the competition system."""
    trajectory = _run_until(competition(), 100, dt=1.0)
    assert trajectory.values('first.n')[-1] == pytest.approx((100 - 0.5 * 80) / (1 - 0.3), rel=1e-3)
    assert trajectory.values('second.n')[-1] == pytest.approx((80 - 0.6 * 100) / (1 - 0.3), rel=1e-3)


def test_competition_excludes_weak_species():
    """Test that strong competition lets one species exclude the other."""
    trajectory = _run_until(competition(alpha12=0.5, alpha21=1.5), 200, dt=1.0)
    assert trajectory.values('second.n')[-1] < 1e-3
    assert trajectory.values('first.n')[-1] == pytest.approx(100, rel=1e-3)


def test_mutualism_with_handling_is_bounded_above_capacities():
    """Test that mutualism with a handling time settles above the capacities, and without it explodes."""
    bounded = _run_until(mutualism(), 100, dt=1.0)
    assert 50 < bounded.values('first.n')[-1] < 100
    assert bounded.values('first.n')[-1] == pytest.approx(bounded.values('first.n')[-2], rel=1e-4)
    explosive = _run_until(mutualism(handling=0.0, a=0.1, b=0.3), 2, dt=1.0)
    assert explosive.values('first.n')[-1] > 150


def test_food_web_species_coexist():
    """Test that the four species of the default food web coexist at a stable equilibrium."""
    trajectory = _run_until(food_webs(), 400, dt=1.0)
    final = [trajectory.values(f'{name}.n')[-1] for name in ('frogs', 'mice', 'snakes', 'hawks')]
    assert min(final) > 10
    assert trajectory.values('snakes.n')[-1] == pytest.approx(trajectory.values('snakes.n')[-20], rel=1e-3)
