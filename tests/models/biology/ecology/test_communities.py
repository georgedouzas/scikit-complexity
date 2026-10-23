"""Test the `_communities` module of `skcomplex.models.biology.ecology`."""

import pytest

from skcomplex.models.biology.ecology import (
    competition,
    food_webs,
    mutualism,
)

from ..conftest import _run_until


def test_competition_settles_at_coexistence():
    """Test that weak competitors coexist at the equilibrium of the competition model."""
    trajectory = _run_until(competition(), 100, dt=1.0)
    assert trajectory['first.n'][-1] == pytest.approx((100 - 0.5 * 80) / (1 - 0.3), rel=1e-3)
    assert trajectory['second.n'][-1] == pytest.approx((80 - 0.6 * 100) / (1 - 0.3), rel=1e-3)


def test_competition_excludes_weak_species():
    """Test that strong competition lets one species exclude the other."""
    trajectory = _run_until(competition(alpha12=0.5, alpha21=1.5), 200, dt=1.0)
    assert trajectory['second.n'][-1] < 1e-3
    assert trajectory['first.n'][-1] == pytest.approx(100, rel=1e-3)


def test_mutualism_with_handling_is_bounded_above_capacities():
    """Test that mutualism with a handling time settles above the capacities, and without it explodes."""
    bounded = _run_until(mutualism(), 100, dt=1.0)
    assert 50 < bounded['first.n'][-1] < 100
    assert bounded['first.n'][-1] == pytest.approx(bounded['first.n'][-2], rel=1e-4)
    explosive = _run_until(mutualism(handling=0.0, a=0.1, b=0.3), 2, dt=1.0)
    assert explosive['first.n'][-1] > 150


def test_food_web_species_coexist():
    """Test that the four species of the default food web coexist at a stable equilibrium."""
    trajectory = _run_until(food_webs(), 400, dt=1.0)
    final = [trajectory[f'{name}.n'][-1] for name in ('frogs', 'mice', 'snakes', 'hawks')]
    assert min(final) > 10
    assert trajectory['snakes.n'][-1] == pytest.approx(trajectory['snakes.n'][-20], rel=1e-3)
