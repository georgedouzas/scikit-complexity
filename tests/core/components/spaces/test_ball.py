"""Test the `_ball` module of `skcomplex.core.components.spaces`."""

import pytest

from skcomplex.core.components.spaces import BallNeighborhood
from skcomplex.core.errors import ValidationError

SQUARE = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
CROSS = ((-1, 0), (0, -1), (0, 1), (1, 0))
HEXAGON = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))


@pytest.mark.parametrize(
    ('neighborhood', 'offsets'),
    [
        (BallNeighborhood(), SQUARE),
        (BallNeighborhood(center='first'), ((0, 0), *SQUARE)),
        (BallNeighborhood(metric='manhattan'), CROSS),
        (BallNeighborhood(metric='manhattan', center='first'), ((0, 0), *CROSS)),
        (BallNeighborhood(n_dim=1, center='inline'), ((-1,), (0,), (1,))),
        (BallNeighborhood(2, n_dim=1, center='inline'), ((-2,), (-1,), (0,), (1,), (2,))),
        (BallNeighborhood(metric='hexagonal'), HEXAGON),
        (BallNeighborhood(metric='hexagonal', center='first'), ((0, 0), *HEXAGON)),
    ],
)
def test_ball_lists_its_offsets_in_order(neighborhood, offsets):
    """Test that every metric and centre gives the offsets in their order."""
    assert neighborhood.prepare().offsets_ == offsets


@pytest.mark.parametrize(
    ('neighborhood', 'size'),
    [
        (BallNeighborhood(2), 24),
        (BallNeighborhood(2, n_dim=3, center='first'), 125),
        (BallNeighborhood(2, metric='manhattan', center='first'), 13),
        (BallNeighborhood(1, n_dim=2, center='inline'), 9),
        (BallNeighborhood(0, center='first'), 1),
    ],
)
def test_ball_counts_the_offsets_within_its_radius(neighborhood, size):
    """Test that a ball holds every offset within its radius under its metric."""
    assert len(neighborhood.prepare().offsets_) == size


def test_hexagonal_ball_of_radius_two_walks_its_rings():
    """Test that the hexagonal ball of radius 2 lists the inner ring, then the outer ring counterclockwise."""
    offsets = BallNeighborhood(2, metric='hexagonal').prepare().offsets_
    assert offsets[:6] == HEXAGON
    assert offsets[6:] == (
        (0, 2),
        (1, 1),
        (2, 0),
        (2, -1),
        (2, -2),
        (1, -2),
        (0, -2),
        (-1, -1),
        (-2, 0),
        (-2, 1),
        (-2, 2),
        (-1, 2),
    )
    assert all(max(abs(a), abs(b), abs(a + b)) == 2 for a, b in offsets[6:])


def test_hexagonal_ball_puts_opposite_directions_three_apart():
    """Test that direction `k + 3` modulo 6 of the hexagonal ball of radius 1 is opposite to direction `k`."""
    offsets = BallNeighborhood(metric='hexagonal').prepare().offsets_
    assert all(offsets[(k + 3) % 6] == (-offsets[k][0], -offsets[k][1]) for k in range(6))


@pytest.mark.parametrize(
    ('neighborhood', 'match'),
    [
        (BallNeighborhood(metric='euclidean'), 'metric'),
        (BallNeighborhood(center=True), 'centre'),
        (BallNeighborhood(-1), 'nonnegative radius'),
        (BallNeighborhood(metric='hexagonal', n_dim=3), 'two dimensions'),
        (BallNeighborhood(metric='hexagonal', center='inline'), 'lexicographic place'),
    ],
)
def test_ball_rejects_invalid_settings(neighborhood, match):
    """Test that an unknown metric or centre, a negative radius, and invalid hexagonal balls are rejected."""
    with pytest.raises(ValidationError, match=match):
        neighborhood.prepare()
