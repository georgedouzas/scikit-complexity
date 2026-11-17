"""Test the `_radius` module of `skcomplex.core.components.spaces`."""

import math

import pytest

from skcomplex.core.components.spaces import RadiusNeighborhood
from skcomplex.core.errors import ValidationError

SQUARE = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
CROSS = ((-1, 0), (0, -1), (0, 1), (1, 0))
HEXAGON = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))


def test_radius_neighbourhood_needs_a_positive_radius():
    """Test that a radius neighbourhood takes a positive radius and rejects zero."""
    assert RadiusNeighborhood(1.5).radius == 1.5
    with pytest.raises(ValidationError, match='positive radius'):
        RadiusNeighborhood(0.0).prepare()


@pytest.mark.parametrize(
    ('radius', 'angle', 'message'),
    [(0.0, 1.0, 'positive radius'), (1.0, 0.0, 'half-angle'), (1.0, 4.0, 'half-angle')],
)
def test_radius_neighbourhood_checks_its_radius_and_angle(radius, angle, message):
    """Test that a cone needs a positive radius and a half-angle in (0, pi]."""
    with pytest.raises(ValidationError, match=message):
        RadiusNeighborhood(radius, angle=angle).prepare()


def test_radius_neighbourhood_takes_the_whole_disk_by_default():
    """Test that a cone of half-angle pi is accepted, and that it is the default angle."""
    assert RadiusNeighborhood(1.0, angle=math.pi).angle == math.pi
    assert RadiusNeighborhood(1.0).angle == math.pi


@pytest.mark.parametrize(
    ('neighborhood', 'n_dim', 'offsets'),
    [
        (RadiusNeighborhood(1, metric='chebyshev'), 2, SQUARE),
        (RadiusNeighborhood(1, metric='chebyshev', center=True), 2, ((0, 0), *SQUARE)),
        (RadiusNeighborhood(1, metric='manhattan'), 2, CROSS),
        (RadiusNeighborhood(1, metric='manhattan', center=True), 2, ((0, 0), *CROSS)),
        (RadiusNeighborhood(1), 2, CROSS),
        (RadiusNeighborhood(1.5), 2, SQUARE),
        (RadiusNeighborhood(2, metric='chebyshev', center=True), 1, ((0,), (-2,), (-1,), (1,), (2,))),
        (RadiusNeighborhood(1, metric='hexagonal'), 2, HEXAGON),
        (RadiusNeighborhood(1, metric='hexagonal', center=True), 2, ((0, 0), *HEXAGON)),
    ],
)
def test_radius_neighbourhood_lists_its_lattice_offsets_in_order(neighborhood, n_dim, offsets):
    """Test that every metric and centre gives the lattice offsets in their order."""
    assert neighborhood._get_offsets(n_dim) == offsets


@pytest.mark.parametrize(
    ('neighborhood', 'n_dim', 'size'),
    [
        (RadiusNeighborhood(2, metric='chebyshev'), 2, 24),
        (RadiusNeighborhood(2, metric='chebyshev', center=True), 3, 125),
        (RadiusNeighborhood(2, metric='manhattan', center=True), 2, 13),
        (RadiusNeighborhood(1, metric='chebyshev'), 3, 26),
        (RadiusNeighborhood(0.5, center=True), 2, 1),
    ],
)
def test_radius_neighbourhood_counts_the_offsets_within_its_radius(neighborhood, n_dim, size):
    """Test that a lattice reads every offset within the radius under the metric."""
    assert len(neighborhood._get_offsets(n_dim)) == size


def test_radius_neighbourhood_needs_a_lattice_for_its_offsets():
    """Test that the offsets wait for the dimensions of a lattice, which the hexagonal metric fixes to two."""
    assert RadiusNeighborhood(1, metric='chebyshev')._get_offsets() is None
    assert RadiusNeighborhood(1, metric='hexagonal')._get_offsets() == HEXAGON


def test_hexagonal_neighbourhood_of_radius_two_walks_its_rings():
    """Test that the hexagonal neighbourhood of radius 2 lists the inner ring, then the outer ring counterclockwise."""
    offsets = RadiusNeighborhood(2, metric='hexagonal')._get_offsets(2)
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


def test_hexagonal_neighbourhood_puts_opposite_directions_three_apart():
    """Test that direction `k + 3` modulo 6 of the hexagonal neighbourhood is opposite to direction `k`."""
    offsets = RadiusNeighborhood(1, metric='hexagonal')._get_offsets(2)
    assert all(offsets[(k + 3) % 6] == (-offsets[k][0], -offsets[k][1]) for k in range(6))


@pytest.mark.parametrize(
    ('neighborhood', 'match'),
    [
        (RadiusNeighborhood(1, metric='taxicab'), 'metric'),
        (RadiusNeighborhood(1, center='first'), 'centre'),
    ],
)
def test_radius_neighbourhood_rejects_invalid_settings(neighborhood, match):
    """Test that an unknown metric and a centre that is not a boolean are rejected."""
    with pytest.raises(ValidationError, match=match):
        neighborhood.prepare()


@pytest.mark.parametrize(
    ('neighborhood', 'n_dim', 'match'),
    [
        (RadiusNeighborhood(1, metric='hexagonal'), 3, 'two dimensions'),
        (RadiusNeighborhood(1, angle=math.pi / 2), 2, 'cone'),
    ],
)
def test_radius_neighbourhood_rejects_lattices_it_cannot_cover(neighborhood, n_dim, match):
    """Test that a hexagonal neighbourhood outside two dimensions and a cone give no lattice offsets."""
    with pytest.raises(ValidationError, match=match):
        neighborhood._get_offsets(n_dim)
