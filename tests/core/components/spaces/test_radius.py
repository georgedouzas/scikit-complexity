"""Test the `_radius` module of `skcomplex.core.components.spaces`."""

import math

import pytest

from skcomplex.core.components.spaces import RadiusNeighborhood
from skcomplex.core.errors import ValidationError


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
