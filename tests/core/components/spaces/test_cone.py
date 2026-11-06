"""Test the `_cone` module of `skcomplex.core.components.spaces`."""

import math

import pytest

from skcomplex.core.components.spaces import ConeNeighborhood
from skcomplex.core.errors import ValidationError


@pytest.mark.parametrize(
    ('radius', 'angle', 'message'),
    [(0.0, 1.0, 'positive radius'), (1.0, 0.0, 'half-angle'), (1.0, 4.0, 'half-angle')],
)
def test_cone_neighbourhood_checks_its_radius_and_angle(radius, angle, message):
    """Test that a cone needs a positive radius and a half-angle in (0, pi]."""
    with pytest.raises(ValidationError, match=message):
        ConeNeighborhood(radius, angle).prepare()


def test_cone_neighbourhood_takes_the_whole_disk():
    """Test that a cone of half-angle pi is accepted."""
    assert ConeNeighborhood(1.0, math.pi).angle == math.pi
