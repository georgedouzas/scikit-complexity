"""Test the region score spec."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.physiology import RegionScoreSpec


def _area(labels):
    """Return a volume constraint of strength 1 and target 40."""
    del labels
    return 1.0, 40.0


def test_region_score_keeps_its_constraints_and_hits():
    """Test that a valid score keeps its constraints and its number of hits."""
    area = _area
    score = RegionScoreSpec(np.maximum, volume=area, hits=3)
    assert (score.volume, score.surface, score.hits) == (area, None, 3)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'contact': 1.0}, 'contact of a region score'),
        ({'contact': np.maximum, 'volume': 40.0}, 'volume of a region score'),
        ({'contact': np.maximum, 'surface': 'perimeter'}, 'surface of a region score'),
        ({'contact': np.maximum, 'hits': 0}, 'at least one copy'),
    ],
)
def test_region_score_rejects_what_it_cannot_run(arguments, match):
    """Test that a contact or a constraint that is not callable, or no hits, raise an error."""
    with pytest.raises(ValidationError, match=match):
        RegionScoreSpec(**arguments)._check()
