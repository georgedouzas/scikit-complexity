"""Test the holonomy score spec."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import HolonomyScoreSpec, MatrixGroupSpec


def test_holonomy_score_keeps_its_group_and_hits():
    """Test that a valid score keeps its group and its number of hits."""
    score = HolonomyScoreSpec(np.sum, MatrixGroupSpec('SU', 3), hits=3)._validated()
    assert (score.group.degree, score.hits) == (3, 3)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'function': 1.0, 'group': MatrixGroupSpec()}, 'callable'),
        ({'function': np.sum, 'group': 'SU(2)'}, 'MatrixGroupSpec'),
        ({'function': np.sum, 'group': MatrixGroupSpec(), 'hits': 0}, 'at least one move'),
    ],
)
def test_holonomy_score_rejects_what_it_cannot_run(arguments, match):
    """Test that a function that is not callable, a group that is no spec, or no hits raise an error."""
    with pytest.raises(ValidationError, match=match):
        HolonomyScoreSpec(**arguments)._validated()
