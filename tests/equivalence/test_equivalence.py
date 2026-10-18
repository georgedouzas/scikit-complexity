"""Test that every recorded model and documentation example gives the results recorded before the move to Sage."""

import numpy as np
import pytest

from ._cases import BASELINE, results


@pytest.fixture(scope='module')
def current():
    """Return the current results."""
    return results()


@pytest.mark.parametrize('name', sorted(np.load(BASELINE).files))
def test_matches_recorded_results(name, current):
    """Test that a case matches its recorded result within a relative tolerance of 1e-10."""
    recorded = np.load(BASELINE)[name]
    assert current[name].shape == recorded.shape
    np.testing.assert_allclose(current[name], recorded, rtol=1e-10, atol=1e-12)
