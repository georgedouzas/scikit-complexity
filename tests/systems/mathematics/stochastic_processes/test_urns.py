"""Test the `_urns` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import pytest
from scipy import stats

from skcomplex.systems.mathematics.stochastic_processes import (
    polya_urn,
)

from ..conftest import _run


def test_polya_urn_shares_tend_to_a_uniform_limit():
    """Test that the red shares of urns started with one ball of each color are uniform with mean 1/2."""
    trajectory = _run(polya_urn(300), 60)
    red, blue = trajectory.values('urns.red')[-1], trajectory.values('urns.blue')[-1]
    assert (red + blue).mean() == pytest.approx(62, rel=0.05)
    shares = red / (red + blue)
    assert shares.mean() == pytest.approx(0.5, abs=0.05)
    assert stats.kstest(shares, 'uniform').pvalue > 0.01
