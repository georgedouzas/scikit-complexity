"""Test the `_urns` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import pytest
from scipy import stats

from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.stochastic_processes import (
    PolyaUrnSystem,
)

from ..conftest import _run


def test_polya_urn_shares_tend_to_a_uniform_limit():
    """Test that the red shares of urns started with one ball of each color are uniform with mean 1/2."""
    trajectory = _run(PolyaUrnSystem(300), 60)
    red, blue = trajectory.get_values('urns.red')[-1], trajectory.get_values('urns.blue')[-1]
    assert (red + blue).mean() == pytest.approx(62, rel=0.05)
    shares = red / (red + blue)
    assert shares.mean() == pytest.approx(0.5, abs=0.05)
    assert stats.kstest(shares, 'uniform').pvalue > 0.01


def test_polya_urn_draws_once_per_step_in_discrete_time():
    """Test that two discrete draws give two reds in about a third of the urns."""
    trajectory = _run(PolyaUrnSystem(3000, time='discrete'), 2)
    red, blue = trajectory.get_values('urns.red')[-1], trajectory.get_values('urns.blue')[-1]
    assert set((red + blue).tolist()) == {4.0}
    assert (red == 3).mean() == pytest.approx(1 / 3, abs=0.03)


def test_polya_urn_rejects_an_unknown_time():
    """Test that an unknown time raises a validation error naming the times."""
    with pytest.raises(ValidationError, match='discrete'):
        PolyaUrnSystem(time='daily').prepare()
