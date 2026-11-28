"""Test the `_urns` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import pytest
from scipy import stats

from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.stochastic_processes import (
    PolyaUrnSystem,
)

from ..conftest import _run


def test_polya_urn_shares_tend_to_a_uniform_limit():
    """Test that the shares of the first colour of urns started with one ball of each colour are uniform."""
    balls = _run(PolyaUrnSystem(300), 60).get_values('urns.balls')[-1]
    assert balls.shape == (300, 2)
    assert balls.sum(axis=1).mean() == pytest.approx(62, rel=0.05)
    shares = balls[:, 0] / balls.sum(axis=1)
    assert shares.mean() == pytest.approx(0.5, abs=0.05)
    assert stats.kstest(shares, 'uniform').pvalue > 0.01


def test_polya_urn_draws_once_per_step_in_discrete_time():
    """Test that two discrete draws give two balls of the first colour in about a third of the urns."""
    balls = _run(PolyaUrnSystem(3000, time='discrete'), 2).get_values('urns.balls')[-1]
    assert set(balls.sum(axis=1).tolist()) == {4.0}
    assert (balls[:, 0] == 3).mean() == pytest.approx(1 / 3, abs=0.03)


def test_polya_urn_of_three_colours_has_a_dirichlet_limit():
    """Test that the share of the third colour of `balls=(1, 1, 2)` has mean 1/2 and variance 1/20."""
    balls = _run(PolyaUrnSystem(3000, balls=(1, 1, 2), time='discrete'), 400).get_values('urns.balls')[-1]
    assert balls.shape == (3000, 3)
    shares = balls[:, 2] / balls.sum(axis=1)
    assert shares.mean() == pytest.approx(1 / 2, abs=0.02)
    assert shares.var() == pytest.approx(1 / 20, rel=0.1)


def test_polya_urn_of_three_colours_draws_at_poisson_times():
    """Test that continuous urns of three colours gain one ball per unit of time on average."""
    balls = _run(PolyaUrnSystem(500, balls=(1, 1, 2)), 30).get_values('urns.balls')[-1]
    assert balls.sum(axis=1).mean() == pytest.approx(34, rel=0.05)


def test_polya_urn_rejects_an_unknown_time():
    """Test that an unknown time raises a validation error naming the times."""
    with pytest.raises(ValidationError, match='discrete'):
        PolyaUrnSystem(time='daily').prepare()


def test_polya_urn_needs_two_colours():
    """Test that an urn of one colour raises a validation error asking for two or more."""
    with pytest.raises(ValidationError, match='two or more colours'):
        PolyaUrnSystem(balls=(3,)).prepare()
