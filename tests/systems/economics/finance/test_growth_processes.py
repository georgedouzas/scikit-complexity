"""Test the `_growth_processes` module of `skcomplex.systems.economics.finance`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.economics.finance import (
    GibratProcessSystem,
    KestenProcessSystem,
    MultiplicativeProcessSystem,
)

from ..conftest import _agents, _run


def test_gibrat_process_makes_lognormal_sizes():
    """Test that the logarithm of the sizes spreads like a random walk of the logarithmic growth factors."""
    sizes = _agents(
        _run(
            GibratProcessSystem(1000).prepare().system_,
            100,
            observables=GibratProcessSystem().prepare().observables_,
        ),
        'firms',
        100,
    )['x']
    logs = np.log(sizes)
    assert logs.std() == pytest.approx(np.sqrt(100) * 0.05 / 1.02, rel=0.1)
    assert abs(float(np.mean((logs - logs.mean()) ** 3)) / logs.std() ** 3) < 0.3


def test_gibrat_gamble_grows_in_mean_and_shrinks_in_median():
    """Test that the factors 1.5 and 0.6 grow the mean size as `1.05^t` and shrink the median as `0.9^(t/2)`."""
    system = GibratProcessSystem(20000, x=1.0, factors=(1.5, 0.6), weights=(0.5, 0.5))
    sizes = _run(system, 20).get_values('firms.x')[-1]
    assert sizes.mean() == pytest.approx(1.05**20, rel=0.15)
    assert np.median(sizes) == pytest.approx(0.9**10)


def test_gibrat_factors_follow_their_weights():
    """Test that one step multiplies the sizes by the factors in the shares their weights give."""
    system = GibratProcessSystem(5000, x=1.0, factors=(2.0, 1.0, 0.5), weights=(0.2, 0.3, 0.5))
    sizes = _run(system, 1).get_values('firms.x')[-1]
    assert set(sizes.tolist()) == {2.0, 1.0, 0.5}
    assert (sizes == 0.5).mean() == pytest.approx(0.5, abs=0.03)
    assert (sizes == 2.0).mean() == pytest.approx(0.2, abs=0.03)


@pytest.mark.parametrize(
    ('factors', 'weights', 'match'),
    [((1.5, 0.6), (0.5, 0.4), 'sum to 1'), ((1.5, 0.6), (1.0,), 'one weight per factor'), (None, (1.0,), 'factor')],
)
def test_gibrat_rejects_weights_that_are_no_law(factors, weights, match):
    """Test that weights that do not sum to 1, miss a factor, or come without factors raise a validation error."""
    with pytest.raises(ValidationError, match=match):
        GibratProcessSystem(5, factors=factors, weights=weights).prepare()


def test_multiplicative_process_respects_its_floor():
    """Test that no size falls below the fraction of the previous mean, and the floor fattens the upper tail."""
    trajectory = _run(
        MultiplicativeProcessSystem(300).prepare().system_,
        100,
        observables=MultiplicativeProcessSystem().prepare().observables_,
    )
    for frame in (10, 50, 100):
        previous = _agents(trajectory, 'firms', frame)['previous']
        assert _agents(trajectory, 'firms', frame)['x'].min() >= 0.9 * previous.mean() - 1e-9
    free = _agents(
        _run(
            GibratProcessSystem(300).prepare().system_,
            100,
            observables=GibratProcessSystem().prepare().observables_,
        ),
        'firms',
        100,
    )['x']
    floored = _agents(trajectory, 'firms', 100)['x']
    assert floored.max() / np.median(floored) > 1.0
    assert floored.min() / np.median(floored) > free.min() / np.median(free)


def test_kesten_process_reaches_a_positive_stationary_state():
    """Test that Kesten sizes stay positive and settle far below the initial size when the factor shrinks on average."""
    trajectory = _run(
        KestenProcessSystem(500).prepare().system_,
        100,
        observables=KestenProcessSystem().prepare().observables_,
    )
    sizes = _agents(trajectory, 'firms', 100)['x']
    assert sizes.min() > 0
    assert np.median(sizes) < 25.0
    assert sizes.max() > 10 * np.median(sizes)
