"""Test the `_barter` module of `skcomplex.systems.economics.markets`."""

import numpy as np
import pytest

from skcomplex.systems import create_simulation
from skcomplex.systems.economics.markets import contract_curve


@pytest.mark.parametrize(
    ('chocolate', 'cigarettes', 'weights'),
    [((9.0, 1.0), (2.0, 8.0), (0.6, 0.3)), ((5.0, 5.0), (1.0, 9.0), (0.2, 0.7)), ((2.0, 6.0), (7.0, 3.0), (0.5, 0.5))],
)
def test_barter_reaches_the_contract_curve(chocolate, cigarettes, weights):
    """Test that Edgeworth barter ends on the contract curve within 2% of the total endowment."""
    simulation = (
        create_simulation('barter', chocolate=chocolate, cigarettes=cigarettes, weights=weights).prepare().run(until=5)
    )
    final = simulation.trajectory_.agents('barterers', 5).values
    expected = contract_curve(final['chocolate'][0], *weights, sum(chocolate), sum(cigarettes))
    assert abs(final['cigarettes'][0] - expected) < 0.02 * sum(cigarettes)
    assert final['chocolate'].sum() == pytest.approx(sum(chocolate))
    assert simulation.trajectory_.values('exchange.volume')[-1] == 0


def test_red_cross_barter_settles_on_one_price():
    """Test that the prices of the barter among many prisoners converge."""
    trajectory = create_simulation('barter', n_barterers=100).prepare().run(until=20).trajectory_
    dispersion = trajectory.values('exchange.dispersion')
    assert np.nanmean(dispersion[-5:]) < 0.1 * dispersion[1]


def test_barter_needs_two_barterers():
    """Test that barter rejects a single barterer."""
    with pytest.raises(ValueError, match='at least two'):
        create_simulation('barter', n_barterers=1).prepare().run()
