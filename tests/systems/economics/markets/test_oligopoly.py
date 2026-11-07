"""Test the `_oligopoly` module of `skcomplex.systems.economics.markets`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation, simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.economics.markets import OligopolyEquilibriaSystem


def _distribution(prices: np.ndarray, alpha: float, beta: float, n_firms: int) -> np.ndarray:
    """Return the equilibrium price distribution of Forrest (2018 Lemma 20.1)."""
    return 1 - (alpha * (1 - prices) / (beta * prices)) ** (1 / (n_firms - 1))


@pytest.mark.parametrize(('n_firms', 'alpha'), [(2, 0.25), (3, 0.2)])
def test_oligopoly_equilibria_mixed_price(n_firms: int, alpha: float) -> None:
    """Test that the static system gives the equilibrium distribution on its support, at the profit `alpha`."""
    trajectory = simulate(OligopolyEquilibriaSystem(n_firms, alpha=alpha), until=1.0)
    beta = 1 - n_firms * alpha
    prices = trajectory.values('prices.price')[0]
    assert prices[0] == pytest.approx(alpha / (alpha + beta))
    assert prices[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(
        trajectory.values('prices.distribution')[0],
        _distribution(prices, alpha, beta, n_firms),
        atol=1e-6,
    )
    assert trajectory.values('lowest_profit')[0] == pytest.approx(alpha)
    assert trajectory.values('highest_profit')[0] == pytest.approx(alpha)


def test_oligopoly_equilibria_cournot_quantities() -> None:
    """Test that the Cournot quantities are the solution of equation 21.6 of Forrest (2018)."""
    system = OligopolyEquilibriaSystem(demand=10.0, c_i=1.0, c_j=2.0, lambda_i=0.5, lambda_j=0.0).prepare().system_
    trajectory = Simulation(system).prepare().run(1.0).trajectory_
    assert trajectory.values('cournot.q_i')[0] == pytest.approx((10.0 - 2 * 1.0 / 1.5 + 2.0) / 3)
    assert trajectory.values('cournot.q_j')[0] == pytest.approx((10.0 - 2 * 2.0 + 1.0 / 1.5) / 3)


def test_oligopoly_equilibria_fictitious_play() -> None:
    """Test that the prices of fictitious play are distributed near the equilibrium in the Kolmogorov distance."""
    distances = []
    for seed in range(5):
        trajectory = simulate(OligopolyEquilibriaSystem(dynamics='fictitious-play'), until=2000, random_state=seed)
        charged = np.concatenate([trajectory.values('lower_price')[1:], trajectory.values('higher_price')[1:]])
        grid = np.unique(charged)
        empirical = np.array([np.mean(charged <= price) for price in grid])
        distances.append(np.abs(empirical - _distribution(grid, 0.25, 0.5, 2)).max())
    assert max(distances) < 0.2


def test_oligopoly_equilibria_unknown_dynamics() -> None:
    """Test that unknown dynamics and fictitious play of other than two firms are refused."""
    with pytest.raises(ValidationError, match='dynamics'):
        OligopolyEquilibriaSystem(dynamics='best-reply').prepare()
    with pytest.raises(ValidationError, match='dynamics'):
        OligopolyEquilibriaSystem(3, alpha=0.2, dynamics='fictitious-play').prepare()
