"""Test the `_land_use` module of `skcomplex.systems.economics.resources`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.resources import VonThunenSystem

YIELDS = (4.0, 2.0, 1.0)
MARGINS = (2.0, 2.5, 3.0)
RATES = (0.2, 0.2, 0.2)


def _land_use(rates: tuple[float, ...] = RATES) -> tuple[np.ndarray, np.ndarray]:
    """Return the crop and the distance of every cell of the default plain at given transport rates."""
    trajectory = simulate(VonThunenSystem(rates=rates).prepare().system_, until=1.0)
    return trajectory.values('plain.crop')[0].ravel(), trajectory.values('plain.distance')[0].ravel()


def test_von_thunen_rings() -> None:
    """Test that the land use forms rings, with boundaries where the bids of neighbouring crops are equal."""
    crop, distance = _land_use()
    grown = crop > 0
    order = np.argsort(distance[grown])
    assert np.all(np.diff(crop[grown][order]) >= 0)
    assert np.all(distance[~grown] > distance[grown].max())
    for first in range(2):
        bids = YIELDS[first] * MARGINS[first] - YIELDS[first + 1] * MARGINS[first + 1]
        boundary = bids / (YIELDS[first] * RATES[first] - YIELDS[first + 1] * RATES[first + 1])
        inner, outer = distance[crop == first + 1].max(), distance[crop == first + 2].min()
        assert abs(inner - boundary) < 1.0
        assert abs(outer - boundary) < 1.0


@pytest.mark.slow
def test_von_thunen_higher_transport_rate_shrinks_a_ring() -> None:
    """Test that raising the transport rate of a crop shrinks its ring."""
    crop, _ = _land_use()
    faster, _ = _land_use((0.3, 0.2, 0.2))
    assert np.sum(faster == 1) < np.sum(crop == 1)
