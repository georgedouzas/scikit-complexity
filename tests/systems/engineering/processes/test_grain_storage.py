"""Test the `_grain_storage` module of `skcomplex.systems.engineering.processes`."""

import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.processes import GrainStorageSystem


def test_grain_storage_oscillates_and_settles_where_the_release_balances_the_receiving():
    """Test that the stock overshoots the desired stock, oscillates, and settles at `6000 / (1 + 0.025 * 5)`."""
    trajectory = simulate(GrainStorageSystem(), until=300)
    stock, procurement = trajectory.values('store.stock'), trajectory.values('store.procurement')
    assert stock.max() > 7000
    assert procurement.min() < 0
    settled = 6000 / (1 + 0.025 * 5)
    assert stock[trajectory.times > 20].min() < settled - 300
    assert stock[-1] == pytest.approx(settled, rel=1e-3)
    assert trajectory.values('store.receiving')[-1] == pytest.approx(trajectory.values('store.release')[-1], rel=1e-3)
