"""Test the `_systems` module of `skcomplex.systems.engineering.processes`."""

import numpy as np
import pytest

from skcomplex.core.analysis import LoopAnalysis
from skcomplex.systems import create_simulation
from skcomplex.systems.engineering.processes import food_relief, irrigation, rice_milling, solid_waste

from ...conftest import _run_monitored


def test_rice_milling_passes_the_harvest_through():
    """Test that over the last years the mill processes the mean harvest and sells its outturn as rice."""
    trajectory = _run_monitored(rice_milling(harvest=2000.0, outturn=0.65), 0.25, 120, 'rice_milling')
    last_years = np.asarray(trajectory.times) >= 72
    assert trajectory.values('mill.milling')[last_years].mean() == pytest.approx(2000.0, rel=0.05)
    assert trajectory.values('mill.rice_sales')[last_years].mean() == pytest.approx(0.65 * 2000.0, rel=0.05)
    np.testing.assert_allclose(
        trajectory.values('mill.husk_output'),
        0.2 / 0.65 * trajectory.values('mill.milled_rice'),
    )
    assert trajectory.values('mill.paddy').max() > 2 * trajectory.values('mill.paddy').min()


def test_solid_waste_fills_the_landfill_then_dumps():
    """Test that the landfill fills up to its capacity before any waste is dumped."""
    trajectory = _run_monitored(solid_waste(landfill_capacity=2e6), 0.1, 40)
    landfill, dumped = trajectory.values('city.landfill'), trajectory.values('city.dumped')
    assert landfill.max() <= 2e6 * (1 + 1e-6)
    assert landfill[-1] == pytest.approx(2e6, rel=1e-3)
    assert dumped[landfill < 0.8 * 2e6].max() == pytest.approx(0.0, abs=1e-6)
    assert dumped[-1] > 0
    assert LoopAnalysis().prepare(solid_waste()).run().polarities_.count('reinforcing') == 1


def test_food_relief_delays_then_meets_the_need():
    """Test that the relief reaches the people only after the shipping delay and then meets the need."""
    trajectory = _run_monitored(food_relief(onset=10.0, shipping_delay=14.0), 0.5, 120)
    times = np.asarray(trajectory.times)
    need = 1e5 * 5e-4
    assert trajectory.values('relief.distribution')[times <= 13].max() < 0.1 * need
    assert trajectory.values('relief.distribution')[-1] == pytest.approx(need, rel=1e-2)
    assert trajectory.values('relief.warehouse').max() > 7 * need


def test_irrigation_pumps_groundwater_in_the_dry_season():
    """Test that the farmers pump the aquifer only when the reservoir runs low, drawing the aquifer down."""
    trajectory = _run_monitored(irrigation(), 1.0, 730, 'irrigation')
    pumping, reservoir = trajectory.values('scheme.pumping'), trajectory.values('scheme.reservoir')
    assert pumping.max() > 0
    assert reservoir[pumping > 0].max() < 10.0
    assert trajectory.values('scheme.groundwater')[-1] < trajectory.values('scheme.groundwater')[0]
    assert irrigation(pump_capacity=0.0).elements[0].pump_capacity == 0.0


def test_grain_storage_oscillates_and_settles_where_the_release_balances_the_receiving():
    """Test that the stock overshoots the desired stock, oscillates, and settles at `6000 / (1 + 0.025 * 5)`."""
    trajectory = create_simulation('grain_storage').prepare().run(until=300).trajectory_
    stock, procurement = trajectory.values('store.stock'), trajectory.values('store.procurement')
    assert stock.max() > 7000
    assert procurement.min() < 0
    settled = 6000 / (1 + 0.025 * 5)
    assert stock[trajectory.times > 20].min() < settled - 300
    assert stock[-1] == pytest.approx(settled, rel=1e-3)
    assert trajectory.values('store.receiving')[-1] == pytest.approx(trajectory.values('store.release')[-1], rel=1e-3)
