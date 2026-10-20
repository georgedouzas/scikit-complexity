"""Test the stock-and-flow models of engineered processes and of the diffusion of innovations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import pytest

from skcomplex.core.analysis import loops
from skcomplex.core.runtime import ConservationMonitor, Simulation
from skcomplex.models.economics.innovation import bass_diffusion
from skcomplex.models.engineering.processes import bathtub, food_relief, irrigation, rice_milling, solid_waste


def _run(system, dt, until):
    """Return the trajectory of a system under a conservation monitor, which raises on any leak."""
    return Simulation(system, dt=dt, observers=[ConservationMonitor()]).run(until=until).trajectory_


def test_bathtub_records_its_flows():
    """Test that the bathtub records the faucet and the drain and conserves water."""
    trajectory = _run(bathtub(), 0.5, 10)
    assert trajectory['tub.drain'][0] == pytest.approx(5.0)
    assert trajectory['tub.faucet'][0] == pytest.approx(0.0)
    assert trajectory['tub.faucet'][-1] == pytest.approx(5.0)
    assert trajectory['tub.water'][-1] == pytest.approx(25.0)


def test_bass_diffusion_matches_closed_form():
    """Test that the adopters follow the closed-form Bass curve and that the market is conserved."""
    p, q, market = 0.03, 0.38, 1000.0
    trajectory = _run(bass_diffusion(market, p=p, q=q), 0.25, 20)
    times = np.asarray(trajectory.times)
    decay = np.exp(-(p + q) * times)
    expected = market * (1 - decay) / (1 + q / p * decay)
    np.testing.assert_allclose(trajectory['market.A'], expected, atol=1e-2 * market)
    np.testing.assert_allclose(trajectory['market.A'] + trajectory['market.P'], market)
    peak = times[np.argmax(trajectory['market.adoption'])]
    assert peak == pytest.approx(np.log(q / p) / (p + q), abs=0.25)


def test_rice_milling_passes_the_harvest_through():
    """Test that over the last years the mill processes the mean harvest and sells its outturn as rice."""
    trajectory = _run(rice_milling(harvest=2000.0, outturn=0.65), 0.25, 120)
    last_years = np.asarray(trajectory.times) >= 72
    assert trajectory['mill.milling'][last_years].mean() == pytest.approx(2000.0, rel=0.05)
    assert trajectory['mill.rice_sales'][last_years].mean() == pytest.approx(0.65 * 2000.0, rel=0.05)
    np.testing.assert_allclose(trajectory['mill.husk_output'], 0.2 / 0.65 * trajectory['mill.milled_rice'])
    assert trajectory['mill.paddy'].max() > 2 * trajectory['mill.paddy'].min()


def test_solid_waste_fills_the_landfill_then_dumps():
    """Test that the landfill fills up to its capacity before any waste is dumped."""
    trajectory = _run(solid_waste(landfill_capacity=2e6), 0.1, 40)
    landfill, dumped = trajectory['city.landfill'], trajectory['city.dumped']
    assert landfill.max() <= 2e6 * (1 + 1e-6)
    assert landfill[-1] == pytest.approx(2e6, rel=1e-3)
    assert dumped[landfill < 0.8 * 2e6].max() == pytest.approx(0.0, abs=1e-6)
    assert dumped[-1] > 0
    assert [loop.polarity for loop in loops(solid_waste()).loops].count('reinforcing') == 1


def test_food_relief_delays_then_meets_the_need():
    """Test that the relief reaches the people only after the shipping delay and then meets the need."""
    trajectory = _run(food_relief(onset=10.0, shipping_delay=14.0), 0.5, 120)
    times = np.asarray(trajectory.times)
    need = 1e5 * 5e-4
    assert trajectory['relief.distribution'][times <= 13].max() < 0.1 * need
    assert trajectory['relief.distribution'][-1] == pytest.approx(need, rel=1e-2)
    assert trajectory['relief.warehouse'].max() > 7 * need


def test_irrigation_pumps_groundwater_in_the_dry_season():
    """Test that the farmers pump the aquifer only when the reservoir runs low, drawing the aquifer down."""
    trajectory = _run(irrigation(), 1.0, 730)
    pumping, reservoir = trajectory['scheme.pumping'], trajectory['scheme.reservoir']
    assert pumping.max() > 0
    assert reservoir[pumping > 0].max() < 10.0
    assert trajectory['scheme.groundwater'][-1] < trajectory['scheme.groundwater'][0]
    assert irrigation(pump_capacity=0.0).elements[0].pump_capacity == 0.0
