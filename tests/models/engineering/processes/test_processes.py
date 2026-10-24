"""Test the `_processes` module of `skcomplex.models.engineering.processes`."""

import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.engineering.processes import bathtub, cascade_classifier

from ...conftest import _run_monitored


def test_bathtub_drains_then_holds():
    """Test that the tub loses five gallons per minute until the faucet opens, and then holds its level."""
    trajectory = Simulation(bathtub(), dt=0.5).run(until=10).trajectory_
    assert trajectory['tub.water'][10] == pytest.approx(25.0)
    assert trajectory['tub.water'][-1] == pytest.approx(25.0)


@pytest.mark.parametrize(('n_stages', 'k', 'feed_stage'), [(10, 0.6, None), (10, 0.45, 3)])
def test_cascade_classifier_fractional_extraction(n_stages, k, feed_stage):
    """Test that the share of the feed in the fine product matches Barsky's fractional extraction."""
    system = cascade_classifier(n_stages, k=k, feed_stage=feed_stage)
    trajectory = Simulation(system, dt=10.0).run(until=400).trajectory_
    fine = trajectory['fine.m']
    entry = (n_stages + 1) // 2 if feed_stage is None else feed_stage
    ratio = (1 - k) / k
    expected = (1 - ratio ** (n_stages + 1 - entry)) / (1 - ratio ** (n_stages + 1))
    assert (fine[-1] - fine[-11]) / 100 == pytest.approx(expected, rel=1e-4)


def test_bathtub_records_its_flows():
    """Test that the bathtub records the faucet and the drain and conserves water."""
    trajectory = _run_monitored(bathtub(), 0.5, 10)
    assert trajectory['tub.drain'][0] == pytest.approx(5.0)
    assert trajectory['tub.faucet'][0] == pytest.approx(0.0)
    assert trajectory['tub.faucet'][-1] == pytest.approx(5.0)
    assert trajectory['tub.water'][-1] == pytest.approx(25.0)
