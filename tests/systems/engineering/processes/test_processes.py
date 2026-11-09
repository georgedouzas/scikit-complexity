"""Test the `_processes` module of `skcomplex.systems.engineering.processes`."""

import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.processes import BathtubSystem, CascadeClassifierSystem

from ...conftest import _run_monitored


def test_bathtub_drains_then_holds():
    """Test that the tub loses five gallons per minute until the faucet opens, and then holds its level."""
    trajectory = simulate(BathtubSystem().prepare().system_, until=10, dt=0.5)
    assert trajectory.get_values('tub.water')[10] == pytest.approx(25.0)
    assert trajectory.get_values('tub.water')[-1] == pytest.approx(25.0)


@pytest.mark.parametrize(('n_stages', 'k', 'feed_stage'), [(10, 0.6, None), (10, 0.45, 3)])
def test_cascade_classifier_fractional_extraction(n_stages, k, feed_stage):
    """Test that the share of the feed in the fine product matches Barsky's fractional extraction."""
    system = CascadeClassifierSystem(n_stages, k=k, feed_stage=feed_stage).prepare().system_
    trajectory = simulate(system, until=400, dt=10.0)
    fine = trajectory.get_values('fine.m')
    entry = (n_stages + 1) // 2 if feed_stage is None else feed_stage
    ratio = (1 - k) / k
    expected = (1 - ratio ** (n_stages + 1 - entry)) / (1 - ratio ** (n_stages + 1))
    assert (fine[-1] - fine[-11]) / 100 == pytest.approx(expected, rel=1e-4)


def test_bathtub_records_its_flows():
    """Test that the bathtub records the faucet and the drain and conserves water."""
    trajectory = _run_monitored(BathtubSystem().prepare().system_, 0.5, 10)
    assert trajectory.get_values('tub.drain')[0] == pytest.approx(5.0)
    assert trajectory.get_values('tub.faucet')[0] == pytest.approx(0.0)
    assert trajectory.get_values('tub.faucet')[-1] == pytest.approx(5.0)
    assert trajectory.get_values('tub.water')[-1] == pytest.approx(25.0)
