"""Test the `_net_energy` module of `skcomplex.systems.physics.climate`."""

import numpy as np

from skcomplex.systems.physics.climate import EnergyReturnSystem

from ...conftest import _peak, _run_monitored


def test_energy_return_net_energy_peaks_before_gross_energy():
    """Test that the net energy per person peaks before the net energy, which peaks before the gross energy."""
    trajectory = _run_monitored(EnergyReturnSystem().prepare().system_, 1.0, 250, 'energy_return')
    gross = _peak(trajectory, 'energy.gross_energy')
    net = _peak(trajectory, 'energy.net_energy')
    per_capita = _peak(trajectory, 'energy.net_per_capita')
    assert per_capita < net < gross
    np.testing.assert_allclose(
        trajectory.values('energy.energy_cost') + trajectory.values('energy.net_energy'),
        trajectory.values('energy.gross_energy'),
    )
    assert trajectory.values('energy.eroi')[-1] < trajectory.values('energy.eroi')[0] / 2
