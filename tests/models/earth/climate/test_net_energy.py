"""Test the `_net_energy` module of `skcomplex.models.earth.climate`."""

import numpy as np

from skcomplex.models.earth.climate import energy_return

from ...conftest import _peak, _run_monitored


def test_energy_return_net_energy_peaks_before_gross_energy():
    """Test that the net energy per person peaks before the net energy, which peaks before the gross energy."""
    trajectory = _run_monitored(energy_return(), 1.0, 250)
    gross = _peak(trajectory, 'energy.gross_energy')
    net = _peak(trajectory, 'energy.net_energy')
    per_capita = _peak(trajectory, 'energy.net_per_capita')
    assert per_capita < net < gross
    np.testing.assert_allclose(
        trajectory['energy.energy_cost'] + trajectory['energy.net_energy'],
        trajectory['energy.gross_energy'],
    )
    assert trajectory['energy.eroi'][-1] < trajectory['energy.eroi'][0] / 2
