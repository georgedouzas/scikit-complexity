"""Test the stock-and-flow models of resources, food, and net energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import pytest

from skcomplex.core.runtime import ConservationMonitor, Simulation
from skcomplex.models.earth.climate import energy_return
from skcomplex.models.economics.resources import (
    cocoa,
    fishery_capital,
    food_security,
    hilsa_fishery,
    oil_economy,
)


def _run(system, dt, until):
    """Return the trajectory of a system under a conservation monitor, which raises on any leak."""
    return Simulation(system, dt=dt, observers=[ConservationMonitor()]).run(until=until).trajectory_


def _peak(trajectory, name):
    """Return the time at which a recorded series peaks."""
    return np.asarray(trajectory.times)[np.argmax(trajectory[name])]


def test_cocoa_booms_and_busts_without_a_subsidy():
    """Test that the cocoa area booms, then collapses under the insects, unless a full subsidy keeps the shade."""
    bust = _run(cocoa(), 0.25, 50)
    area = bust['cocoa.area']
    assert area.max() > 10 * area[0]
    assert 5 < _peak(bust, 'cocoa.area') < 25
    assert area[-1] < 0.05 * area.max()
    assert bust['cocoa.insects'][-1] > 90
    sustained = _run(cocoa(subsidy=1.0), 0.25, 50)['cocoa.area']
    assert sustained[-1] > 0.9 * sustained.max()


def test_hilsa_fishery_settles_and_collapses_with_the_catch():
    """Test that the stock settles at the spawning-ground limit, gains from a lower juvenile catch, and collapses."""
    base = _run(hilsa_fishery(), 0.5, 240)['hilsa.spawners']
    assert np.ptp(base[-24:]) < 1e-3 * base[-1]
    protected = _run(hilsa_fishery(juvenile_harvest=0.2), 0.5, 240)['hilsa.spawners']
    assert protected[-1] > 2 * base[-1]
    overfished = _run(hilsa_fishery(adult_harvest=0.9), 0.5, 240)['hilsa.spawners']
    assert overfished[-1] < 0.1 * overfished[0]


def test_food_security_falls_and_recovers_with_policy():
    """Test that self-sufficiency falls as the population grows and that research and extension raise it."""
    baseline = _run(food_security(), 0.25, 50)
    sufficiency = baseline['rice.self_sufficiency']
    assert sufficiency[-1] < 0.4 * sufficiency[0]
    assert 0 < _peak(baseline, 'rice.rice_area') < 10
    withdrawn = _run(food_security(subsidy=0.0), 0.25, 50)['rice.self_sufficiency']
    assert withdrawn[-1] < sufficiency[-1]
    policy = food_security(subsidy=0.335, bio_target=0.5, research=1.0, training_target=1.0, intensity_target=2.0)
    assert _run(policy, 0.25, 50)['rice.self_sufficiency'][-1] > sufficiency[-1]


def test_energy_return_net_energy_peaks_before_gross_energy():
    """Test that the net energy per person peaks before the net energy, which peaks before the gross energy."""
    trajectory = _run(energy_return(), 1.0, 250)
    gross = _peak(trajectory, 'energy.gross_energy')
    net = _peak(trajectory, 'energy.net_energy')
    per_capita = _peak(trajectory, 'energy.net_per_capita')
    assert per_capita < net < gross
    np.testing.assert_allclose(
        trajectory['energy.energy_cost'] + trajectory['energy.net_energy'],
        trajectory['energy.gross_energy'],
    )
    assert trajectory['energy.eroi'][-1] < trajectory['energy.eroi'][0] / 2


@pytest.mark.parametrize(('factory', 'label'), [(oil_economy, 'oil'), (fishery_capital, 'fishery')])
def test_resource_capital_conserves_its_flows(factory, label):
    """Test that Meadows' resource models move their resource and capital only through flows."""
    trajectory = _run(factory(), 1.0, 100)
    assert trajectory[f'{label}.investment'].min() >= 0
    assert trajectory[f'{label}.depreciation'][0] == pytest.approx(5.0 / 20.0)
