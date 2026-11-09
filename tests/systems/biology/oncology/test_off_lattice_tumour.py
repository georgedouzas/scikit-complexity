"""Test the `_off_lattice_tumour` module of `skcomplex.systems.biology.oncology`."""

import numpy as np
from scipy.spatial.distance import pdist

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.oncology import OffLatticeTumourGrowthSystem


def test_tumour_grows_faster_after_the_switch():
    """Test that the tumour area grows faster per day after day 10, when `g_age` drops from 20 to 5 hours."""
    system = OffLatticeTumourGrowthSystem(g_age1=20.0, g_age2=5.0, tau=10.0)
    trajectory = simulate(system, until=720, dt=24, random_state=0)
    area = np.asarray(trajectory.values('tumour_area'))
    before, after = (area[10] - area[0]) / 10, (area[20] - area[10]) / 10
    assert after > before > 0


def test_springs_keep_cells_apart():
    """Test that no two cell centres come closer than half the natural spring length over 10 days."""
    trajectory = simulate(OffLatticeTumourGrowthSystem(), until=240, dt=24, random_state=0)
    for locations in trajectory.agent_locations('cells'):
        assert pdist(locations).min() > 0.5


def test_tumour_records_its_divisions_and_invasions():
    """Test that a run that records holds the named division draws of the cancer cells."""
    trajectory = simulate(OffLatticeTumourGrowthSystem(side=8.0), until=48, random_state=0, record_draws=True)
    assert {'growth.divide', 'growth.invade'} <= set(trajectory.draws.draws)
