"""Test the `_angiogenesis` module of `skcomplex.systems.biology.oncology`."""

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.biology.oncology import tumour_angiogenesis


def test_tumour_recruits_vessels_and_grows():
    """Test that vessel tips move toward the hypoxic tumour, which then grows."""
    trajectory = Simulation(tumour_angiogenesis(), dt=1, random_state=0).prepare().run(until=120).trajectory_
    assert trajectory.agents('cells/tumour', 50).values['hypoxic'].all()
    assert trajectory.agents('cells/sprouts', 120).locations[:, 0].mean() > 15
    assert len(trajectory.agents('cells/tumour', 120).ids) > 2 * len(trajectory.agents('cells/tumour', 0).ids)
