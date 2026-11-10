"""Test the `_angiogenesis` module of `skcomplex.systems.biology.oncology`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.oncology import TumourAngiogenesisSystem


def test_tumour_recruits_vessels_and_grows():
    """Test that vessel tips move toward the hypoxic tumour, which then grows."""
    trajectory = simulate(TumourAngiogenesisSystem(), dt=1, random_state=0, until=120)
    assert trajectory.get_agents('cells/tumour', 50).values['hypoxic'].all()
    assert trajectory.get_agents('cells/sprouts', 120).locations[:, 0].mean() > 15
    assert len(trajectory.get_agents('cells/tumour', 120).ids) > 2 * len(trajectory.get_agents('cells/tumour', 0).ids)
