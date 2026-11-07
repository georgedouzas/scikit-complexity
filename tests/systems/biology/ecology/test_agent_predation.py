"""Test the `_agent_predation` module of `skcomplex.systems.biology.ecology`."""

from skcomplex.systems.biology.ecology import (
    AgentBasedLotkaVolterraSystem,
)

from ..conftest import _run_until


def test_agent_based_lotka_volterra_cycles():
    """Test that agent-based prey and predators persist and cycle."""
    trajectory = _run_until(AgentBasedLotkaVolterraSystem(random_state=0), 100, random_state=0)
    prey = trajectory.values('prey.count')
    assert trajectory.values('predators.count').min() > 0
    assert prey.max() > 2 * prey[30:].min()
