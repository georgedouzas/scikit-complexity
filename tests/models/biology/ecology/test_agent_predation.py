"""Test the `_agent_predation` module of `skcomplex.models.biology.ecology`."""

from skcomplex.models.biology.ecology import (
    agent_based_lotka_volterra,
)

from ..conftest import _run_until


def test_agent_based_lotka_volterra_cycles():
    """Test that agent-based prey and predators persist and cycle."""
    trajectory = _run_until(agent_based_lotka_volterra(random_state=0), 100, random_state=0)
    prey = trajectory.values('prey.count')
    assert trajectory.values('predators.count').min() > 0
    assert prey.max() > 2 * prey[30:].min()
