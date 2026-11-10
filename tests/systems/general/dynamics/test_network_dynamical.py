"""Test the `_network_dynamical` module of `skcomplex.systems.general.dynamics`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.general.dynamics import NetworkDynamicalSystem


def diffusion(states: dict[str, object]) -> NetworkDynamicalSystem:
    """Return diffusion on a path of three nodes with the given initial values."""
    return NetworkDynamicalSystem(
        nx.path_graph(3),
        states=states,
        coupling={'u': lambda s: s.member.D * (s.other.u - s.member.u)},
        parameters={'D': 1.0},
    )


def test_network_dynamical_system_diffuses_on_a_path():
    """Test that diffusion on a path spreads to `1/3` on every node and keeps the total."""
    trajectory = simulate(diffusion({'u': [1.0, 0.0, 0.0]}), until=30.0, dt=1.0)
    values = trajectory.get_values('nodes.u')
    np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(values[-1], 1 / 3, atol=1e-4)


def test_network_dynamical_system_asks_for_a_graph():
    """Test that preparing without a graph asks for one."""
    system = NetworkDynamicalSystem(states={'u': 0.0}, coupling={'u': lambda s: s.other.u - s.member.u})
    with pytest.raises(ValidationError, match='graph'):
        system.prepare()


def test_network_dynamical_system_broadcasts_a_number_to_every_node():
    """Test that a number is the initial value of every node and an array gives one value per node."""
    broadcast = simulate(diffusion({'u': 0.5}), until=1.0, dt=1.0).get_values('nodes.u')
    np.testing.assert_allclose(broadcast, 0.5)
    per_node = simulate(diffusion({'u': [0.0, 1.0, 2.0]}), until=0.0, dt=1.0).get_values('nodes.u')
    np.testing.assert_allclose(per_node[0], [0.0, 1.0, 2.0])


def test_network_dynamical_system_rejects_the_coupling_of_an_unknown_state():
    """Test that a coupling of an undeclared state names it."""
    system = NetworkDynamicalSystem(nx.path_graph(2), states={'u': 0.0}, coupling={'v': lambda s: 0.0})
    with pytest.raises(ValidationError, match="'v'"):
        system.prepare()


def test_network_dynamical_system_tags():
    """Test that the tags name a rate on a network in continuous time."""
    tags = diffusion({'u': 0.0}).get_tags()
    assert 'rate' in tags.laws
    assert tags.time == frozenset({'continuous'})
    assert tags.randomness == 'deterministic'
