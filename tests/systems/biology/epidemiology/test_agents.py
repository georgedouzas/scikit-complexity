"""Test the `_agents` module of `skcomplex.systems.biology.epidemiology`."""

import networkx as nx
import numpy as np

from skcomplex.systems.biology.epidemiology import AgentBasedSIRSystem, NetworkSIRSystem, RandomWalkerEpidemicsSystem

from ..conftest import _run_until


def _health(trajectory, group, frame):
    """Return the health of the agents of a group at a frame."""
    return trajectory.agents(group, frame).values['health']


def test_agent_based_sir_spreads_more_with_higher_infection():
    """Test that the final size of the agent-based SIR system grows with the infection probability."""
    sizes = []
    for beta in (0.05, 0.8):
        system = (
            AgentBasedSIRSystem(30, 30, people=300, infected=5, beta=beta, gamma=0.05, random_state=0).prepare().system_
        )
        trajectory = _run_until(system, 200, random_state=0)
        sizes.append(np.sum(_health(trajectory, 'people', 200) == 3))
    assert sizes[1] > 2 * sizes[0]


def test_agent_based_sir_extensions():
    """Test that mortality kills, incubation delays infectiousness, and immunity wanes."""
    extended = {'mu': 0.05, 'alpha': 0.05, 'incubation': 3}
    system = (
        AgentBasedSIRSystem(20, 20, people=200, infected=10, beta=0.8, random_state=0, **extended).prepare().system_
    )
    trajectory = _run_until(system, 60, random_state=0)
    health = np.concatenate([_health(trajectory, 'people', frame) for frame in range(0, 61, 3)])
    assert {1.0, 2.0, 3.0, 4.0} <= set(np.unique(health).tolist())
    locked = (
        AgentBasedSIRSystem(20, 20, people=200, lockdown_length=50, lockdown_d=0.0, random_state=0).prepare().system_
    )
    still = _run_until(locked, 5, random_state=0)
    assert np.array_equal(still.agents('people', 5).locations, still.agents('people', 0).locations)


def test_random_walkers_die_after_infection():
    """Test that sick walkers infect others and fall dead, on a grid and on a network."""
    grid = _run_until(RandomWalkerEpidemicsSystem(32, 32, walkers=400, random_state=0), 100, random_state=0)
    assert np.sum(_health(grid, 'walkers', 100) == 2) > 1
    world = nx.watts_strogatz_graph(30, 4, 0.1, seed=0)
    system = RandomWalkerEpidemicsSystem(walkers=300, graph=world, infection=0.5, random_state=0).prepare().system_
    network = _run_until(system, 60, random_state=0)
    assert np.sum(_health(network, 'walkers', 60) > 0) > 10


def test_network_sir_spreads_between_cities():
    """Test that travellers carry the infection to several cities, where the infected recover."""
    trajectory = _run_until(NetworkSIRSystem(tau=0.1, m=0.1, random_state=0), 150, random_state=0)
    final = trajectory.agents('travellers', 150)
    recovered = final.values['health'] == 3
    assert recovered.sum() > 50
    assert len({final.locations[index] for index in np.flatnonzero(recovered)}) > 5
