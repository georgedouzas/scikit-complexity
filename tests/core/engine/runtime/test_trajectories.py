"""Test reading the recorded values of a trajectory by name."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.experiments._replicate import _replicate_prepared
from skcomplex.core.systems import System
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.computer_science.automata import GameOfLifeSystem
from skcomplex.systems.physics.active_matter import VicsekSystem
from skcomplex.systems.physics.statistical_mechanics import IsingSystem


def test_trajectory_reads_values_by_name():
    """Test that a trajectory returns the values of a state by name and is not indexed."""
    trajectory = LotkaVolterraSystem().prepare().system_.prepare().simulate(until=1.0, dt=0.1)
    values = trajectory.get_values('prey.x')
    assert values.shape[0] == trajectory.times.size
    with pytest.raises(TypeError):
        trajectory['prey.x']


def test_member_keeps_the_parameters_of_interactions():
    """Test that a member of a replicated system reads its own states and the shared parameters of interactions."""
    lotka = LotkaVolterraSystem().prepare().system_
    plain = System(
        elements=[Element(element.label, values={'x': element.x}) for element in lotka.elements],
        interactions=lotka.interactions,
    )
    ensemble = _replicate_prepared(plain, 2, {'prey.x': [20.0, 40.0]})
    trajectory = ensemble.prepare().simulate(until=1.0, dt=0.1)
    member = trajectory.get_member(1, LotkaVolterraSystem().prepare().system_._problem)
    assert float(member.get_values('prey.x')[0]) == 40.0
    assert float(member.get_values('growth.r')[0]) == 1.1


def _falling_trajectory() -> Trajectory:
    """Return the trajectory of a body falling for one second."""
    gravity = Interaction('gravity', roles={'target': 'body'}, laws=lambda s: [RateLaw(s.target.h.diff(s.t, 2), -9.81)])
    system = System(elements=[Element('body', values={'h': 1.0})], interactions=[gravity]).prepare()
    return system.simulate(until=1.0, dt=0.5, initial_state={"body.h'": 2.0})


def test_names_show_derivatives_with_primes():
    """The names of a trajectory show derivatives with primes."""
    assert _falling_trajectory().names[:2] == ['body.h', "body.h'"]


def test_get_values_reads_a_derivative_by_its_prime():
    """A derivative is read by its name with a prime, and the start takes the value given by that name."""
    trajectory = _falling_trajectory()
    assert trajectory.get_values("body.h'")[0] == pytest.approx(2.0)
    assert trajectory.get_values("body.h'")[-1] == pytest.approx(2.0 - 9.81)


def test_get_values_lists_names_with_primes_when_one_is_unknown():
    """An unknown name raises an error that lists the names with primes."""
    with pytest.raises(KeyError, match=r"body\.h'"):
        _falling_trajectory().get_values('body.v')


def _age(agent, system):
    """Grow one step older."""
    agent.age += 1.0


def _age_and_breed(agent, system):
    """Grow one step older and create one offspring of age zero."""
    agent.age += 1.0
    agent.create(age=0.0)


def _herd(rule) -> Trajectory:
    """Return the trajectory of two agents that follow a rule for two steps."""
    rules = Interaction('rules', roles={'target': 'herd'}, laws=lambda s: [UpdateLaw(s.target, rule)])
    system = System(elements=[Element('herd', values={'age': 0.0}, size=2)], interactions=[rules])
    return system.prepare().simulate(until=2)


def test_get_values_reads_a_state_of_a_growing_group_per_frame():
    """A state of agents whose group grows is one array per frame, as the agents of that frame hold it."""
    trajectory = _herd(_age_and_breed)
    values = trajectory.get_values('herd.age')
    assert isinstance(values, list)
    assert [len(frame) for frame in values] == [2, 4, 8]
    for frame, read in enumerate(values):
        np.testing.assert_array_equal(read, trajectory.get_agents('herd', frame).values['age'])


def test_get_values_reads_a_state_of_a_fixed_group_as_rows():
    """A state of agents whose group keeps its members is one row per frame."""
    trajectory = _herd(_age)
    assert trajectory.get_values('herd.age').tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    assert 'herd.age' in trajectory.names


def test_get_network_reads_the_von_neumann_pairs_of_an_ising_lattice():
    """The network of an Ising lattice joins every site to its four von Neumann neighbours."""
    network = IsingSystem(5).prepare().simulate(until=1, random_state=0).get_network(0)
    graph = network.get_graph()
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (25, 50)
    assert {degree for _, degree in graph.degree()} == {4}
    assert set(graph.neighbors(0)) == {1, 4, 5, 20}


def test_get_network_reads_the_moore_pairs_of_life():
    """The network of Life joins every cell to its eight Moore neighbours."""
    network = GameOfLifeSystem((6, 6)).prepare().simulate(until=1, random_state=0).get_network(0)
    graph = network.get_graph()
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (36, 144)
    assert graph.has_edge(0, 7)


def test_get_network_reads_the_agents_within_the_radius_of_a_flock():
    """The network of a Vicsek flock joins agents within the interaction radius, and changes with the frame."""
    trajectory = VicsekSystem(100, size=5.0).prepare().simulate(until=3, random_state=0)
    positions = trajectory.get_values('flock.x')
    edges = []
    for frame in (0, 3):
        network = trajectory.get_network(frame)
        difference = positions[frame][network.sources] - positions[frame][network.targets]
        difference -= 5.0 * np.rint(difference / 5.0)
        assert np.all(np.hypot(*difference.T) <= 1.0)
        assert network.nodes.size == 100
        edges.append(set(zip(network.sources.tolist(), network.targets.tolist(), strict=True)))
    assert edges[0] != edges[1]


def test_get_network_names_both_cases_without_a_network():
    """A system with neither a changing network nor a fixed neighbourhood raises an error naming both."""
    trajectory = LotkaVolterraSystem().prepare().simulate(until=1.0)
    with pytest.raises(KeyError, match=r'variable=True.*within a radius'):
        trajectory.get_network(0)
