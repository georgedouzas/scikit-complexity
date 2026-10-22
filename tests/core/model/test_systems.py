"""Test systems defined by their components, interactions, space, and time."""

from typing import ClassVar

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.base import (
    Domain,
    HilbertSpace,
    SpaceError,
    Structure,
    Tensor,
    ValidationError,
    parameter,
    role,
    state,
)
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.model import (
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    Element,
    Generator,
    Group,
    Interaction,
    LatticeSpace,
    NetworkSpace,
    Particle,
    Phase,
    Phases,
    Production,
    QuantumView,
    Rate,
    Rewrite,
    Rule,
    StaticTime,
    System,
    Update,
    Word,
    is_quantum,
    structure_of,
)
from skcomplex.core.runtime import Simulation


class Stock(Element):
    """Element with one nonnegative state."""

    n: float = state(1.0, domain=Domain.NONNEGATIVE)


class Growth(Interaction):
    """Exponential growth of the state of its target."""

    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        """Return the rate of growth."""
        return [Rate(s.target.n, s.r * s.target.n)]


class Counter(Element):
    """Element whose state grows by one every step."""

    k: float = state(0.0)

    def laws(self, s):
        """Return the update of the count."""
        return [Update(s.k, s.k + 1)]


def test_four_part_system_runs():
    """Test that a system of components, interactions, no space, and continuous time runs and lists its parameters."""
    system = System(
        'ecology',
        components=[Stock('rabbits'), Stock('foxes')],
        interactions=[Growth('growth', target='rabbits')],
        space=None,
        time=ContinuousTime(),
    )
    trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
    assert trajectory['rabbits.n'][-1] > 1.0
    assert Simulation(system).run(until=0).problem_.parameters.names == ['growth.r']


@pytest.mark.parametrize(
    ('argument', 'home'),
    [
        ('noise', 'Noise(..., convention=...)'),
        ('history', 'state(history=...)'),
        ('phases', 'DiscreteTime(schedule=Phases([...]))'),
        ('observables', 'Simulation(observables=...)'),
        ('elements', '`components`'),
    ],
)
def test_execution_arguments_are_rejected(argument, home):
    """Test that an argument of the former signature raises a type error that names its new home."""
    with pytest.raises(TypeError, match=argument) as error:
        System(**{argument: []})
    assert home in str(error.value)


def test_systems_nest_as_components():
    """Test that nested systems are components and an interaction couples their elements by path."""
    system = System(
        components=[System('meadow', components=[Stock('rabbits', n=2.0)])],
        interactions=[Growth('growth', target='meadow/rabbits')],
    )
    assert system.systems[0].label == 'meadow'
    assert system.elements == ()
    trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
    assert trajectory['meadow/rabbits.n'][-1] > 2.0


def test_duplicate_labels_are_reported():
    """Test that two direct children with one label give one issue at the system that names the label."""
    report = validate(System('farm', components=[Stock('a'), Stock('a')]))
    (issue,) = report.errors
    assert issue.path == 'farm'
    assert "'a'" in issue.message


def test_rate_under_discrete_time_is_reported():
    """Test that a rate law in a system of discrete time is reported with the time it needs."""
    system = System(components=[Stock('a')], interactions=[Growth('g', target='a')], time=DiscreteTime())
    (issue,) = validate(system).errors
    assert issue.code == 'law-time'
    assert 'continuous' in issue.message
    assert 'update' in issue.fix


def test_nested_system_inherits_time():
    """Test that a nested system without a time is checked against the time of its parent."""
    nested = System('inner', components=[Counter('clock')])
    assert validate(System(components=[nested], time=DiscreteTime())).ok
    (issue,) = validate(System(components=[nested], time=ContinuousTime())).errors
    assert issue.path == 'inner/clock'


def test_ordered_phases_are_a_schedule():
    """Test that a discrete time with phases lists them in order, and rejects repeated names."""
    time = DiscreteTime(schedule=Phases([Phase('produce'), Phase('trade'), Phase('consume')]))
    assert [phase.name for phase in time.phases] == ['produce', 'trade', 'consume']
    assert DiscreteTime().phases == ()
    with pytest.raises(ValidationError, match='repeat a name'):
        Phases([Phase('trade'), Phase('trade')])


def test_unknown_schedule_is_rejected():
    """Test that a schedule other than the known ones or phases is rejected."""
    with pytest.raises(ValidationError, match='schedule'):
        DiscreteTime(schedule='parallel')


def test_static_time_rejects_dynamic_laws():
    """Test that a static system holding a rate law is reported."""
    system = System(components=[Stock('a')], interactions=[Growth('g', target='a')], time=StaticTime())
    assert [issue.code for issue in validate(system).errors] == ['law-time']


def test_periodic_space_wraps_positions_without_box_component():
    """Test that a particle crossing the side of a periodic space re-enters on the opposite side."""
    system = System(
        components=[Particle('p', x=[0.9, 0.5], v=[1.0, 0.0])],
        space=ContinuousSpace(1.0, 1.0, boundary='periodic'),
    )
    trajectory = Simulation(system, dt=0.2).run(until=0.2).trajectory_
    assert trajectory['p.x'][-1].tolist() == pytest.approx([0.1, 0.5])
    assert system.interactions == ()


def test_continuous_space_checks_its_box():
    """Test that an unbounded space rejects walls and a box needs one positive side per dimension."""
    with pytest.raises(ValidationError, match='unbounded'):
        ContinuousSpace(n_dim=2, boundary='periodic')
    with pytest.raises(ValidationError, match='positive side'):
        ContinuousSpace(1.0, -1.0)
    with pytest.raises(ValidationError, match='boundary'):
        ContinuousSpace(1.0, boundary='sticky')


def test_agents_and_fields_share_one_lattice_kind():
    """Test that agents and fields both live on lattice spaces, which declare agent neighbourhoods and capacities."""
    space = LatticeSpace((4, 3), boundary='periodic', neighborhood='von_neumann', capacity=1)
    assert (space.width, space.height, space.torus, space.capacity) == (4, 3, True, 1)
    assert str(space.space_kind) == 'lattice'


def test_network_hyperedges_hold_nodes_of_the_graph():
    """Test that a hyperedge needs at least two nodes of the graph."""
    graph = nx.complete_graph(3)
    assert NetworkSpace(graph, hyperedges=[(0, 1, 2)]).hyperedges == [(0, 1, 2)]
    with pytest.raises(ValidationError, match='hyperedge'):
        NetworkSpace(graph, hyperedges=[(0, 7)])


class Cell(Element):
    """Element whose members divide every step."""

    size: float = state(1.0)

    def laws(self, s):
        """Return the rule of division."""
        return [Rule(self.divide)]

    def divide(self, agent, model):
        """Create one offspring."""
        agent.create()


def test_variable_group_of_elements_grows():
    """Test that a variable group of 10 dividing members has 20, 40, and 80 after one, two, and three steps."""
    system = System(components=[Group('cells', element=Cell, size=10, variable=True)])
    trajectory = Simulation(system).run(until=3).trajectory_
    assert trajectory['cells.count'].tolist() == [10.0, 20.0, 40.0, 80.0]


def test_group_on_edges_has_one_member_per_edge():
    """Test that a group on the edges of a network with 30 edges holds 30 members, sized from the inherited space."""
    weights = Group('weights', element=Stock, on='edges')
    problem = compile_system(System(components=[weights], space=NetworkSpace(nx.cycle_graph(30))))
    assert problem.states.entry('weights.n').shape == (30,)


def test_group_on_edges_outside_a_network_is_reported():
    """Test that a group on edges in a lattice space is reported with the space it needs."""
    (issue,) = validate(System(components=[Group('weights', element=Stock, on='edges')], space=LatticeSpace(4))).errors
    assert 'need a network space' in issue.message


def test_unknown_part_is_rejected():
    """Test that a group on a part other than sites, edges, and hyperedges is rejected."""
    with pytest.raises(SpaceError, match='faces'):
        Group('cells', element=Stock, on='faces', space=LatticeSpace(4))


PAULI = HilbertSpace(
    2,
    {'x': (np.array([[0.0, 1.0], [1.0, 0.0]]), True), 'z': (np.diag([1.0, -1.0]), True)},
    basis={'up': 0, 'down': 1},
)


class TwoLevel(Element):
    """Element whose only state lies in a two-dimensional Hilbert space, driven by the operator `x`."""

    psi: object = state('up', domain=PAULI)

    def laws(self, s):
        """Return the generator of the rotation about `x`."""
        return [Generator(s.x / 2)]


def test_element_with_hilbert_state_runs_unitary_dynamics():
    """Test that a plain element with a state in a Hilbert space and a generator law runs unitary dynamics."""
    simulation = Simulation(System(components=[TwoLevel('q')]), solver='unitary', dt=np.pi).run(until=np.pi)
    assert str(simulation.problem_.form.name) == 'QUANTUM'
    assert QuantumView(TwoLevel('q', psi='down')).initial_vector().tolist() == [0j, 1 + 0j]
    assert is_quantum(TwoLevel)
    assert not is_quantum(Stock)


def test_sequence_state_is_rewritten_in_parallel():
    """Test that an element with a sequence state is rewritten by its productions at every step."""
    rules = [Production('A', 'AB'), Production('B', 'A')]
    system = System(components=[Word('algae', axiom='A')], interactions=[Rewrite('rules', target='algae', rules=rules)])
    trajectory = Simulation(system).run(until=4).trajectory_
    assert trajectory['algae.size'].tolist() == [1.0, 2.0, 3.0, 5.0, 8.0]
    assert structure_of(Word) == 'sequence'


def test_structured_domains_name_their_kind():
    """Test that tensor and structure domains report their kinds, and an unknown structure is rejected."""
    assert Tensor(2, 'traceless').value == 'traceless tensor of rank 2'
    with pytest.raises(ValueError, match='structure'):
        Structure('tree')


def test_state_on_an_interaction_is_rejected():
    """Test that an interaction type declaring a state raises an error that suggests a group on edges."""
    with pytest.raises(ValidationError, match='on="edges"'):
        type('Weighted', (Interaction,), {'__annotations__': {'weight': float}, 'weight': state(1.0)})


def test_unknown_binding_is_rejected():
    """Test that an interaction type with an unknown binding raises an error that lists the bindings."""
    with pytest.raises(ValidationError, match='members, pairs, neighbours, hyperedges'):

        class Everywhere(Interaction):
            """Interaction with an unknown binding."""

            over: ClassVar[str] = 'everything'


def test_pairs_binding_needs_its_roles_and_cutoff():
    """Test that a binding over pairs without a cutoff is reported with the declarations it lacks."""

    class Contact(Interaction):
        """Pair interaction without a cutoff."""

        over: ClassVar[str] = 'pairs'
        first: str = role()
        second: str = role()

    system = System(components=[Stock.group('herd', 3)], interactions=[Contact('c', first='herd', second='herd')])
    issue = next(issue for issue in validate(system).errors if issue.code == 'binding')
    assert 'cutoff' in issue.fix


def test_hyperedges_binding_needs_a_network_with_hyperedges():
    """Test that a binding over hyperedges of a group on a lattice is reported."""

    class Triadic(Interaction):
        """Interaction over the hyperedges of its target."""

        over: ClassVar[str] = 'hyperedges'
        target: str = role()

    system = System(
        components=[Stock.lattice('nodes', LatticeSpace(4))],
        interactions=[Triadic('triads', target='nodes')],
    )
    (issue,) = [issue for issue in validate(system).errors if issue.code == 'binding']
    assert 'network space with hyperedges' in issue.message
