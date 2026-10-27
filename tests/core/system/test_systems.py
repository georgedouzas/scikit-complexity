"""Test systems defined by their components, interactions, space, and time."""

from typing import ClassVar

import networkx as nx
import numpy as np
import pytest
from skcomplex.core.components import Element, GroupElement

from skcomplex.core.base import (
    BaseInteraction,
    NumericDomain,
    SpaceError,
    TensorDomain,
    ValidationError,
    parameter,
    role,
    state,
)
from skcomplex.core.elements import LatticeElement, ParticleElement, WordElement
from skcomplex.core.interactions import PairPotentialInteraction
from skcomplex.core.laws import GeneratorLaw, ProductionLaw, RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.spaces import ContinuousSpace, ContinuousTime, DiscreteTime, LatticeSpace, NetworkSpace, StaticTime
from skcomplex.core.system import System
from skcomplex.engine.compile import HilbertDomain, QuantumView, compile_system, is_quantum, validate
from skcomplex.engine.compile.passes import structure_of
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.biology.ecology import PopulationElement


class StockElement(Element):
    """Element with one nonnegative state."""

    n: float = state(1.0, domain=NumericDomain.NONNEGATIVE)


class GrowthInteraction(BaseInteraction):
    """Exponential growth of the state of its target."""

    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        """Return the rate of growth."""
        return [RateLaw(s.target.n, s.r * s.target.n)]


class CounterElement(Element):
    """Element whose state grows by one every step."""

    k: float = state(0.0)

    def laws(self, s):
        """Return the update of the count."""
        return [UpdateLaw(s.k, s.k + 1)]


def test_four_part_system_runs():
    """Test that a system of components, interactions, no space, and continuous time runs and lists its parameters."""
    system = System(
        'ecology',
        elements=[StockElement('rabbits'), StockElement('foxes')],
        interactions=[GrowthInteraction('growth', target='rabbits')],
        space=None,
        time=ContinuousTime(),
    )
    trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
    assert trajectory.values('rabbits.n')[-1] > 1.0
    assert Simulation(system).run(until=0).problem_.parameters.names == ['growth.r']


def test_systems_nest_as_components():
    """Test that nested systems are components and an interaction couples their elements by path."""
    system = System(
        systems=[System('meadow', elements=[StockElement('rabbits', n=2.0)])],
        interactions=[GrowthInteraction('growth', target='meadow/rabbits')],
    )
    assert system.systems[0].label == 'meadow'
    assert system.elements == ()
    trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
    assert trajectory.values('meadow/rabbits.n')[-1] > 2.0


def test_duplicate_labels_are_reported():
    """Test that two direct children with one label give one issue at the system that names the label."""
    report = validate(System('farm', elements=[StockElement('a'), StockElement('a')]))
    (issue,) = report.errors
    assert issue.path == 'farm'
    assert "'a'" in issue.message


def test_rate_under_discrete_time_is_reported():
    """Test that a rate law in a system of discrete time is reported with the time it needs."""
    system = System(
        elements=[StockElement('a')],
        interactions=[GrowthInteraction('g', target='a')],
        time=DiscreteTime(),
    )
    (issue,) = validate(system).errors
    assert issue.code == 'law-time'
    assert 'continuous' in issue.message
    assert 'update' in issue.fix


def test_nested_system_inherits_time():
    """Test that a nested system without a time is checked against the time of its parent."""
    nested = System('inner', elements=[CounterElement('clock')])
    assert validate(System(elements=[nested], time=DiscreteTime())).ok
    (issue,) = validate(System(elements=[nested], time=ContinuousTime())).errors
    assert issue.path == 'inner/clock'


def test_unknown_schedule_is_rejected():
    """Test that a schedule other than the known ones or phases is rejected."""
    with pytest.raises(ValidationError, match='schedule'):
        DiscreteTime(schedule='parallel').prepare()


def test_static_time_rejects_dynamic_laws():
    """Test that a static system holding a rate law is reported."""
    system = System(
        elements=[StockElement('a')],
        interactions=[GrowthInteraction('g', target='a')],
        time=StaticTime(),
    )
    assert [issue.code for issue in validate(system).errors] == ['law-time']


def test_periodic_space_wraps_positions_without_box_component():
    """Test that a particle crossing the side of a periodic space re-enters on the opposite side."""
    system = System(
        elements=[ParticleElement('p', x=[0.9, 0.5], v=[1.0, 0.0])],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
    )
    trajectory = Simulation(system, dt=0.2).run(until=0.2).trajectory_
    assert trajectory.values('p.x')[-1].tolist() == pytest.approx([0.1, 0.5])
    assert system.interactions == ()


def test_continuous_space_checks_its_box():
    """Test that an unbounded space rejects walls and a box needs one positive side per dimension."""
    with pytest.raises(ValidationError, match='unbounded'):
        ContinuousSpace(n_dim=2, boundary='periodic').check()
    with pytest.raises(ValidationError, match='positive side'):
        ContinuousSpace(extent=(1.0, -1.0)).check()
    with pytest.raises(ValidationError, match='boundary'):
        ContinuousSpace(extent=(1.0,), boundary='sticky').check()


def test_agents_and_fields_share_one_lattice_kind():
    """Test that agents and fields both live on lattice spaces, which declare agent neighbourhoods and capacities."""
    space = LatticeSpace((4, 3), boundary='periodic', neighborhood='von_neumann', capacity=1).prepare()
    assert (space.shape_, space.periodic_, space.capacity) == ((4, 3), True, 1)
    assert sorted(space.run(0).tolist()) == [1, 2, 3, 9]
    assert str(space.space_kind) == 'lattice'


def test_network_hyperedges_hold_nodes_of_the_graph():
    """Test that a hyperedge needs at least two nodes of the graph."""
    graph = nx.complete_graph(3)
    assert NetworkSpace(graph, hyperedges=[(0, 1, 2)]).hyperedges == [(0, 1, 2)]
    with pytest.raises(ValidationError, match='hyperedge'):
        NetworkSpace(graph, hyperedges=[(0, 7)]).prepare()


class CellElement(Element):
    """Element whose members divide every step."""

    size: float = state(1.0)

    def laws(self, s):
        """Return the rule of division."""
        return [RuleLaw(self.divide)]

    def divide(self, agent, model):
        """Create one offspring."""
        agent.create()


def test_variable_group_of_elements_grows():
    """Test that a variable group of 10 dividing members has 20, 40, and 80 after one, two, and three steps."""
    system = System(elements=[GroupElement('cells', element=CellElement, size=10, variable=True)])
    trajectory = Simulation(system).run(until=3).trajectory_
    assert trajectory.values('cells.count').tolist() == [10.0, 20.0, 40.0, 80.0]


def test_group_on_edges_has_one_member_per_edge():
    """Test that a group on the edges of a network with 30 edges holds 30 members, sized from the inherited space."""
    weights = GroupElement('weights', element=StockElement, on='edges')
    problem = compile_system(System(elements=[weights], space=NetworkSpace(nx.cycle_graph(30))))
    assert problem.states.entry('weights.n').shape == (30,)


def test_group_on_edges_outside_a_network_is_reported():
    """Test that a group on edges in a lattice space is reported with the space it needs."""
    (issue,) = validate(
        System(elements=[GroupElement('weights', element=StockElement, on='edges')], space=LatticeSpace(4)),
    ).errors
    assert 'need a network space' in issue.message


def test_unknown_part_is_rejected():
    """Test that checking a group on a part other than sites, edges, and hyperedges raises an error."""
    group = GroupElement('cells', element=StockElement, on='faces', space=LatticeSpace(4))
    with pytest.raises(SpaceError, match='faces'):
        group.check()


PAULI = HilbertDomain(
    2,
    {'x': (np.array([[0.0, 1.0], [1.0, 0.0]]), True), 'z': (np.diag([1.0, -1.0]), True)},
    basis={'up': 0, 'down': 1},
)


class TwoLevelElement(Element):
    """Element whose only state lies in a two-dimensional Hilbert space, driven by the operator `x`."""

    psi: object = state('up', domain=PAULI)

    def laws(self, s):
        """Return the generator of the rotation about `x`."""
        return [GeneratorLaw(s.x / 2)]


def test_element_with_hilbert_state_runs_unitary_dynamics():
    """Test that a plain element with a state in a Hilbert space and a generator law runs unitary dynamics."""
    simulation = Simulation(System(elements=[TwoLevelElement('q')]), solver='unitary', dt=np.pi).run(until=np.pi)
    assert str(simulation.problem_.form.name) == 'QUANTUM'
    assert QuantumView(TwoLevelElement('q', psi='down')).initial_vector().tolist() == [0j, 1 + 0j]
    assert is_quantum(TwoLevelElement)
    assert not is_quantum(StockElement)


def test_sequence_state_is_rewritten_in_parallel():
    """Test that an element with a sequence state is rewritten by its productions at every step."""
    rules = [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]

    class AlgaeElement(WordElement):
        """Algae element."""

        def laws(self, s):
            """Return the laws of the algae element."""
            return [RewriteLaw(rules)]

    system = System(elements=[AlgaeElement('algae', axiom='A')])
    trajectory = Simulation(system).run(until=4).trajectory_
    assert trajectory.values('algae.size').tolist() == [1.0, 2.0, 3.0, 5.0, 8.0]
    assert structure_of(WordElement) == 'sequence'


def test_structured_domains_name_their_kind():
    """Test that tensor and structure domains report their kinds, and an unknown structure is rejected."""
    assert TensorDomain(2, 'traceless').value == 'traceless tensor of rank 2'
    with pytest.raises(ValidationError, match='structure'):
        StructureDomain('tree').check()
    with pytest.raises(ValidationError, match='structure'):
        state(domain=StructureDomain('tree'))


def test_state_on_an_interaction_is_rejected():
    """Test that creating an interaction whose type declares a state raises an error that suggests a group on edges."""
    weighted = type('WeightedInteraction', (BaseInteraction,), {'weight': state(1.0)})
    with pytest.raises(ValidationError, match='on="edges"'):
        weighted('w').check()


def test_unknown_binding_is_rejected():
    """Test that creating an interaction with an unknown binding raises an error that lists the bindings."""

    class EverywhereInteraction(BaseInteraction):
        """Interaction with an unknown binding."""

        over: ClassVar[str] = 'everything'

    with pytest.raises(ValidationError, match='members, pairs, neighbors, hyperedges'):
        EverywhereInteraction('everywhere').check()


def test_pairs_binding_needs_its_roles_and_cutoff():
    """Test that a binding over pairs without a cutoff is reported with the declarations it lacks."""

    class ContactInteraction(BaseInteraction):
        """Pair interaction without a cutoff."""

        over: ClassVar[str] = 'pairs'
        first: str = role()
        second: str = role()

    system = System(
        elements=[GroupElement('herd', element=StockElement, size=3)],
        interactions=[ContactInteraction('c', first='herd', second='herd')],
    )
    issue = next(issue for issue in validate(system).errors if issue.code == 'binding')
    assert 'cutoff' in issue.fix


def test_hyperedges_binding_needs_a_network_with_hyperedges():
    """Test that a binding over hyperedges of a group on a lattice is reported."""

    class TriadicInteraction(BaseInteraction):
        """Interaction over the hyperedges of its target."""

        over: ClassVar[str] = 'hyperedges'
        target: str = role()

    system = System(
        elements=[LatticeElement('nodes', element=StockElement, space=LatticeSpace(4))],
        interactions=[TriadicInteraction('triads', target='nodes')],
    )
    (issue,) = [issue for issue in validate(system).errors if issue.code == 'binding']
    assert 'network space with hyperedges' in issue.message


class SoftSphereInteraction(PairPotentialInteraction):
    """Soft repulsion `k * (cutoff - r) ** 2 / 2`."""

    k: float = parameter(50.0, domain=NumericDomain.POSITIVE)

    def pair_energy(self, s, r):
        """Return the soft repulsion energy."""
        return s.k * (s.cutoff - r) ** 2 / 2


def test_system_prints_its_parameters():
    """Test that a system prints the parameters it was given, through scikit-learn."""
    system = System('meadow', elements=[PopulationElement('rabbits')])
    assert repr(system) == "System(elements=[PopulationElement(label='rabbits', n=1.0)], label='meadow')"


def test_periodic_box_wraps_member_with_same_velocity():
    """Test that a member leaving one side reappears at the opposite side with the same velocity."""
    system = System(
        elements=[
            GroupElement('gas', element=ParticleElement, size=1, values={'x': [[0.95, 0.5]], 'v': [[1.0, 0.0]]}),
        ],
        space=ContinuousSpace(extent=(1.0, 1.0), boundary='periodic'),
    )
    trajectory = Simulation(system, dt=0.1).run(until=0.2).trajectory_
    assert np.allclose(trajectory.values('gas.x')[-1], [[0.15, 0.5]])
    assert np.allclose(trajectory.values('gas.v')[-1], [[1.0, 0.0]])


@pytest.mark.slow
def test_pair_interaction_conserves_momentum():
    """Test that soft-sphere collisions inside an isolated group keep the total momentum."""
    rng = np.random.default_rng(0)
    system = System(
        elements=[
            GroupElement(
                'gas',
                element=ParticleElement,
                size=200,
                values={'x': rng.uniform(0, 3, (200, 2)), 'v': rng.normal(0, 1, (200, 2))},
            ),
        ],
        interactions=[SoftSphereInteraction('soft', first='gas', second='gas', cutoff=0.2)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = Simulation(system, solver=ScipyODESolver(rtol=1e-9, atol=1e-12), dt=0.05).run(until=2).trajectory_
    momentum = trajectory.values('gas.v').sum(axis=1)
    assert np.allclose(momentum, momentum[0], atol=1e-8)
