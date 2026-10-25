"""Test the `_components` module of `skcomplex.core.base`."""

from typing import ClassVar

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin
from sklearn.base import clone

from skcomplex.core.base import (
    ArgumentError,
    BaseComponent,
    BaseDomain,
    BaseInteraction,
    BaseLaw,
    BaseSpace,
    BaseTime,
    ConservationError,
    RoleError,
    ValidationError,
    parameter,
    role,
    state,
    symbol,
)
from skcomplex.core.components import ElementComponent, GroupComponent
from skcomplex.core.laws import (
    AuxiliaryQuantity,
    InvariantQuantity,
    JumpLaw,
    RateLaw,
    Symbols,
    UpdateLaw,
    neighbors,
)
from skcomplex.core.problems import BaseBackend, BaseProblem, BaseSolver
from skcomplex.core.space import DiscreteTime, LatticeSpace, Layer, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import BaseSystemCompiler, compile_system, validate
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.models.biology.ecology import PopulationComponent, PredationInteraction


class TallyComponent(ElementComponent):
    """Element with a setting next to its declared state."""

    value: float = state(0.0)

    def __init__(self, label, *, source='', **values):
        """Store the label, the source, and the values."""
        super().__init__(label, **values)
        self.source = source


class TanksComponent(ElementComponent):
    """Two tanks that drain into each other at a rate an auxiliary names, under the invariant of their total."""

    a: float = state(3.0)
    b: float = state(1.0)
    k: float = parameter(0.5)

    def laws(self, s):
        """Return the rate of the first tank and of the second through the auxiliary."""
        outflow, _ = self.quantities(s)
        return [RateLaw(s.a, -outflow.symbol), RateLaw(s.b, outflow.symbol)]

    def quantities(self, s):
        """Return the outflow of the first tank and the invariant of the total."""
        return [AuxiliaryQuantity('outflow', s.k * s.a), InvariantQuantity('total', s.a + s.b)]


class PendulumComponent(ElementComponent):
    """Pendulum whose angle has order 2.

    Args:
        label:
            The label of the pendulum.

        theta:
            The initial angle.
    """

    theta: float = state(0.0, order=2)


def test_component_is_the_base_of_elements_and_groups_only():
    """Test that elements and groups derive from the component base, and interactions and systems do not."""
    for kind in (ElementComponent, GroupComponent):
        assert issubclass(kind, BaseComponent)
    for kind in (BaseInteraction, System):
        assert not issubclass(kind, BaseComponent)
    system = System(components=[System('meadow', components=[PopulationComponent('rabbits')])])
    assert not isinstance(system, BaseComponent)


def test_component_stores_companions_without_class_attributes():
    """Test that a state of order 2 takes its derivative under a derived name that is not a class attribute."""
    bob = PendulumComponent('bob', theta=0.1, theta_d1=0.5)
    assert not hasattr(PendulumComponent, 'theta_d1')
    assert (bob.theta, bob.theta_d1) == (0.1, 0.5)
    assert {'theta', 'theta_d1'} <= set(bob.get_params())
    assert list(bob.prepare().declarations_) == ['theta', 'theta_d1']


def test_component_rejects_unknown_and_missing_arguments():
    """Test that an undeclared value and a missing role are stored, and checking raises a named type error."""
    rabbits = PopulationComponent('rabbits', foo=1)
    assert rabbits.foo == 1
    with pytest.raises(ArgumentError, match="'foo'"):
        rabbits.check()
    growth = BaseInteraction.define('GrowthInteraction', roles=['target'])('growth')
    with pytest.raises(TypeError, match="'target'"):
        growth.check()


def test_validation_checks_every_component_with_its_path():
    """Test that validating a system reports a component with an undeclared value by its path."""
    system = System(components=[System('meadow', components=[PopulationComponent('rabbits', foo=1)])])
    report = validate(system)
    assert not report.ok
    assert report.errors[0].message.startswith("meadow/rabbits: PopulationComponent got 'foo'")


def test_interaction_rejects_a_layer_when_checked():
    """Test that an interaction not over neighbours takes a layer but rejects it when checked."""
    interaction = PredationInteraction('hunt', predator='foxes', prey='rabbits', layer='work')
    assert interaction.layer == 'work'
    with pytest.raises(ValidationError, match='only interactions over neighbours'):
        interaction.check()


def test_component_runs_its_laws_and_prepares_its_declarations():
    """Test that `run` returns the laws of the `laws` hook and `prepare` sets the declarations."""
    growth = BaseInteraction.define(
        'GrowthInteraction',
        roles=['target'],
        parameters={'r': 1.0},
        laws=lambda s: [RateLaw(s.target.n, s.r * s.target.n)],
    )('growth', target='rabbits').prepare()
    assert list(growth.declarations_) == ['target', 'r']
    n, r = symbol('n'), symbol('r')
    (law,) = growth.run(Symbols(target=Symbols(n=n), r=r))
    assert isinstance(law, RateLaw)
    assert law.value == r * n


def test_quantities_are_none_by_default():
    """Test that a component and an interaction return no quantities unless their type declares some."""
    s = Symbols(n=symbol('n'))
    assert PopulationComponent('rabbits').quantities(s) == []
    assert BaseInteraction('coupling').quantities(s) == []


def test_quantities_substitute_into_laws_and_hold_every_frame():
    """Test that the auxiliary of a component reaches its rates and its invariant is checked at every frame."""
    system = System(components=[TanksComponent('tanks')])
    problem = compile_system(system)
    assert str(problem.equations['tanks.a']) == '-a_tanks*k_tanks'
    trajectory = Simulation(system, dt=0.5).run(until=2).trajectory_
    np.testing.assert_allclose(trajectory.values('tanks.total'), 4.0)
    np.testing.assert_allclose(trajectory.values('tanks.outflow'), 0.5 * trajectory.values('tanks.a'))


def test_violated_quantity_is_checked_at_every_frame():
    """Test that an invariant the laws break raises the conservation error at the first frame that breaks it."""
    LeakyTanksComponent = ElementComponent.define(
        'LeakyTanksComponent',
        states={'a': 1.0},
        laws=lambda s: [RateLaw(s.a, 1.0)],
        quantities=lambda s: [InvariantQuantity('total', s.a)],
    )
    with pytest.raises(ConservationError, match=r'invariant leaky\.total .* at frame 1'):
        Simulation(System(components=[LeakyTanksComponent('leaky')]), dt=0.5).run(until=2)


def test_component_parameters_list_settings_and_declared_names():
    """Test that the parameters are the label, the settings, and every declared name with its value or default."""
    tally = TallyComponent('size', source='farmers')
    assert tally.get_params() == {'label': 'size', 'source': 'farmers', 'value': 0.0}
    assert repr(tally) == "TallyComponent(label='size', source='farmers', value=0.0)"


def test_clone_keeps_declared_values():
    """Test that a clone of a group of 100 members has the same values and compiles to the same problem."""
    group = PopulationComponent.group('herds', 100, n=np.linspace(1.0, 2.0, 100))
    cloned = clone(group)
    assert (cloned.label, cloned.size) == ('herds', 100)
    assert np.array_equal(cloned.values['n'], group.values['n'])
    first, second = compile_system(System(components=[group])), compile_system(System(components=[cloned]))
    assert np.array_equal(first.initial_state, second.initial_state)
    assert first.equations == second.equations
    single = PendulumComponent('bob', theta=0.3)
    assert clone(single).get_params() == single.get_params()


def test_component_supports_estimator_contract():
    """Test that components follow the scikit-learn parameter contract."""
    interaction = PredationInteraction('hunt', predator='foxes', prey='rabbits', a=0.4)
    assert interaction.get_params()['a'] == 0.4
    cloned = clone(interaction)
    assert cloned is not interaction
    assert cloned.a == 0.4


BACKENDS = ('numpy', 'numba')


NeuronComponent = ElementComponent.define('NeuronComponent', states={'a': 0.0})


SynapseComponent = ElementComponent.define('SynapseComponent', states={'w': 0.1})


PersonComponent = ElementComponent.define('PersonComponent', states={'infected': 0.0, 'wage': 0.0})


OscillatorComponent = ElementComponent.define('OscillatorComponent', states={'theta': 0.0}, parameters={'omega': 0.0})


class HebbianInteraction(BaseInteraction):
    """Weights that grow with the product of the activities at both ends of an edge."""

    over: ClassVar[str] = 'neighbors'
    nodes: str = role()
    synapses: str = role()
    eta: float = parameter(1.0)

    def laws(self, s):
        """Return the growth of the weight of one edge."""
        return [RateLaw(s.edge.w, s.eta * s.source.a * s.target.a)]


class InfectionInteraction(BaseInteraction):
    """Contacts recorded on edges, with an update of an end that edges cannot set."""

    over: ClassVar[str] = 'neighbors'
    people: str = role()
    contacts: str = role()

    def laws(self, s):
        """Return the record of a contact and the infection of the source."""
        return [UpdateLaw(s.edge.contacted, 1.0), UpdateLaw(s.source.infected, s.target.infected)]


class ContagionInteraction(BaseInteraction):
    """Infection along the edges of a layer, without an edge group."""

    over: ClassVar[str] = 'neighbors'
    people: str = role()

    def laws(self, s):
        """Return the infection of both ends by the other."""
        return [
            RateLaw(s.source.infected, s.target.infected * (1 - s.source.infected)),
            RateLaw(s.target.infected, s.source.infected * (1 - s.target.infected)),
        ]


class WagesInteraction(BaseInteraction):
    """Wages that flow from the source of a work edge to its target."""

    over: ClassVar[str] = 'neighbors'
    people: str = role()
    jobs: str = role()

    def laws(self, s):
        """Return the transfer of wages along one work edge."""
        return [RateLaw(s.target.wage, s.edge.salary), RateLaw(s.source.wage, -s.edge.salary)]


class CouplingInteraction(BaseInteraction):
    """Kuramoto coupling along the weighted edges of a network."""

    over: ClassVar[str] = 'neighbors'
    oscillators: str = role()
    links: str = role()

    def laws(self, s):
        """Return the pull of both ends towards each other."""
        return [
            RateLaw(s.source.theta, s.edge.w * sin(s.target.theta - s.source.theta)),
            RateLaw(s.target.theta, s.edge.w * sin(s.source.theta - s.target.theta)),
        ]


def _hebbian(space, activities):
    """Return a network of neurons with fixed activities and Hebbian synapses on its edges."""
    return System(
        components=[
            NeuronComponent.lattice('neurons', space, a=activities),
            GroupComponent('synapses', element=SynapseComponent, space=space, on='edges'),
        ],
        interactions=[HebbianInteraction('hebb', nodes='neurons', synapses='synapses', eta=2.0)],
    )


def test_interaction_requires_roles():
    """Test that an interaction without its roles is created but does not pass its check."""
    with pytest.raises(TypeError, match='predator'):
        PredationInteraction('hunt', prey='rabbits').check()


@pytest.mark.parametrize('backend', BACKENDS)
def test_hebbian_weights_grow_between_coactive_nodes(backend):
    """Test that Hebbian weights end higher on edges between co-active nodes than on edges between inactive ones."""
    space = NetworkSpace(nx.cycle_graph(6))
    activities = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    simulation = Simulation(_hebbian(space, activities), backend=backend, solver=RK4Solver(dt=0.01), dt=0.1)
    weights = simulation.run(until=1.0).trajectory_.values('synapses.w')[-1]
    sources, targets = space.prepare().edges_[None]
    coactive = activities[sources] * activities[targets] > 0
    assert np.allclose(weights[coactive], 2.1, atol=1e-9)
    assert np.allclose(weights[~coactive], 0.1, atol=1e-12)
    assert weights[coactive].min() > weights[~coactive].max()


@pytest.mark.parametrize('backend', BACKENDS)
def test_edge_laws_read_the_ends_in_the_direction_of_directed_edges(backend):
    """Test that the source and target of a directed edge follow its direction, and edge states weigh the flow."""
    JobComponent = ElementComponent.define('JobComponent', states={'salary': 1.0})
    graph = nx.DiGraph([(0, 1), (2, 1), (1, 3)])
    space = NetworkSpace(graph)
    system = System(
        components=[
            PersonComponent.lattice('people', space),
            GroupComponent('jobs', element=JobComponent, space=space, on='edges', values={'salary': [1.0, 2.0, 4.0]}),
        ],
        interactions=[WagesInteraction('wages', people='people', jobs='jobs')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    salaries = dict(zip(graph.edges, [1.0, 2.0, 4.0], strict=True))
    expected = np.zeros(4)
    for (first, second), salary in salaries.items():
        expected[first] -= salary
        expected[second] += salary
    assert np.allclose(problem.states.read(drift, 'people.wage'), expected, atol=1e-12)


def test_two_layers_carry_infection_and_wages_apart():
    """Test that infection spreads only along contact edges and wages flow only along work edges."""
    contact = nx.path_graph(5)
    work = nx.Graph([(0, 4), (2, 3)])
    work.add_nodes_from(range(5))
    JobComponent = ElementComponent.define('JobComponent', states={'salary': 1.0})
    space = NetworkSpace(contact, layers=[Layer('work', work)])
    system = System(
        components=[
            PersonComponent.lattice('people', space, infected=[1.0, 0.0, 0.0, 0.0, 0.0]),
            GroupComponent('jobs', element=JobComponent, space=space, on='edges', layer='work'),
        ],
        interactions=[
            ContagionInteraction('contagion', people='people'),
            WagesInteraction('wages', people='people', jobs='jobs'),
        ],
    )
    trajectory = Simulation(system, solver=RK4Solver(dt=0.01), dt=0.5).run(until=20.0).trajectory_
    infected, wages = trajectory.values('people.infected')[-1], trajectory.values('people.wage')[-1]
    assert np.all(infected > 0.9)
    assert np.allclose(wages, [-20.0, 0.0, -20.0, 20.0, 20.0], atol=1e-9)
    isolated = System(
        components=[
            PersonComponent.lattice('people', space, infected=[1.0, 0.0, 0.0, 0.0, 0.0]),
            GroupComponent('jobs', element=JobComponent, space=space, on='edges', layer='work'),
        ],
        interactions=[ContagionInteraction('contagion', people='people', layer='work')],
    )
    trajectory = Simulation(isolated, solver=RK4Solver(dt=0.01), dt=0.5).run(until=20.0).trajectory_
    assert np.allclose(trajectory.values('people.infected')[-1], [1.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize('backend', BACKENDS)
def test_edge_states_weigh_neighbourhood_reductions(backend):
    """Test that a reduction over a network weighs every neighbour with the state of the edge that joins them."""
    space = NetworkSpace(nx.path_graph(4))
    WeightedInteraction = BaseInteraction.define(
        'WeightedInteraction',
        roles=['nodes', 'links'],
        laws=lambda s: [RateLaw(s.nodes.theta, neighbors(s.nodes.other.theta, weight=s.links.w))],
    )
    system = System(
        components=[
            OscillatorComponent.lattice('nodes', space, theta=[1.0, 2.0, 3.0, 4.0]),
            GroupComponent(
                'links',
                element=SynapseComponent,
                space=space,
                on='edges',
                values={'w': [10.0, 100.0, 1000.0]},
            ),
        ],
        interactions=[WeightedInteraction('weighted', nodes='nodes', links='links')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = [10.0 * 2, 10.0 * 1 + 100.0 * 3, 100.0 * 2 + 1000.0 * 4, 1000.0 * 3]
    assert np.allclose(problem.states.read(drift, 'nodes.theta'), expected, atol=1e-9)


@pytest.mark.parametrize('backend', BACKENDS)
def test_weighted_kuramoto_on_edges_matches_the_adjacency_coupling(backend):
    """Test that the coupling written once per edge matches the sum over the weighted adjacency of every node."""
    graph = nx.gnp_random_graph(12, 0.4, seed=4)
    space = NetworkSpace(graph)
    rng = np.random.default_rng(1)
    theta, weights = rng.uniform(0, 2 * np.pi, 12), rng.uniform(0.5, 1.5, graph.number_of_edges())
    system = System(
        components=[
            OscillatorComponent.lattice('oscillators', space, theta=theta),
            GroupComponent('links', element=SynapseComponent, space=space, on='edges', values={'w': weights}),
        ],
        interactions=[CouplingInteraction('coupling', oscillators='oscillators', links='links')],
    )
    problem = compile_system(system, backend=backend)
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    drift = problem.states.read(rates, 'oscillators.theta')
    matrix = np.zeros((12, 12))
    sources, targets = space.prepare().edges_[None]
    matrix[sources, targets] = matrix[targets, sources] = weights
    expected = (matrix * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
    assert np.allclose(drift, expected, atol=1e-12)


def test_edge_jumps_change_the_ends_of_their_edge():
    """Test that a jump law of an edge changes the states at its ends and that its channels follow the edges."""
    SISComponent = ElementComponent.define('SISComponent', states={'s': 1.0, 'i': 0.0})
    SpreadInteraction = BaseInteraction.define(
        'SpreadInteraction',
        roles=['nodes'],
        parameters={'beta': 1.0},
        over='neighbors',
        laws=lambda s: [JumpLaw(s.beta * s.source.i * s.target.s, {s.target.s: -1, s.target.i: 1})],
    )
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    space = NetworkSpace(graph)
    system = System(
        components=[SISComponent.lattice('nodes', space, s=[0.0, 1.0, 1.0, 1.0], i=[1.0, 0.0, 0.0, 0.0])],
        interactions=[SpreadInteraction('spread', nodes='nodes')],
    )
    trajectory = Simulation(system, dt=1.0, random_state=0).run(until=60.0).trajectory_
    assert trajectory.values('nodes.i')[-1].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_edge_interactions_run_on_lattice_bonds():
    """Test that an interaction over neighbours on a lattice runs once per bond of neighbouring sites."""
    DiffusionInteraction = BaseInteraction.define(
        'DiffusionInteraction',
        roles=['cells'],
        over='neighbors',
        laws=lambda s: [
            RateLaw(s.source.theta, s.target.theta - s.source.theta),
            RateLaw(s.target.theta, s.source.theta - s.target.theta),
        ],
    )
    system = System(
        components=[OscillatorComponent.lattice('cells', LatticeSpace(4), theta=[1.0, 0.0, 0.0, 0.0])],
        interactions=[DiffusionInteraction('diffusion', cells='cells')],
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [-2.0, 1.0, 0.0, 1.0]


def test_update_laws_set_the_edge_member_only():
    """Test that an update law of an edge sets the edge member, and one that sets an end raises an error."""
    ContactComponent = ElementComponent.define('ContactComponent', states={'contacted': 0.0})
    space = NetworkSpace(nx.path_graph(3))
    system = System(
        components=[
            PersonComponent.lattice('people', space, infected=[1.0, 0.0, 0.0]),
            GroupComponent('contacts', element=ContactComponent, space=space, on='edges'),
        ],
        interactions=[InfectionInteraction('infection', people='people', contacts='contacts')],
        time=DiscreteTime(),
    )
    with pytest.raises(ValidationError, match='at an end of every edge'):
        compile_system(system)


def test_edge_group_of_another_layer_is_rejected():
    """Test that an interaction that names another layer than its edge group raises a role error."""
    JobComponent = ElementComponent.define('JobComponent', states={'salary': 1.0})
    space = NetworkSpace(nx.path_graph(3), layers=[Layer('work', nx.complete_graph(3))])
    system = System(
        components=[
            PersonComponent.lattice('people', space),
            GroupComponent('jobs', element=JobComponent, space=space, on='edges'),
        ],
        interactions=[WagesInteraction('wages', people='people', jobs='jobs', layer='work')],
    )
    with pytest.raises(RoleError, match="layer 'work'"):
        compile_system(system)


def test_validation_reports_a_missing_layer():
    """Test that an interaction that names a layer the network lacks gets a binding issue listing the layers."""
    space = NetworkSpace(nx.path_graph(3), layers=[Layer('work', nx.complete_graph(3))])
    system = System(
        components=[PersonComponent.lattice('people', space)],
        interactions=[ContagionInteraction('contagion', people='people', layer='trade')],
    )
    issues = validate(system).issues
    assert [issue.code for issue in issues] == ['binding']
    assert "'work'" in issues[0].message


def test_no_base_is_another_base():
    """Test that no base class of the framework derives from another."""
    bases = [
        BaseComponent,
        BaseInteraction,
        BaseSpace,
        BaseTime,
        BaseLaw,
        BaseDomain,
        BaseProblem,
        BaseSolver,
        BaseBackend,
        BaseSystemCompiler,
    ]
    assert not [(a, b) for a in bases for b in bases if a is not b and issubclass(a, b)]
