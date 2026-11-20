"""Test agent-based systems on the Mesa backend: recognition, spaces, populations, reproducibility, and support."""

import json
from importlib.resources import files
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from shapely import Point, Polygon, box

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element, GroupElement, Member
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.components.spaces import (
    ContinuousSpace,
    DelaunayNeighborhood,
    DiscreteTime,
    LatticeSpace,
    NetworkSpace,
)
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile import PhaseSpec
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile.passes._agents_problem import AGENT_RESERVED_NAMES
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import AgentView, MesaSolver, StepSpec
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import (
    BackendError,
    IncompatibleLawsError,
    LabelError,
    SimulationError,
    SpaceError,
    ValidationError,
)
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.collective_behaviour import BoidsSystem
from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.systems.general.agents import MovementInteraction

from ....conftest import QueueInteraction, _queue_run

TOWNS = ['Bree', 'Dale', 'Esgaroth']
SEEN: list[object] = []
UNDECLARED: list[bool] = []
LOG: list[tuple[str, int]] = []
PROBED: list[object] = []
RECORDED_BOIDS = 'recorded_boids_two_dimensions.json'
PERIODIC_POSITIONS_GAP = (
    'Engine gap: the boundary of a periodic continuous space targets the position `x_<group>` of agents, '
    'which is not a state symbol (x_flock).'
)
NESTED_PHASES_GAP = (
    'Engine gap: the agent solver reads only the phases of the solver of the root system, not those of a '
    'nested system.'
)
RESERVED_KIND_GAP = (
    'Engine gap: the value `kind` of a plain element is reserved and read as the element class (reserved kind).'
)


def _rule(label, target, rule, phase=None, **roles):
    """Return a one-off interaction that gives a rule to the members of its target and binds other groups by role."""
    return Interaction(
        label,
        roles={'target': target, **roles},
        laws=lambda s: [UpdateLaw(s.target, rule, phase=phase)],
    )


def _breed(agent, system):
    """Create one offspring."""
    agent.create()


def _wander(agent, system):
    """Move to a random neighbouring cell and count the move."""
    cells = agent.get_neighbor_cells()
    agent.move_to(cells[int(system.rng.integers(len(cells)))])
    agent.moves += 1


def _probe(agent, system):
    """Record the number of neighbouring cells and of neighbours."""
    agent.cells = len(agent.get_neighbor_cells())
    agent.near = len(agent.get_neighbors())


def _count_near(agent, system):
    """Count the agents within the radius of the agent."""
    agent.near = len(agent.get_neighbors(radius=agent.radius))


def _count_triangulated(agent, system):
    """Count the agents joined to the agent by an edge of the Delaunay triangulation."""
    agent.near = len(agent.get_neighbors(over=DelaunayNeighborhood()))


def _stay(agent, system):
    """Stay in place."""


def _idle(agent, system):
    """Do nothing."""


def _breeders(size):
    """Return a herd of breeders, each with one offspring every step."""
    return System(elements=[Element('herd', size=size)], interactions=[_rule('breed', 'herd', _breed)])


def _walkers(size, space, **values):
    """Return walkers that move to a random neighbouring cell and count their moves."""
    walkers = Element('walkers', values={'moves': 0.0, **values}, size=size, space=space)
    return walkers, _rule('wander', 'walkers', _wander)


def _probes(label, size, space, rule=None, **values):
    """Return a group of probes that record their neighbouring cells and neighbours, and the rule that probes."""
    probes = Element(label, values={'cells': 0.0, 'near': 0.0, **values}, size=size, space=space)
    return probes, _rule('read', label, rule or _probe)


def _read_neighbours(agent, system):
    """Record the dimensions of the system and the numbers of neighbours within two radii."""
    PROBED.append((agent.id, system.dimensions, len(agent.get_neighbors(2.0)), len(agent.get_neighbors(4.0))))


def _simulate(system, until=0, random_state=0):
    """Return the trajectory of a system on the Mesa solver."""
    return system.set_params(solver=MesaSolver()).prepare().simulate(until=until, random_state=random_state)


def _concave_polygon():
    """Return an L of three unit squares around a missing fourth, with a hole of area 0.1 in the first square."""
    hole = [(0.2, 0.2), (0.7, 0.2), (0.7, 0.4), (0.2, 0.4)]
    return Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], holes=[hole]), box(0.2, 0.2, 0.7, 0.4)


def _lengths(values):
    """Return the length of the queue of every agent."""
    return np.asarray([len(item) for item in values], dtype=float)


def _post(agent, system):
    """Offer the next time."""
    agent.offer = system.time + 1.0


def _accept(agent, system):
    """Accept the best offer and count an offer that is not of this step."""
    best = max(employer.offer for employer in system.get_agents('employers', AgentView))
    agent.stale += float(best != system.time + 1.0)
    agent.accepted = best


def _employers(size):
    """Return employers that post an offer in the phase `offers`, and their rule."""
    return Element('employers', values={'offer': 0.0}, size=size), _rule('post', 'employers', _post, phase='offers')


class Give(Interaction):
    """Boltzmann exchange, where an agent with wealth gives the transfer to a random agent of its group."""

    def __init__(self, label: str, *, target: Role, transfer: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.transfer = transfer

    def _prepare(self, context):
        """Set the giving rule of the agents."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target, self.give)]

    def give(self, agent, system):
        """Give the transfer to a random agent of the group."""
        if agent.wealth > 0:
            transfer = system.get_parameter('transfer')
            agents = system.get_agents(agent.group, AgentView)
            other = agents[int(system.rng.integers(len(agents)))]
            other.wealth += transfer
            agent.wealth -= transfer


def _money_system():
    """Return the Boltzmann wealth system as an ordinary system of 50 agents."""
    agents = Element('agents', values={'wealth': 1.0}, size=50)
    return System('economy', elements=[agents], interactions=[Give('give', target='agents')])


class Culler(Interaction):
    """System rule that removes the agent with the smallest id once."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context):
        """Set the culling rule of the whole system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context, self.cull)]

    def cull(self, system):
        """Remove the oldest agent while more than two remain."""
        agents = sorted(system.get_agents(self.target, AgentView), key=lambda agent: agent.id)
        if len(agents) > 2:
            agents[0].remove()


def test_rules_make_groups_and_elements_agents():
    """Test that groups and elements bound to rules, and groups their rules grow, run on the agent stepper."""
    walkers, wander = _walkers(3, LatticeSpace((5, 5), boundary='fixed'))
    coin = Element('coin', values={'heads': 0})
    flip = _rule('flip', 'coin', _idle)
    herd = System(
        elements=[PopulationElement('herd')],
        interactions=[ExponentialGrowthInteraction('growth', target='herd')],
    )
    assert System(elements=[walkers], interactions=[wander]).prepare().form_ == FormKind.AGENT
    assert System(elements=[coin], interactions=[flip]).prepare().form_ == FormKind.AGENT
    assert _breeders(2).prepare().form_ == FormKind.AGENT
    assert herd.prepare().form_ != FormKind.AGENT


def _buy(agent, system):
    """Buy one unit from the first stall with stock."""
    stall = system.get_agents('stalls', AgentView)[0]
    if stall.stock > 0:
        stall.stock -= 1


def test_elements_without_laws_are_passive_agents():
    """Test that a group without laws next to agents with rules is a group of agents that the rules change."""
    system = System(
        elements=[Element('stalls', values={'stock': 3.0}, size=1), Element('shoppers', size=2)],
        interactions=[_rule('buy', 'shoppers', _buy, stalls='stalls')],
    )
    trajectory = system.prepare().simulate(until=2, random_state=0)
    assert [trajectory.get_agents('stalls', frame).values['stock'].tolist() for frame in (1, 2)] == [[1.0], [0.0]]


def test_rules_give_agent_form():
    """Test that a system of agent groups with rules compiles to the agent-based form."""
    walkers, wander = _walkers(3, LatticeSpace((5, 5), boundary='fixed'))
    system = System(elements=[walkers], interactions=[wander]).prepare()
    assert system.form_ == FormKind.AGENT
    assert system._problem.agents[0].states == ('moves',)


def test_rules_mixed_with_rates_raise_error():
    """Test that rules and rates in one system raise the law kind error."""
    walkers, wander = _walkers(3, LatticeSpace((5, 5), boundary='fixed'))
    system = System(
        elements=[walkers, PopulationElement('herd')],
        interactions=[wander, ExponentialGrowthInteraction('growth', target='herd')],
    )
    with pytest.raises(IncompatibleLawsError, match='RateLaw: herd; RuleLaw: walkers'):
        system.prepare()


def test_agents_are_born_every_step():
    """Test that one offspring per agent per step gives 10, 20, 40, and 80 agents."""
    trajectory = _breeders(10).prepare().simulate(until=3)
    assert trajectory.get_values('herd.count').tolist() == [10.0, 20.0, 40.0, 80.0]
    assert len(trajectory.get_agents('herd', 3).ids) == 80


def test_fixed_group_rejects_births():
    """Test that a rule creating a member of a group of fixed size raises an error that suggests a variable group."""

    class BreederElement(Element):
        """Agent that creates one offspring every step."""

        def _get_laws(self, s):
            """Return the breeding rule."""
            return [RuleLaw(_breed)]

    with pytest.raises(SimulationError, match='variable=True'):
        simulate(CompositeSystem(elements=[GroupElement('herd', element=BreederElement, size=10)]), until=1)


def test_system_rules_remove_agents():
    """Test that an interaction rule removes agents and frames report the agents alive."""
    walkers, wander = _walkers(5, LatticeSpace((5, 5), boundary='fixed'))
    system = System(elements=[walkers], interactions=[wander, Culler('cull', target='walkers')])
    trajectory = system.prepare().simulate(until=3, random_state=0)
    assert trajectory.get_values('walkers.count').tolist() == [5.0, 4.0, 3.0, 2.0]
    assert 1 not in trajectory.get_agents('walkers', 1).ids.tolist()


@pytest.mark.parametrize(('metric', 'expected'), [('chebyshev', 8), ('manhattan', 4)])
def test_grid_neighbours(metric, expected):
    """Test that an agent in the middle of a 10 by 10 grid has 8 Moore or 4 von Neumann neighbouring cells."""
    space = LatticeSpace((10, 10), boundary='fixed', metric=metric)
    probe, rule = _probes('probe', 1, space, x=5, y=5)
    snapshot = System(elements=[probe], interactions=[rule]).prepare().simulate(until=1).get_agents('probe', 1)
    assert snapshot.values['cells'].tolist() == [expected]


def test_network_neighbours_follow_edges():
    """Test that agents on a cycle network see their two neighbours."""
    nodes, rule = _probes('nodes', 6, NetworkSpace(nx.cycle_graph(6)))
    snapshot = System(elements=[nodes], interactions=[rule]).prepare().simulate(until=1).get_agents('nodes', 1)
    assert snapshot.values['near'].tolist() == [2.0] * 6
    assert sorted(snapshot.locations.tolist()) == list(range(6))


def test_continuous_neighbours_within_radius():
    """Test that agents in a continuous space find the agents within a radius."""
    space = ContinuousSpace(extent=(10.0, 10.0))
    values = {'near': 0.0, 'radius': 1.0, 'x': [1.0, 1.5, 8.0], 'y': [1.0, 1.0, 8.0]}
    drifters = Element('drifters', values=values, size=3, space=space)
    system = System(elements=[drifters], interactions=[_rule('count', 'drifters', _count_near)])
    snapshot = system.prepare().simulate(until=1).get_agents('drifters', 1)
    assert snapshot.values['near'].tolist() == [1.0, 1.0, 0.0]


def test_continuous_neighbours_of_a_delaunay_triangulation():
    """Test that the centre of a square of agents has four Delaunay neighbours and every corner three."""
    space = ContinuousSpace(extent=(10.0, 10.0))
    values = {'near': 0.0, 'x': [2.0, 6.0, 6.0, 2.0, 4.0], 'y': [2.0, 2.0, 6.0, 6.0, 4.0]}
    cells = Element('cells', values=values, size=5, space=space)
    system = System(elements=[cells], interactions=[_rule('count', 'cells', _count_triangulated)])
    snapshot = system.prepare().simulate(until=1).get_agents('cells', 1)
    assert snapshot.values['near'].tolist() == [3.0, 3.0, 3.0, 3.0, 4.0]


def test_agent_runs_are_reproducible():
    """Test that equal random states give equal agent trajectories."""
    runs = []
    for _ in range(2):
        walkers, wander = _walkers(5, LatticeSpace((10, 10), boundary='fixed'))
        runs.append(System(elements=[walkers], interactions=[wander]).prepare().simulate(until=5, random_state=0))
    first, second = (run.get_agents('walkers', 5) for run in runs)
    assert np.array_equal(first.locations, second.locations)
    assert first.values['moves'].tolist() == [5.0] * 5


def test_agent_frames_use_integer_steps():
    """Test that agent-based systems reject a non-integer frame duration."""
    with pytest.raises(ValidationError, match='positive integer'):
        _breeders(2).prepare().simulate(until=1, dt=0.5)


def test_boltzmann_wealth_keeps_wealth():
    """Test that the Boltzmann wealth system as an ordinary system keeps its total wealth and its agents' ids."""
    trajectory = _money_system().prepare().simulate(until=100, random_state=0)
    totals = {int(trajectory.get_agents('agents', frame).values['wealth'].sum()) for frame in range(101)}
    assert totals == {50}
    assert trajectory.get_agents('agents', 5).ids.tolist() == list(range(1, 51))


def test_agent_parameter_changes_live():
    """Test that a parameter update reaches the rules of the agents."""
    schedule = {'give.transfer': lambda t: 1.0 if t < 5 else 0.0}
    trajectory = _money_system().prepare().simulate(until=10, schedule=schedule, random_state=0)
    frames = [trajectory.get_agents('agents', frame).values['wealth'] for frame in (7, 10)]
    assert np.array_equal(*frames)


def test_mesa_declares_its_support():
    """Test that the Mesa backend declares discrete time and rule laws."""
    supports = MesaSolver._get_support()
    assert supports.time == {DiscreteTime}
    assert not supports.get_missing('laws', {RuleLaw})


def test_mesa_validates_systems_against_its_support():
    """Test that validation reports a system of rate laws and continuous time for the Mesa backend."""
    system = System(
        elements=[PopulationElement('herd')],
        interactions=[ExponentialGrowthInteraction('growth', target='herd')],
    )
    issues = PassCompiler(solver=MesaSolver()).prepare(system).report_.issues
    assert [issue.code for issue in issues] == ['solver-unsupported']
    assert 'RateLaw' in issues[0].message
    assert PassCompiler(solver=MesaSolver()).prepare(_money_system()).report_.ok


class Census(Interaction):
    """Interaction whose rule records its parameter, the groups of its roles, and the labels of their agents."""

    def __init__(
        self,
        label: str,
        *,
        counted: Role,
        others: Role,
        threshold: float = 0.9,
        looked_up: str = 'counted',
    ) -> None:
        super().__init__(label)
        self.counted = counted
        self.others = others
        self.threshold = threshold
        self.looked_up = looked_up

    def _prepare(self, context):
        """Set the census rule of the whole system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context, self.record)]

    def record(self, system):
        """Record the parameter, the counts and labels of the groups, and the space of the system."""
        counted = system.get_agents(self.looked_up, AgentView)
        SEEN.append(
            {
                'threshold': system.get_parameter('threshold'),
                'counted': len(counted),
                'others': system.count('others'),
                'labels': {agent.group_label for agent in counted},
                'paths': {agent.group for agent in counted},
                'dimensions': system.dimensions,
                'torus': system.torus,
            },
        )


def _census(looked_up='counted', space=None, schedule=None):
    """Return the records of a census of two groups of 42 and 43 agents nested in the subsystem `farm`."""
    SEEN.clear()
    farm = System(
        'farm',
        elements=[Element('cows', size=42), Element('patches', size=43)],
        interactions=[
            _rule('graze', 'cows', _stay),
            _rule('grow', 'patches', _stay),
            Census('census', counted='cows', others='patches', looked_up=looked_up),
        ],
    )
    system = System(systems=[farm], space=space)
    trajectory = system.prepare().simulate(until=3, schedule=schedule, random_state=0)
    return trajectory, list(SEEN)


def test_rule_reads_parameter_of_nested_owner():
    """Test that a nested interaction reads its own parameter by name, also after a live update."""
    _, records = _census(schedule={'farm/census.threshold': lambda t: 0.9 if t < 2 else 0.5})
    assert [record['threshold'] for record in records] == [0.9, 0.9, 0.5]


def test_rule_binds_two_agent_groups_through_roles():
    """Test that an interaction binds two agent groups of different sizes and reads each through its role."""
    trajectory, records = _census()
    assert trajectory.problem.form == FormKind.AGENT
    assert records[0]['counted'] == 42
    assert records[0]['others'] == 43
    assert records[0]['paths'] == {'farm/cows'}
    assert records[0]['labels'] == {'cows'}


def test_rule_reads_group_by_relative_path():
    """Test that a rule reads a group of its system by a path relative to the system."""
    _, records = _census(looked_up='patches')
    assert records[0]['counted'] == 43


def test_rule_reading_unknown_name_raises_error():
    """Test that a rule reading no role and no group lists the owner's roles and the available groups."""
    with pytest.raises(SimulationError, match=r"'meadow'.*counted, others.*farm/cows, farm/patches"):
        _census(looked_up='meadow')


def test_rule_reads_grid_dimensions_and_torus():
    """Test that a rule reads the extent and the wrap of a grid."""
    _, records = _census(space=LatticeSpace((30, 20), boundary='periodic'))
    assert records[0]['dimensions'] == (30, 20)
    assert records[0]['torus'] is True


def _views(space, **values):
    """Return views of two agents placed at the values of their positions in a space."""
    cows = Element('cows', values=values, size=2, space=space)
    stepper = _stepped(System(elements=[cows], interactions=[_rule('graze', 'cows', _stay)]), until=0)
    return [AgentView(agent, stepper) for agent in stepper._get_group_agents('cows')]


@pytest.mark.skip(reason=PERIODIC_POSITIONS_GAP)
def test_displacement_across_torus_seam():
    """Test that the displacement in a toroidal continuous space is the shortest image across the seam."""
    space = ContinuousSpace(extent=(100.0, 1.0), boundary='periodic')
    first, second = _views(space, x=[99.0, 1.0], y=[0.5, 0.5])
    assert first.get_displacement_to(second).tolist() == [2.0, 0.0]
    assert first.get_distance_to(second) == 2.0
    assert first.get_distance_to((97.0, 0.5)) == 2.0


def test_location_key_matches_the_cells_around_an_agent():
    """Test that the key of a grid location equals the neighbouring cell another agent stands on."""
    first, second = _views(LatticeSpace((5, 5), metric='chebyshev', boundary='fixed'), x=[1, 2], y=[1, 2])
    assert first.location_key == (1, 1)
    assert second.location_key in first.get_neighbor_cells()


def test_location_key_is_hashable_in_every_space():
    """Test that the key of a position is a tuple of floats and the key of a network node is the node."""
    first, _ = _views(ContinuousSpace(extent=(10.0, 10.0)), x=[1.5, 3.0], y=[2.0, 4.0])
    assert first.location_key == (1.5, 2.0)
    assert {first.location_key}
    node, _ = _views(NetworkSpace(nx.path_graph(5)), node=[0, 3])
    assert node.location_key == 0


def test_network_distance_counts_hops():
    """Test that the distance on a network counts hops and that a displacement raises a space error."""
    first, second = _views(NetworkSpace(nx.path_graph(5)), node=[0, 3])
    assert first.get_distance_to(second) == 3.0
    with pytest.raises(SpaceError, match='network'):
        first.get_displacement_to(second)


def test_agent_view_names_are_reserved():
    """Test that every public name of the agent view is reserved for the states of agents."""
    public = {name for name in dir(AgentView) if not name.startswith('_')}
    assert public <= AGENT_RESERVED_NAMES


def test_agent_state_with_reserved_name_raises_error():
    """Test that an agent state named like an attribute of the agent view raises a label error at compilation."""
    movers = Element('movers', values={'alive': 1.0}, size=2)
    system = System(elements=[movers], interactions=[_rule('move', 'movers', _idle)])
    with pytest.raises(LabelError, match='declare the variables alive'):
        system.prepare()


def test_non_agent_state_may_take_agent_view_name():
    """Test that an element without rule laws keeps states named like attributes of the agent view."""
    cells = Element('cells', values={'alive': 1.0}, size=2)
    decay = Interaction('decay', roles={'target': 'cells'}, laws=lambda s: [RateLaw(s.target.alive, -s.target.alive)])
    assert System(elements=[cells], interactions=[decay]).prepare().states_ == ('cells.alive[i]',)


def _record_time(agent, system):
    """Store the system time."""
    agent.seen = system.time


def _clocks(size, space=None, **values):
    """Return a group of clocks that record the system time at every step."""
    clocks = Element('clocks', values={'seen': -1.0, 'n': 0.0, **values}, size=size, space=space)
    return System(elements=[clocks], interactions=[_rule('record', 'clocks', _record_time)])


def test_rules_read_system_time():
    """Test that rules read the system time k during the step that produces frame k."""
    trajectory = _clocks(2).prepare().simulate(until=3)
    assert [trajectory.get_agents('clocks', step).values['seen'].tolist() for step in (1, 2, 3)] == [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]


def test_one_agent_takes_one_entry_values():
    """Test that a group of one agent reads a one-entry position and a one-entry value as its own."""
    system = _clocks(1, LatticeSpace((5, 5), boundary='fixed'), n=[3.0], x=[1], y=[1])
    snapshot = system.prepare().simulate(until=0).get_agents('clocks', 0)
    assert snapshot.locations.tolist() == [[1, 1]]
    assert snapshot.values['n'].tolist() == [3.0]


def test_mesa_solver_run_sets_agents_and_snapshot():
    """Test that every run of the Mesa solver sets the alive agents by group and the snapshot of the step."""
    problem = _breeders(2).prepare()._problem
    solver = MesaSolver().set_params(random_state=np.random.default_rng(0), until=3.0).prepare(problem)
    assert len(solver.agents_['herd']) == 2
    solver.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0))
    assert len(solver.agents_['herd']) == 8
    assert solver.snapshot_.reporters['herd.count'] == 8.0
    for name in ('agents_of', 'snapshot', 'create_agent', 'resolve_group', 'spec_of', 'group_agents'):
        assert not hasattr(solver, name)


def _forage(agent, system):
    """Gather one unit of food and try to read a value the agent does not have."""
    agent.food += 1
    try:
        _ = agent.wealth
    except AttributeError:
        UNDECLARED.append(True)


def test_agent_views_read_and_set_states_as_properties():
    """Test that a rule sets a state through its view and that a value the agent lacks raises Python's own error."""
    UNDECLARED.clear()
    foragers = Element('foragers', values={'food': 0.0}, size=2)
    system = System(elements=[foragers], interactions=[_rule('forage', 'foragers', _forage)])
    trajectory = system.prepare().simulate(until=3, dt=1.0, random_state=0)
    assert trajectory.get_agents('foragers', 3).values['food'].tolist() == [3.0, 3.0]
    assert UNDECLARED
    assert all(UNDECLARED)


def _save(agent, system):
    """Record the number of remembered values of the money, and save one unit."""
    agent.seen = float(len(agent.get_lags('money')))
    agent.money += 1


@pytest.mark.skip(
    reason='Engine gap: an agent value has no form of the new API for a memory of its last 3 values, which '
    '`get_lags` reads (agent lags).',
)
def test_memory_window():
    """Agents remember the last values of the states their type lists."""
    savers = Element('savers', values={'money': 0.0, 'seen': 0.0}, size=2)
    system = System(elements=[savers], interactions=[_rule('save', 'savers', _save)])
    trajectory = system.prepare().simulate(until=5, random_state=0)
    assert trajectory.get_agents('savers', 5).values['seen'].tolist() == [3.0, 3.0]


class AccountElement(Element):
    """Agent that names a member of the group `stalls` and records the stock of that member."""

    def __init__(
        self,
        label: str,
        *,
        stall: Annotated[int, Member('stalls')] = -1,
        seen: float = -1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.stall = stall
        self.seen = seen


def _read_stall(agent, system):
    """Record the stock of the member the reference names, or -1 for none."""
    stall = agent.get_referenced('stall')
    agent.seen = -1.0 if stall is None else stall.stock


def _accounts(stall, stalls=None):
    """Return accounts that name members of the stalls and record their stock, next to the stalls if given."""
    accounts = AccountElement('accounts', stall=np.asarray(stall), size=len(stall))
    if stalls is None:
        return System(elements=[accounts], interactions=[_rule('read', 'accounts', _read_stall)])
    stalls = Element('stalls', values={'stock': np.asarray(stalls)}, size=len(stalls))
    return System(
        elements=[stalls, accounts],
        interactions=[_rule('read', 'accounts', _read_stall, stalls='stalls')],
    )


def test_reference_states_name_members_of_another_group_by_id():
    """Test that initial positions in the referenced group become ids that `referenced` resolves to the members."""
    trajectory = _accounts([1, -1, 0], [5.0, 7.0]).prepare().simulate(until=1, random_state=0)
    snapshot = trajectory.get_agents('accounts', 1)
    assert snapshot.values['seen'].tolist() == [7.0, -1.0, 5.0]
    assert snapshot.values['stall'].tolist() == [2.0, -1.0, 1.0]


def test_reference_to_a_group_the_system_lacks_names_no_member():
    """Test that a reference state names no member when the system has no referenced group."""
    snapshot = _accounts([-1, -1]).prepare().simulate(until=1, random_state=0).get_agents('accounts', 1)
    assert snapshot.values['seen'].tolist() == [-1.0, -1.0]


class Budding(Interaction):
    """Interaction whose rule gives the first agent a child at the cell `(3, 4)`."""

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context):
        """Set the rule of the agents."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context.target, self.bud)]

    def bud(self, agent, system):
        """Create a child at the cell `(3, 4)` while the agent is alone."""
        if system.count('target') == 1:
            agent.create(None, (3, 4), mass=1.0)


def test_created_agent_takes_the_coordinates_of_its_location():
    """Test that an agent created at a cell without its positions takes the coordinates of that cell."""
    cells = Element('cells', values={'x': 0, 'y': 1, 'mass': 2.0}, size=1, space=LatticeSpace((5, 5)))
    system = System(elements=[cells], interactions=[Budding('budding')]).prepare()
    snapshot = system.simulate(until=1, random_state=0).get_agents('cells', 1)
    assert snapshot.values['x'].tolist() == [0, 3]
    assert snapshot.values['y'].tolist() == [1, 4]


class Hop(Interaction):
    """Interaction whose rule moves every walker to the next node of a ring of named towns."""

    def __init__(self, label: str, *, target: Role = 'walkers') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context):
        """Set the rule of the walkers."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context.target, self.hop)]

    def hop(self, agent, system):
        """Move to the next town by changing the value `node`."""
        agent.node = TOWNS[(TOWNS.index(agent.node) + 1) % len(TOWNS)]


def test_agents_on_a_network_start_at_the_node_their_value_gives_and_move_by_changing_it():
    """Test that more walkers than towns start at the towns their values `node` name and hop along the ring."""
    ring = NetworkSpace(nx.cycle_graph(TOWNS))
    walkers = Element('walkers', values={'node': ['Bree', 'Bree', 'Dale', 'Esgaroth']}, size=4, space=ring)
    trajectory = System(elements=[walkers], interactions=[Hop('hop')], space=ring).prepare().simulate(until=1)
    assert trajectory.get_agents('walkers', 0).locations.tolist() == ['Bree', 'Bree', 'Dale', 'Esgaroth']
    assert trajectory.get_agents('walkers', 1).locations.tolist() == ['Dale', 'Dale', 'Esgaroth', 'Bree']
    assert trajectory.get_agents('walkers', 1).values['node'].tolist() == ['Dale', 'Dale', 'Esgaroth', 'Bree']


def test_reference_beyond_the_referenced_group_raises():
    """Test that an initial position beyond the members of the referenced group raises an error."""
    with pytest.raises(SimulationError, match='position 3'):
        _accounts([3], [3.0]).prepare().simulate(until=1, random_state=0)


def test_memory_of_unremembered_state_raises():
    """Reading the memory of a state the type does not remember raises an error."""
    agents = Element('agents', values={'money': 0.0}, size=1)
    system = System(
        elements=[agents],
        interactions=[_rule('read', 'agents', lambda agent, system: agent.get_lags('money'))],
    )
    with pytest.raises(KeyError, match='remember'):
        system.prepare().simulate(until=1)


def _tick(agent, system):
    """Log the agent with the tag of the first rule."""
    LOG.append(('element', agent.id))


class Echo(Interaction):
    """Interaction whose members log their id in their own activation, with the gain of the interaction."""

    def __init__(self, label: str, *, target: Role, gain: float = 2.0) -> None:
        super().__init__(label)
        self.target = target
        self.gain = gain

    def _prepare(self, context):
        """Set the rule of the members."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target, self.echo)]

    def echo(self, agent, system):
        """Log the agent with the gain of the interaction."""
        LOG.append((f'member {system.get_parameter("gain")}', agent.id))


def test_member_rules_run_in_the_activation_of_every_member_after_its_first_rules():
    """Test that the rule of a second interaction runs right after the rule of the first for each agent."""
    LOG.clear()
    tickers = Element('tickers', size=5)
    system = System(elements=[tickers], interactions=[_rule('tick', 'tickers', _tick), Echo('echo', target='tickers')])
    system.prepare().simulate(until=2, random_state=0)
    assert len(LOG) == 20
    for first, second in zip(LOG[::2], LOG[1::2], strict=True):
        assert (first[0], second) == ('element', ('member 2.0', first[1]))


class VisitedView(AgentView):
    """View of a visited agent, with its visits, its trait, and the weight of its group."""

    visits: float
    trait: float
    weight: float


def _visit(agent, system):
    """Count the visit, record its time and the counts of the traits, and flip the trait."""
    agent.visits += 1
    agent.trait = 1.0 - agent.trait
    traits = [item.trait for item in system.get_agents('visited')]
    system.storage.setdefault('times', []).append(system.time)
    system.storage.setdefault('compositions', []).append(
        (system.get_composition('visited', 'trait'), {value: traits.count(value) for value in set(traits)}),
    )


def _stepped(
    system: System,
    until: int,
    random_state: int = 0,
    observables: tuple[Observable, ...] = (),
) -> MesaSolver:
    """Return the agent-based solver of a system after it stepped one unit of time at a time until a time."""
    problem = system.prepare().simulate(until=0, random_state=random_state, observables=observables).problem
    solver = system.solver if isinstance(system.solver, MesaSolver) else MesaSolver()
    solver = solver.set_params(random_state=random_state, until=until).prepare(problem)
    state, parameters = solver.initial_state_, problem.initial_parameters
    for time in range(until):
        state = solver.run(StepSpec(time, state, parameters, time + 1), final=time + 1 == until).state_
    return solver


def _visited_system(phase: PhaseSpec, size: int = 4, interactions=()) -> System:
    """Return a group of visited agents whose rule runs in a phase of the solver."""
    visited = Element('visited', values={'visits': 0.0, 'trait': 0.0, 'weight': 2.0}, size=size)
    return System(
        elements=[visited],
        interactions=[_rule('visit', 'visited', _visit, phase='visit'), *interactions],
        solver=MesaSolver(phases=[phase]),
    )


def _visits(visits: float) -> PhaseSpec:
    """Return a phase of visits of agents drawn with replacement."""
    return PhaseSpec('visit', activation='visits', visits=visits)


def test_visits_draw_agents_with_replacement():
    """Test that a phase of visits activates agents drawn with replacement, visits times the size of the group."""
    trajectory = _visited_system(_visits(2.5)).prepare().simulate(until=4, random_state=0)
    visits = trajectory.get_agents('visited', 4).values['visits']
    assert visits.sum() == 40
    assert len(set(visits.tolist())) > 1


def test_visits_advance_the_time_by_a_fraction_of_the_step():
    """Test that rules read the time `k + i / n` at visit `i` of `n` in step `k`."""
    solver = _stepped(_visited_system(_visits(1.0)), until=2)
    assert solver.storage_['visit']['times'] == [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]


def test_composition_counts_the_values_of_a_state():
    """Test that the composition of a group keeps the counts of a state as agents change it."""
    compositions = _stepped(_visited_system(_visits(3.0)), until=2, random_state=1).storage_['visit']['compositions']
    assert len(compositions) == 24
    assert all(kept == counted for kept, counted in compositions)


def test_periodic_phase_runs_every_period_steps_from_its_offset():
    """Test that a phase of period three and offset one runs in the steps 1, 4, and 7 only."""
    phase = PhaseSpec('visit', activation='sequential', period=3, offset=1)
    assert _stepped(_visited_system(phase, size=1), until=8).storage_['visit']['times'] == [1.0, 4.0, 7.0]


class OccupancyReader(Interaction):
    """Interaction that stores, at every step, the counts of the walkers by cell."""

    def _prepare(self, context):
        """Set the reading rule of the whole system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context, self.read)]

    def read(self, system):
        """Store the counts of the walkers by cell."""
        system.storage.setdefault('occupancies', []).append(system.get_occupancy('walkers'))


def test_occupancy_counts_the_agents_of_a_group_by_cell():
    """Test that the occupancy of a group counts its agents in every cell that holds one, by the key of the cell."""
    walkers, wander = _walkers(3, LatticeSpace((4, 4), boundary='periodic'), x=[0, 0, 2], y=[0, 0, 1])
    system = System(elements=[walkers], interactions=[wander, OccupancyReader('reader')])
    solver = _stepped(system, until=2)
    occupancies = solver.storage_['reader']['occupancies']
    locations = solver.snapshot_.groups['walkers'].locations.tolist()
    expected = {}
    for location in locations:
        expected[tuple(location)] = expected.get(tuple(location), 0) + 1
    assert occupancies[-1] == expected
    assert sum(occupancies[0].values()) == 3


class TotalReader(Interaction):
    """Interaction that stores, at every step, the history of the observable it reads."""

    def _prepare(self, context):
        """Set the reading rule of the whole system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context, self.read, phase='visit')]

    def read(self, system):
        """Store the history of the total of the visits."""
        system.storage.setdefault('histories', []).append(system.get_history('total').tolist())


def test_history_reads_an_observable_at_earlier_steps():
    """Test that a rule reads the values of an observable of the run at the end of every earlier step."""
    system = _visited_system(_visits(1.0), interactions=[TotalReader('reader')])
    observables = [Observable('total', group='visited', variable='visits', reduce='sum')]
    solver = _stepped(system, until=3, observables=tuple(observables))
    assert solver.storage_['reader']['histories'] == [[], [4.0], [4.0, 8.0]]


def _weight_of(agent: VisitedView) -> float:
    """Return the weight of an agent, a parameter of its group."""
    return agent.weight


def test_observables_read_agent_views():
    """Test that an observable of an agent group reads the views of its agents, with the parameters of the group."""
    observables = [Observable('weights', group='visited', expression=_weight_of, reduce='sum')]
    trajectory = _visited_system(_visits(1.0)).prepare().simulate(until=1, observables=observables)
    assert trajectory.get_values('weights').tolist() == [8.0, 8.0]


def _hold(agent, system):
    """Count the tokens of the neighbouring nodes, and the agents of every group there."""
    agent.around = len(agent.get_neighbors(group='tokens'))
    agent.every = len(agent.get_neighbors())


def test_network_neighbours_of_one_group_read_the_member_sets_of_the_nodes_around():
    """Test that a holder on a path reads the tokens on its neighbouring nodes apart from the holders there."""
    path = NetworkSpace(nx.path_graph(4))
    holders = Element('holders', values={'around': 0.0, 'every': 0.0, 'node': [0, 1, 2, 3]}, size=4, space=path)
    tokens = Element('tokens', values={'node': [0, 0, 1, 2, 3, 3]}, size=6, space=path)
    system = System(elements=[holders, tokens], interactions=[_rule('hold', 'holders', _hold, tokens='tokens')])
    snapshot = system.prepare().simulate(until=1, random_state=0).get_agents('holders', 1)
    places = snapshot.locations.tolist()
    assert dict(zip(places, snapshot.values['around'].tolist(), strict=True)) == {0: 1.0, 1: 3.0, 2: 3.0, 3: 1.0}
    assert dict(zip(places, snapshot.values['every'].tolist(), strict=True)) == {0: 2.0, 1: 5.0, 2: 5.0, 3: 2.0}


def _toss(agent, system):
    """Draw a named uniform value and the next value of a copy of the generator."""
    copy = np.random.Generator(type(system.rng.bit_generator)())
    copy.bit_generator.state = system.rng.bit_generator.state
    agent.expected = 2 + 3 * copy.random()
    agent.drawn = system.draw('toss', 'uniform', low=2, high=5)


def _coins(rule):
    """Return a group of five coins whose rule draws."""
    coins = Element('coins', values={'drawn': 0.0, 'expected': 0.0}, size=5)
    return System(elements=[coins], interactions=[_rule('toss', 'coins', rule)])


def test_rule_draw_samples_from_the_generator_of_the_system():
    """Test that the named draw of a rule samples from the generator of the system and is recorded."""
    trajectory = _coins(_toss).prepare().simulate(until=3, random_state=0, record_draws=True)
    agents = trajectory.get_agents('coins', 3).values
    np.testing.assert_allclose(agents['drawn'], agents['expected'])
    record = trajectory.draws.draws['toss.toss']
    assert record.values.size == 15
    assert np.all(record.arguments['low'] == 2)


def test_rule_draw_rejects_an_unknown_distribution():
    """Test that a named draw of an unknown distribution raises a named error."""
    system = _coins(lambda agent, system: system.draw('toss', 'wobbly', a=1))
    with pytest.raises(ValidationError, match="distribution 'wobbly'"):
        system.prepare().simulate(until=1, random_state=0)


def test_recorded_two_dimensional_trajectories_load():
    """Test that the trajectories recorded before three-dimensional spaces load with their frames and shapes."""
    recorded = json.loads((files(__package__) / RECORDED_BOIDS).read_text())
    assert sorted(int(frame) for frame in recorded) == [0, 1, 5, 10, 20]
    assert np.asarray(recorded['20']['locations']).shape == (10, 2)


@pytest.mark.skip(reason=PERIODIC_POSITIONS_GAP)
def test_two_dimensional_space_keeps_recorded_trajectories():
    """Test that a system on a two-dimensional continuous space repeats the trajectories recorded before the change."""
    recorded = json.loads((files(__package__) / RECORDED_BOIDS).read_text())
    for _ in range(2):
        trajectory = BoidsSystem(10).prepare().simulate(until=20, dt=1, random_state=0)
        for frame, values in recorded.items():
            snapshot = trajectory.get_agents('flock', int(frame))
            assert snapshot.ids.tolist() == values['ids']
            np.testing.assert_allclose(snapshot.locations, values['locations'], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(snapshot.values["x'"], values['vx'], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(snapshot.values["y'"], values['vy'], rtol=1e-9, atol=1e-9)


def _depth_probes(size, rule, space, **values):
    """Return a system of probes in a box with a rule of their own."""
    probes = Element('probes', values={'near': 0.0, **values}, size=size, space=space)
    return System(elements=[probes], interactions=[_rule('read', 'probes', rule)])


def test_three_dimensional_box_places_agents_inside():
    """Test that the system view reports three dimensions, and agents without a position start inside the box."""
    PROBED.clear()
    system = _depth_probes(200, _read_neighbours, ContinuousSpace(extent=(10.0, 5.0, 2.0)))
    locations = _simulate(system, until=1).get_agents('probes', 0).locations
    assert PROBED[0][1] == (10.0, 5.0, 2.0)
    assert locations.shape == (200, 3)
    assert locations.min() >= 0.0
    assert np.all(locations.max(axis=0) <= (10.0, 5.0, 2.0))
    assert locations[:, 2].max() > 1.0


def _count_close(agent, system):
    """Count the neighbours within 0.5."""
    agent.near = float(len(agent.get_neighbors(0.5)))


def _depth_shift(agent, system):
    """Keep the depth of the shortest vector to the other agent."""
    other = next(item for item in system.get_agents('probes') if item.id != agent.id)
    agent.near = float(agent.get_displacement_to(other)[2])


@pytest.mark.skip(reason=PERIODIC_POSITIONS_GAP)
def test_neighbours_across_the_depth_boundary():
    """Test that two agents across the periodic third side find each other within a radius of 0.5."""
    space = ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic')
    system = _depth_probes(2, _count_close, space, x=5.0, y=5.0, z=[0.2, 9.9])
    assert _simulate(system, until=1).get_agents('probes', 1).values['near'].tolist() == [1.0, 1.0]


@pytest.mark.skip(reason=PERIODIC_POSITIONS_GAP)
def test_displacement_takes_the_minimum_image_in_depth():
    """Test that the displacement between agents across the third side wraps every coordinate."""
    space = ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic')
    system = _depth_probes(2, _depth_shift, space, x=5.0, y=5.0, z=[0.2, 9.9])
    near = _simulate(system, until=1).get_agents('probes', 1).values['near']
    np.testing.assert_allclose(near, [-0.3, 0.3])


def test_depth_counts_in_the_distance():
    """Test that agents that differ by 3 in the third coordinate are neighbours within 4 and not within 2."""
    PROBED.clear()
    system = _depth_probes(2, _read_neighbours, ContinuousSpace(extent=(10.0, 10.0, 10.0)), x=5.0, y=5.0, z=[1.0, 4.0])
    _simulate(system, until=1)
    assert [item[2:] for item in PROBED] == [(0, 1), (0, 1)]


def _walk_in_box(size, step, **values):
    """Return walkers in a periodic box of side 10 that a movement moves by a step."""
    space = ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic')
    walkers = Element('walkers', values=values, size=size, space=space)
    walk = MovementInteraction('walk', target='walkers', step=step, scale=0.1)
    return System(elements=[walkers], interactions=[walk])


@pytest.mark.skip(reason=PERIODIC_POSITIONS_GAP)
def test_gaussian_steps_spread_in_depth():
    """Test that the variance of the third coordinate grows under Gaussian steps and positions stay in the box."""
    trajectory = _simulate(_walk_in_box(100, 'gaussian', x=5.0, y=5.0, z=5.0), until=100)
    variances = [trajectory.get_agents('walkers', frame).locations[:, 2].var() for frame in (0, 10, 100)]
    assert variances[0] < variances[1] < variances[2]
    for frame in (1, 50, 100):
        locations = trajectory.get_agents('walkers', frame).locations
        assert locations.min() >= 0.0
        assert locations.max() <= 10.0


def test_trajectory_reports_three_coordinates_per_agent():
    """Test that a trajectory holds one row of 3 coordinates per agent alive at frame 5, with the agent ids."""
    snapshot = _simulate(_walk_in_box(7, 'gaussian'), until=5).get_agents('walkers', 5)
    assert snapshot.locations.shape == (7, 3)
    assert len(snapshot.ids) == 7


@pytest.mark.parametrize(
    ('space', 'dimension'),
    [
        (ContinuousSpace(extent=(5.0, 5.0, 5.0, 5.0)), '4 dimensions'),
        (ContinuousSpace(n_dim=2), '2 dimensions'),
        (ContinuousSpace(extent=(5.0,)), '1 dimensions'),
    ],
)
def test_unsupported_continuous_space_is_rejected_before_the_first_frame(space, dimension):
    """Test that a space that is no box of two or three sides raises a named error when the solver prepares."""
    system = _depth_probes(2, _read_neighbours, space)
    problem = PassCompiler(solver=MesaSolver()).prepare(system).run().problem_
    with pytest.raises(BackendError, match=f"continuous space 'space'.*{dimension}.*two or three dimensions"):
        MesaSolver().prepare(problem)
    with pytest.raises(BackendError, match='two or three dimensions'):
        _simulate(system, until=1)


@pytest.mark.parametrize(
    'location',
    [(1.0, 2.0), [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)], box(0.0, 0.0, 1.0, 1.0)],
)
def test_location_of_the_wrong_length_is_reported(location):
    """Test that the compile report holds an initial location that does not fit a three-dimensional space."""
    group = GroupElement('probes', element=Element, size=3, location=location)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
    report = PassCompiler().prepare(system).report_
    issue = next(item for item in report.issues if item.code == 'location-dimension')
    assert issue.path == 'probes'
    assert '2 coordinates' in issue.message
    assert '3 dimensions' in issue.message


def test_locations_that_fit_their_space_are_not_reported():
    """Test that fitting positions and polygons in a two-dimensional space leave the compile report clean."""
    for location in ((1.0, 2.0, 3.0), [(1.0, 2.0, 3.0)] * 2):
        group = GroupElement('probes', element=Element, size=2, location=location)
        system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
        assert PassCompiler().prepare(system).report_.ok
    group = GroupElement('probes', element=Element, size=2, location=[box(0.0, 0.0, 1.0, 1.0)] * 2)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
    assert PassCompiler().prepare(system).report_.ok


def test_move_to_of_the_wrong_length_raises_space_error():
    """Test that a move to a location that does not fit the space raises a named error during the run."""
    space = ContinuousSpace(extent=(5.0, 5.0, 5.0))
    walkers = Element('walkers', values={'x': 1.0, 'y': 1.0, 'z': 1.0}, size=1, space=space)
    walk = MovementInteraction('walk', target='walkers', step=lambda agent, system: (2.0, 2.0))
    with pytest.raises(SpaceError, match='has 2 coordinates, and the space has 3 dimensions'):
        _simulate(System(elements=[walkers], interactions=[walk]), until=1)


def _people(size, space, rule=_idle, **values):
    """Return a group of people on a space with a rule of their own."""
    people = Element('people', values=values, size=size, space=space)
    return System(elements=[people], interactions=[_rule('live', 'people', rule)])


def test_agents_per_census_region():
    """Test that members start inside the regions their values name."""
    regions = [box(0.0, 0.0, 4.0, 4.0), box(5.0, 5.0, 9.0, 9.0), box(0.0, 5.0, 4.0, 9.0)]
    space = ContinuousSpace(extent=(10.0, 10.0), regions=dict(zip(('north', 'south', 'west'), regions, strict=True)))
    system = _people(150, space, region=['north'] * 100 + ['south'] * 50)
    snapshot = _simulate(system).get_agents('people', 0)
    assert all(regions[0].contains(Point(point)) for point in snapshot.locations[:100])
    assert all(regions[1].contains(Point(point)) for point in snapshot.locations[100:])


def test_one_region_is_shared_by_every_member():
    """Test that one region as the start of a group places every member at its own point inside it."""
    region = box(1.0, 1.0, 2.0, 2.0)
    system = _people(5, ContinuousSpace(extent=(5.0, 5.0), regions={'square': region}), region='square')
    locations = _simulate(system).get_agents('people', 0).locations
    assert all(region.contains(Point(point)) for point in locations)
    assert len({tuple(point) for point in locations}) == 5


def test_region_placement_is_reproducible():
    """Test that two runs with one random state start the agents at the same points, and another seed differs."""

    def start(random_state):
        space = ContinuousSpace(extent=(5.0, 5.0), regions={'square': box(0.0, 0.0, 4.0, 4.0)})
        return (
            _simulate(_people(20, space, region='square'), random_state=random_state).get_agents('people', 0).locations
        )

    np.testing.assert_array_equal(start(0), start(0))
    assert not np.array_equal(start(0), start(1))


def test_placement_is_uniform_in_a_concave_polygon_with_a_hole():
    """Test that 20000 agents lie in an L with a hole, with the share of each unit square equal to its area share."""
    polygon, hole = _concave_polygon()
    system = _people(20000, ContinuousSpace(extent=(2.0, 2.0), regions={'l': polygon}), region='l')
    locations = _simulate(system).get_agents('people', 0).locations
    assert all(polygon.contains(Point(point)) for point in locations[:2000])
    x, y = locations[:, 0], locations[:, 1]
    assert not np.any((x > 0.2) & (x < 0.7) & (y > 0.2) & (y < 0.4))
    assert not np.any((x > 1.0) & (y > 1.0))
    shares = [np.mean((x < 1) & (y < 1)), np.mean((x >= 1) & (y < 1)), np.mean((x < 1) & (y >= 1))]
    np.testing.assert_allclose(shares, np.array([0.9, 1.0, 1.0]) / polygon.area, atol=0.02)
    assert hole.area == pytest.approx(0.1)


def test_move_to_a_favourite_place():
    """Test that a step returning the polygon of a region moves the agent into it, and the agent finds it there."""
    places = [box(0.0, 0.0, 2.0, 2.0), box(3.0, 3.0, 5.0, 5.0), box(4.0, 4.0, 6.0, 6.0)]
    found = []

    def to_favourite(agent, system):
        found.append(system.get_regions('places'))
        return system.get_region('places', 1)

    def read(agent, system):
        found.append(agent.get_regions_here('places'))

    space = ContinuousSpace(extent=(10.0, 10.0), regions={'places': places, 'home': places[0]})
    people = Element('people', values={'region': 'home'}, size=30, space=space)
    move = MovementInteraction('move', target='people', step=to_favourite, schedule='member')
    system = System(elements=[people], interactions=[_rule('read', 'people', read), move])
    locations = _simulate(system, until=2).get_agents('people', 2).locations
    assert all(places[1].contains(Point(point)) for point in locations)
    assert 3 in found
    assert [0] in found
    assert any(1 in item for item in found if isinstance(item, list))


def test_created_agent_is_placed_in_a_polygon():
    """Test that a created agent starts inside the polygon of its location."""
    region = box(1.0, 1.0, 2.0, 2.0)

    def spawn(agent, system):
        if system.count('people') == 1:
            system.create('people', location=region)

    system = _people(1, ContinuousSpace(extent=(5.0, 5.0)), spawn, x=4.0, y=4.0)
    locations = _simulate(system, until=1).get_agents('people', 1).locations
    assert locations.shape == (2, 2)
    assert region.contains(Point(locations[1]))


@pytest.mark.parametrize(
    ('step', 'error', 'message'),
    [
        (lambda agent, system: agent.get_regions_here('absent'), SimulationError, 'no region layer'),
        (lambda agent, system: system.get_region('places', 5), SimulationError, 'has 1 regions'),
        (lambda agent, system: Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]), SpaceError, 'invalid polygon'),
    ],
)
def test_region_queries_and_invalid_polygons_raise_named_errors(step, error, message):
    """Test that an unknown layer, an unknown region, and an invalid polygon raise errors that say what is wrong."""
    space = ContinuousSpace(extent=(5.0, 5.0), regions={'places': [box(0.0, 0.0, 1.0, 1.0)]})
    people = Element('people', values={'x': 1.0, 'y': 1.0}, size=1, space=space)
    move = MovementInteraction('move', target='people', step=step)
    with pytest.raises(error, match=message):
        _simulate(System(elements=[people], interactions=[_rule('live', 'people', _idle), move]), until=1)


def test_polygon_is_no_location_of_a_lattice():
    """Test that a polygon as the location of an agent on a lattice raises a named error."""

    def spawn(agent, system):
        system.create('people', location=box(0.0, 0.0, 1.0, 1.0))

    with pytest.raises(SpaceError, match='location of a continuous space'):
        _simulate(_people(1, LatticeSpace((5, 5)), spawn, x=0, y=0), until=1)


def test_snapshots_keep_tuples_as_they_were():
    """Test that each snapshot keeps one tuple per agent, in a one-dimensional array of objects."""
    trajectory = _queue_run()
    first, last = trajectory.get_agents('queues', 0), trajectory.get_agents('queues', 3)
    assert first.values['orders'].shape == (2,)
    assert first.values['orders'].dtype == object
    assert first.values['orders'].tolist() == [(), (7,)]
    assert last.values['orders'].tolist() == [(2, 3), (2, 3)]


def _queues():
    """Return two queues, the first empty and the second with the order 7."""
    orders = np.empty(2, dtype=object)
    orders[:] = [(), (7,)]
    return Element('queues', values={'orders': orders}, size=2)


def test_composition_counts_agents_by_tuple():
    """Test that the composition of a group counts its agents by the tuples they hold."""
    seen = []

    class CountingQueue(QueueInteraction):
        """Rule of a queue that records the composition of its group."""

        def _step(self, agent, system):
            """Record the composition, and then step."""
            seen.append(dict(system.get_composition('queues', 'orders')))
            super()._step(agent, system)

    system = System(elements=[_queues()], interactions=[CountingQueue('queue', target='queues')])
    system.prepare().simulate(until=1, random_state=0)
    assert seen[0] == {(): 1, (7,): 1}
    assert sum(seen[1].values()) == 2


def test_history_keeps_reported_objects():
    """Test that the reports of a run keep an object at every frame."""
    values = _queue_run().get_values('queue.longest')
    assert values.dtype == object
    assert isinstance(values[-1], tuple)


def test_observable_keeps_a_hashable_value():
    """Test that an observable of a group by a function of its values may give an object."""
    observables = [Observable('lengths', group='queues', variable='orders', reduce=_lengths)]
    system = System(elements=[_queues()], interactions=[QueueInteraction('queue', target='queues')])
    trajectory = system.prepare().simulate(until=2, observables=observables, random_state=0)
    assert trajectory.get_values('lengths')[0] == pytest.approx([0, 1])


def _market(label='system', solver=None, workers=0):
    """Return employers that post offers, and workers that accept the best one, in the phases of a solver."""
    employers, post = _employers(3 if workers else 2)
    elements, interactions = [employers], [post]
    if workers:
        elements.append(Element('workers', values={'accepted': -1.0, 'stale': 0.0}, size=workers))
        interactions.append(_rule('accept', 'workers', _accept, phase='acceptance', employers='employers'))
    return System(label, elements=elements, interactions=interactions, solver=solver or 'auto')


def test_offer_before_acceptance():
    """Every acceptance refers to an offer made earlier in the same step."""
    solver = MesaSolver(phases=[PhaseSpec('offers'), PhaseSpec('acceptance')])
    trajectory = _market(solver=solver, workers=5).prepare().simulate(until=10, random_state=0)
    workers = trajectory.get_agents('workers', 10)
    assert np.all(workers.values['stale'] == 0)
    assert np.all(workers.values['accepted'] == trajectory.get_agents('employers', 10).values['offer'].max())


def test_undeclared_phase_raises():
    """A rule in a phase that the system does not declare raises an error."""
    with pytest.raises(SimulationError, match='offers'):
        _market().prepare().simulate(until=1)


@pytest.mark.skip(reason=NESTED_PHASES_GAP)
def test_run_reads_the_phases_of_a_nested_system():
    """Test that rules run in a phase that the solver of a nested system declares."""
    nested = _market('market', solver=MesaSolver(phases=[PhaseSpec('offers')]))
    trajectory = System(systems=[nested]).prepare().simulate(until=3, random_state=0)
    assert np.all(trajectory.get_agents('market/employers', 3).values['offer'] == 4.0)


def test_phases_keep_unphased_systems():
    """Rules without a phase run in the default phase."""

    def tick(agent, system):
        agent.count += 1

    counters = Element('counters', values={'count': 0.0}, size=3)
    system = System(
        elements=[counters],
        interactions=[_rule('tick', 'counters', tick)],
        solver=MesaSolver(phases=[PhaseSpec('extra', repeat=2)]),
    )
    trajectory = system.prepare().simulate(until=4, random_state=0)
    assert trajectory.get_agents('counters', 4).values['count'].tolist() == [4.0, 4.0, 4.0]


def test_sequential_and_by_type_order():
    """Sequential activation follows creation, and by-type activation follows the listed groups."""
    seen = []

    def system(phase):
        return System(
            elements=[Element('first', size=3), Element('second', size=3)],
            interactions=[
                _rule('one', 'first', lambda agent, system: seen.append(('first', agent.id)), phase='go'),
                _rule('two', 'second', lambda agent, system: seen.append(('second', agent.id)), phase='go'),
            ],
            solver=MesaSolver(phases=[phase]),
        )

    by_type = PhaseSpec('go', activation='by-type', groups=['second', 'first'])
    system(by_type).prepare().simulate(until=1, random_state=0)
    assert [group for group, _ in seen] == ['second'] * 3 + ['first'] * 3
    seen.clear()
    system(PhaseSpec('go', 'sequential')).prepare().simulate(until=1, random_state=0)
    assert [identity for _, identity in seen] == sorted(identity for _, identity in seen)


class AgentRule(Interaction):
    """Interaction whose rule changes every member of its target group at each step."""

    def __init__(self, label, *, target: Role, rule=None, phase=None):
        super().__init__(label)
        self.target = target
        self.rule = rule
        self.phase = phase

    def _prepare(self, context):
        """Set the rule of the members of the target group."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target, self.rule, phase=self.phase)]


class Branching(Interaction):
    """Galton-Watson branching, where every individual is replaced by a Poisson number of offspring."""

    def __init__(self, label, *, target: Role, mean=1.0):
        super().__init__(label)
        self.target = target
        self.mean = mean

    def _prepare(self, context):
        """Set the rule of reproduction."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target, self.reproduce)]

    def reproduce(self, agent, system):
        """Create the offspring of one individual, one generation later, and remove the individual."""
        for _ in range(system.rng.poisson(self.mean)):
            agent.create(generation=agent.generation + 1)
        agent.remove()


def _walk(agent, system):
    """Move a walker by a standard normal step."""
    agent.x += system.rng.normal()


def test_random_walk_of_agents_spreads_with_the_number_of_steps():
    """Test that walkers that each take a standard normal step reach a variance equal to the number of steps."""
    walkers = Element('walkers', values={'x': 0.0}, size=2000)
    system = System(elements=[walkers], interactions=[AgentRule('walk', target='walkers', rule=_walk)]).prepare()
    assert system.states_ == ('walkers.x',)
    assert system.form_ == FormKind.AGENT
    assert isinstance(system.solver_, MesaSolver)
    x = system.simulate(until=25, random_state=0).get_agents('walkers', 25).values['x']
    assert abs(x.mean()) < 0.5
    assert x.var() == pytest.approx(25.0, rel=0.15)


def test_galton_watson_process_grows_by_the_mean_number_of_offspring():
    """Test that generation `n` of a branching process has `n_0 m^n` members on average, all of generation `n`."""
    population = Element('population', values={'generation': 0}, size=20)
    system = System(elements=[population], interactions=[Branching('branching', target='population', mean=1.5)])
    system.prepare()
    sizes = []
    for seed in range(40):
        trajectory = system.simulate(until=3, random_state=seed)
        sizes.append(trajectory.get_values('population.count')[-1])
        assert set(trajectory.get_agents('population', 3).values['generation'].tolist()) <= {3}
    assert np.mean(sizes) == pytest.approx(20 * 1.5**3, rel=0.1)


def test_galton_watson_process_without_offspring_dies_out():
    """Test that a branching process whose individuals have no offspring is extinct after one generation."""
    population = Element('population', values={'generation': 0}, size=10)
    system = System(elements=[population], interactions=[Branching('branching', target='population', mean=0.0)])
    trajectory = system.prepare().simulate(until=2, random_state=0)
    assert trajectory.get_values('population.count').tolist() == [10.0, 0.0, 0.0]


def test_birth_and_death_are_rules_that_add_and_remove_members():
    """Test that cells that divide and share their mass, and cells that die, are rules of their group."""

    def divide(cell, system):
        cell.mass /= 2
        cell.create(mass=cell.mass)

    cells = Element('cells', values={'mass': 8.0}, size=1)
    division = System(elements=[cells], interactions=[AgentRule('division', target='cells', rule=divide)])
    trajectory = division.prepare().simulate(until=2, random_state=0)
    assert trajectory.get_agents('cells', 2).values['mass'].tolist() == [2.0, 2.0, 2.0, 2.0]
    death = System(elements=[cells], interactions=[AgentRule('death', target='cells', rule=lambda a, s: a.remove())])
    assert death.prepare().simulate(until=1).get_values('cells.count').tolist() == [1.0, 0.0]


def test_single_element_with_a_rule_is_one_agent():
    """Test that an element without a size whose interaction gives it a rule runs as a group of one agent."""

    def flip(coin, system):
        coin.heads += 1

    coin = Element('coin', values={'heads': 0})
    system = System(elements=[coin], interactions=[AgentRule('flip', target='coin', rule=flip)]).prepare()
    assert system.simulate(until=3).get_agents('coin', 3).values['heads'].tolist() == [3]


def test_activation_of_the_solver_orders_the_agents():
    """Test that the activation setting of the solver activates the agents in their order of creation."""
    seen = []
    walkers = Element('walkers', values={'x': 0.0}, size=5)
    rule = AgentRule('log', target='walkers', rule=lambda agent, system: seen.append(agent.id))
    solver = MesaSolver(activation='sequential')
    System(elements=[walkers], interactions=[rule], solver=solver).prepare().simulate(until=2, random_state=0)
    assert seen == sorted(seen[:5]) * 2


def test_phases_of_the_solver_run_in_order():
    """Test that rules in named phases of the solver run phase after phase, so a clerk reads the bids of the step."""

    def quote(trader, system):
        trader.bid = system.rng.uniform(0, 1)

    def settle(clerk, system):
        clerk.best = max(trader.bid for trader in system.get_agents('traders'))

    class Settle(AgentRule):
        def __init__(self, label, *, target: Role, traders: Role, rule=None, phase=None):
            super().__init__(label, target=target, rule=rule, phase=phase)
            self.traders = traders

    traders = Element('traders', values={'bid': 0.0}, size=4)
    clerk = Element('clerk', values={'best': 0.0})
    system = System(
        elements=[traders, clerk],
        interactions=[
            Settle('settle', target='clerk', traders='traders', rule=settle, phase='settle'),
            AgentRule('quote', target='traders', rule=quote, phase='quote'),
        ],
        solver=MesaSolver(phases=[PhaseSpec('quote'), PhaseSpec('settle')]),
    )
    trajectory = system.prepare().simulate(until=3, random_state=0)
    for frame in (1, 2, 3):
        best = trajectory.get_agents('clerk', frame).values['best'][0]
        assert best == trajectory.get_agents('traders', frame).values['bid'].max()


def test_unknown_activation_raises():
    """Test that an activation the solver does not know is named in the error."""
    walkers = Element('walkers', values={'x': 0.0}, size=2)
    system = System(
        elements=[walkers],
        interactions=[AgentRule('walk', target='walkers', rule=_walk)],
        solver=MesaSolver(activation='chaotic'),
    )
    with pytest.raises(ValidationError, match="activation of the Mesa solver is 'chaotic'"):
        system.prepare().simulate(until=1)


def test_rule_writing_a_value_the_agent_lacks_raises():
    """Test that a rule cannot give an agent a value its element does not have."""

    def tail(agent, system):
        agent.tails = 1

    coins = Element('coins', values={'heads': 0}, size=2)
    system = System(elements=[coins], interactions=[AgentRule('tail', target='coins', rule=tail)]).prepare()
    with pytest.raises(AttributeError, match='tails'):
        system.simulate(until=1)


def _step_right(agent, system):
    """Move an agent one unit along the first axis."""
    agent.x += 1


def test_agents_start_at_the_points_of_their_values_and_move_with_them():
    """Test that walkers on a plane start at their values `x` and `y` and move when a rule changes `x`."""
    plane = ContinuousSpace(extent=[50.0, 50.0])
    walkers = Element('walkers', values={'x': [22.7, 5.0], 'y': [1.5, 30.0]}, size=2, space=plane)
    walk = AgentRule('walk', target='walkers', rule=_step_right)
    trajectory = System(elements=[walkers], interactions=[walk]).prepare().simulate(until=2, random_state=0)
    for frame, shift in enumerate([0.0, 1.0, 2.0]):
        snapshot = trajectory.get_agents('walkers', frame)
        expected = [[22.7 + shift, 1.5], [5.0 + shift, 30.0]]
        np.testing.assert_allclose(snapshot.locations, expected)
        np.testing.assert_allclose(np.column_stack([snapshot.values['x'], snapshot.values['y']]), expected)


def test_agents_start_at_the_cells_of_their_values_and_wrap_around_a_ring():
    """Test that agents on a periodic lattice start at the cells their values name and wrap from the last cell."""
    walkers = Element('walkers', values={'x': [0, 3], 'y': [4, 2]}, size=2, space=LatticeSpace((5, 5)))
    walk = AgentRule('walk', target='walkers', rule=_step_right)
    trajectory = System(elements=[walkers], interactions=[walk]).prepare().simulate(until=2, random_state=0)
    assert trajectory.get_agents('walkers', 0).locations.tolist() == [[0, 4], [3, 2]]
    assert trajectory.get_agents('walkers', 2).locations.tolist() == [[2, 4], [0, 2]]
    assert trajectory.get_agents('walkers', 2).values['x'].tolist() == [2, 0]


def _drive(car, system):
    """Move a car by its speed in cells."""
    car.x += car.v


class Braking(Interaction):
    """Braking of every car of the road at once, an event drawn once per step for the whole road."""

    def __init__(self, label, *, p=1.0):
        super().__init__(label)
        self.p = p

    def _prepare(self, context):
        """Set the rule of the whole road."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context, self.brake)]

    def brake(self, system):
        """Slow every car by one cell per step with probability `p`."""
        if system.rng.random() < self.p:
            for car in system.get_agents('cars'):
                car.v = max(car.v - 1, 0)


@pytest.mark.parametrize(('p', 'moved'), [(1.0, [2, 3, 3]), (0.0, [2, 4, 6])])
def test_a_rule_of_the_whole_road_runs_once_per_step_after_the_cars(p, moved):
    """Test that a braking event of the whole road slows every car once per step, after the cars have moved."""
    cars = Element('cars', values={'x': [0, 3, 8], 'v': 2}, size=3, space=LatticeSpace(10))
    road = System(elements=[cars], interactions=[AgentRule('drive', target='cars', rule=_drive), Braking('road', p=p)])
    trajectory = road.prepare().simulate(until=3, random_state=0)
    for frame, distance in enumerate(moved, start=1):
        assert trajectory.get_agents('cars', frame).values['x'].tolist() == [(x + distance) % 10 for x in (0, 3, 8)]


class _AgentBraking(Braking):
    """Braking whose rule wrongly takes an agent and the system."""

    def brake(self, agent, system):
        """Slow one car."""
        agent.v = 0


def test_a_rule_of_the_whole_system_takes_the_system_alone():
    """Test that a rule of the whole system that takes an agent too raises an error that shows the right form."""
    cars = Element('cars', values={'x': [0], 'v': 1}, size=1, space=LatticeSpace(10))
    road = System(elements=[cars], interactions=[AgentRule('drive', target='cars', rule=_drive), _AgentBraking('road')])
    with pytest.raises(ValidationError, match=r'def brake\(system\)'):
        road.prepare()


class Arrive(Interaction):
    """A rule of the whole system that adds one member with a given age to its group every step."""

    def __init__(self, label: str, *, target: Role = 'people') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: object = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context, self.arrive)]

    def arrive(self, system: SimpleNamespace) -> None:
        """Add one member of age 0."""
        system.create('target', age=0.0)


@pytest.mark.skip(reason=RESERVED_KIND_GAP)
def test_created_agents_take_the_values_their_group_shares():
    """Test that a new agent of a plain element takes the shared values its creation leaves out."""
    people = Element('people', values={'age': 5.0, 'kind': 2.0}, size=1)
    trajectory = System(elements=[people], interactions=[Arrive('arrive')]).prepare().simulate(until=2)
    agents = trajectory.get_agents('people', 2)
    assert agents.values['age'].tolist() == [5.0, 0.0, 0.0]
    assert agents.values['kind'].tolist() == [2.0, 2.0, 2.0]


@pytest.mark.skip(reason=RESERVED_KIND_GAP)
def test_created_agents_need_the_values_their_group_gives_per_member():
    """Test that a new agent of a plain element needs the values its group gives one per member."""
    people = Element('people', values={'age': 5.0, 'kind': np.array([1.0, 2.0])}, size=2)
    system = System(elements=[people], interactions=[Arrive('arrive')]).prepare()
    with pytest.raises(SimulationError, match="needs a value of 'kind'"):
        system.simulate(until=1)


class PersonElement(Element):
    """A person whose class gives the default of its age and kind."""

    def __init__(
        self,
        label: str,
        *,
        age: object = 1.0,
        kind: object = 7.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.age = age
        self.kind = kind


@pytest.mark.skip(reason=RESERVED_KIND_GAP)
def test_created_agents_take_the_defaults_of_their_element_class():
    """Test that a new agent takes the class default of a value given per member that its creation leaves out."""
    people = PersonElement('people', age=np.array([5.0, 6.0]), kind=np.array([1.0, 2.0]), size=2)
    trajectory = System(elements=[people], interactions=[Arrive('arrive')]).prepare().simulate(until=1)
    agents = trajectory.get_agents('people', 1)
    assert agents.values['age'].tolist() == [5.0, 6.0, 0.0]
    assert agents.values['kind'].tolist() == [1.0, 2.0, 7.0]


@pytest.mark.skip(reason=RESERVED_KIND_GAP)
def test_created_agents_take_the_value_their_group_shares_over_the_class_default():
    """Test that a new agent takes a value its group shares rather than the default of its element class."""
    people = PersonElement('people', age=np.array([5.0, 6.0]), kind=3.0, size=2)
    trajectory = System(elements=[people], interactions=[Arrive('arrive')]).prepare().simulate(until=1)
    assert trajectory.get_agents('people', 1).values['kind'].tolist() == [3.0, 3.0, 3.0]


@pytest.mark.skip(reason=RESERVED_KIND_GAP)
def test_created_agents_of_a_plain_element_name_every_value_they_lack():
    """Test that creating an agent of a plain element without the values its group gives per member names them all."""
    people = Element('people', values={'kind': np.array([1.0, 2.0]), 'age': 5.0, 'height': np.ones(2)}, size=2)
    system = System(elements=[people], interactions=[Arrive('arrive')]).prepare()
    with pytest.raises(SimulationError, match="needs a value of 'kind', 'height', which its group gives per member"):
        system.simulate(until=1)
