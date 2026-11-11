"""Test agent-based systems on the Mesa backend: recognition, spaces, populations, reproducibility, and support."""

import json
from importlib.resources import files
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from shapely import Point, Polygon, box

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import (
    ContinuousSpace,
    DelaunayNeighborhood,
    DiscreteTime,
    LatticeSpace,
    NetworkSpace,
)
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile.passes._agents import AgentsPass
from skcomplex.core.engine.compile.passes._agents_problem import AGENT_RESERVED_NAMES
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import AgentView, MesaSolver
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import (
    BackendError,
    IncompatibleLawsError,
    LabelError,
    SimulationError,
    SpaceError,
    ValidationError,
)
from skcomplex.core.specs import (
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RoleSpec,
    StateSpec,
    StepSpec,
    VisualSpec,
)
from skcomplex.systems.biology.collective_behaviour import BoidsSystem
from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.systems.general.agents import MovementInteraction

from ....conftest import QueueElement, _queue_run

SEEN: list[object] = []
UNDECLARED: list[bool] = []
LOG: list[tuple[str, int]] = []
RECORDED_BOIDS = 'recorded_boids_two_dimensions.json'


class BreederElement(Element):
    """Agent that creates one offspring every step."""

    def _get_laws(self, s):
        """Return the breeding rule."""
        return [RuleLaw(self.breed)]

    def breed(self, agent, system):
        """Create one offspring."""
        agent.create()


class WalkerElement(Element):
    """Agent that moves to a random neighbouring cell and counts its moves."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        moves: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.moves = moves

    def _get_laws(self, s):
        """Return the walking rule."""
        return [RuleLaw(self.walk)]

    def walk(self, agent, system):
        """Move to a random neighbouring cell."""
        cells = agent.get_neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])
        agent.moves += 1


class ProbeElement(Element):
    """Agent that records how many neighbouring cells and neighbours it has."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        cells: Annotated[float, StateSpec()] = 0.0,
        near: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cells = cells
        self.near = near

    def _get_laws(self, s):
        """Return the probing rule."""
        return [RuleLaw(self.probe)]

    def probe(self, agent, system):
        """Record the number of neighbouring cells and of neighbours."""
        agent.cells = len(agent.get_neighbor_cells())
        agent.near = len(agent.get_neighbors())


class DrifterElement(Element):
    """Agent in a continuous space that counts the agents within a radius."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        near: Annotated[float, StateSpec()] = 0.0,
        radius: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.near = near
        self.radius = radius

    def _get_laws(self, s):
        """Return the counting rule."""
        return [RuleLaw(self.count)]

    def count(self, agent, system):
        """Count the agents within the radius."""
        agent.near = len(agent.get_neighbors(radius=agent.radius))


class CullerInteraction(Interaction):
    """System rule that removes the agent with the smallest id once."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the culling rule."""
        return [RuleLaw(self.cull)]

    def cull(self, system):
        """Remove the oldest agent while more than two remain."""
        agents = sorted(system.get_agents(self.target, AgentView), key=lambda agent: agent.id)
        if len(agents) > 2:
            agents[0].remove()


class MoneyAgentElement(Element):
    """Agent of the Boltzmann wealth system, which gives the transfer to a random agent when it has wealth."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 1.0,
        transfer: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth
        self.transfer = transfer

    def _get_laws(self, s):
        """Return the giving rule."""
        return [RuleLaw(self.give)]

    def give(self, agent, system):
        """Give the transfer to a random agent of the group."""
        if agent.wealth > 0:
            agents = system.get_agents(agent.group, AgentView)
            other = agents[int(system.rng.integers(len(agents)))]
            other.wealth += agent.transfer
            agent.wealth -= agent.transfer


def _money_system():
    """Return the Boltzmann wealth system as an ordinary system of 50 agents."""
    return CompositeSystem(
        'economy',
        elements=[GroupElement('agents', element=MoneyAgentElement, size=50)],
        time=DiscreteTime(),
    )


class NeighbourProbeElement(Element):
    """Test helper that records what a rule reads of the system and its neighbours."""

    seen: list[object]

    def __init__(self, label, visual=None, *, near: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.near = near

    def _get_laws(self, namespace):
        """Test helper laws."""
        return [RuleLaw(self.read)]

    def read(self, agent, system):
        """Test helper rule that keeps the dimensions and counts the neighbours within radii."""
        NeighbourProbeElement.seen.append(
            (agent.id, system.dimensions, len(agent.get_neighbors(2.0)), len(agent.get_neighbors(4.0))),
        )


class IdleElement(Element):
    """Test helper agent whose rule does nothing."""

    def _get_laws(self, namespace):
        """Test helper laws."""
        return [RuleLaw(self.idle)]

    def idle(self, agent, system):
        """Test helper rule."""


def _simulate(system, until=0, random_state=0):
    """Return the trajectory of a system on the Mesa solver."""
    return simulate(system, until=until, solver=MesaSolver(), random_state=random_state)


def _concave_polygon():
    """Return an L of three unit squares around a missing fourth, with a hole of area 0.1 in the first square."""
    hole = [(0.2, 0.2), (0.7, 0.2), (0.7, 0.4), (0.2, 0.4)]
    return Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], holes=[hole]), box(0.2, 0.2, 0.7, 0.4)


def _lengths(values):
    """Return the length of the queue of every agent."""
    return np.asarray([len(item) for item in values], dtype=float)


class EmployerElement(Element):
    """Test helper Employer."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        offer: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.offer = offer

    def _get_laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.post, phase='offers')]

    def post(self, agent, system):
        """Test helper post."""
        agent.offer = system.time + 1.0


class WorkerElement(Element):
    """Test helper Worker."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        accepted: Annotated[float, StateSpec()] = -1.0,
        stale: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.accepted = accepted
        self.stale = stale

    def _get_laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.accept, phase='acceptance')]

    def accept(self, agent, system):
        """Test helper accept."""
        best = max(employer.offer for employer in system.get_agents('employers', AgentView))
        agent.stale += float(best != system.time + 1.0)
        agent.accepted = best


def test_rule_laws_make_elements_agents():
    """Test that elements and groups with rule laws, and variable groups, run on the agent stepper."""

    class PlainElement(Element):
        """Plain element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

    plain = PlainElement
    assert AgentsPass._runs_rules(GroupElement('walkers', element=WalkerElement, size=3))
    assert AgentsPass._runs_rules(WalkerElement('walker'))
    assert AgentsPass._runs_rules(GroupElement('births', element=plain, size=2, variable=True))
    assert not AgentsPass._runs_rules(GroupElement('plain', element=plain, size=2))
    assert not AgentsPass._runs_rules(PopulationElement('herd'))


class StallElement(Element):
    """Passive agent without laws, whose stock shoppers buy."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stock: Annotated[float, StateSpec()] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stock = stock


class ShopperElement(Element):
    """Agent that buys one unit from the first stall that has stock."""

    def _get_laws(self, s):
        """Return the buying rule."""
        return [RuleLaw(self.buy)]

    def buy(self, agent, system):
        """Buy one unit from the first stall with stock."""
        stall = system.get_agents('stalls', AgentView)[0]
        if stall.stock > 0:
            stall.stock -= 1


def test_elements_without_laws_are_passive_agents():
    """Test that a group without laws next to agents with rules is a group of agents that the rules change."""
    system = CompositeSystem(
        elements=[
            GroupElement('stalls', element=StallElement, size=1),
            GroupElement('shoppers', element=ShopperElement, size=2),
        ],
    )
    trajectory = simulate(system, until=2, random_state=0)
    assert [trajectory.get_agents('stalls', frame).values['stock'].tolist() for frame in (1, 2)] == [[1.0], [0.0]]


def test_rules_give_agent_form():
    """Test that a system of agent groups with rules compiles to the agent-based form."""
    system = CompositeSystem(
        elements=[GroupElement('walkers', element=WalkerElement, size=3)],
        space=LatticeSpace((5, 5), boundary='fixed'),
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.form == FormKind.AGENT
    assert problem.agents[0].states == ('moves',)


def test_rules_mixed_with_rates_raise_error():
    """Test that rules and rates in one system raise the law kind error."""
    system = CompositeSystem(
        elements=[GroupElement('walkers', element=WalkerElement, size=3), PopulationElement('herd')],
        interactions=[ExponentialGrowthInteraction('growth', target='herd')],
        space=LatticeSpace((5, 5), boundary='fixed'),
    )
    with pytest.raises(IncompatibleLawsError, match='RateLaw: herd; RuleLaw: walkers'):
        PassCompiler().prepare(system).run()


def test_agents_are_born_every_step():
    """Test that one offspring per agent per step gives 10, 20, 40, and 80 agents."""
    trajectory = simulate(
        CompositeSystem(elements=[GroupElement('herd', element=BreederElement, size=10, variable=True)]),
        until=3,
    )
    assert trajectory.get_values('herd.count').tolist() == [10.0, 20.0, 40.0, 80.0]
    assert len(trajectory.get_agents('herd', 3).ids) == 80


def test_fixed_group_rejects_births():
    """Test that a rule creating a member of a group of fixed size raises an error that suggests a variable group."""
    with pytest.raises(SimulationError, match='variable=True'):
        simulate(CompositeSystem(elements=[GroupElement('herd', element=BreederElement, size=10)]), until=1)


def test_system_rules_remove_agents():
    """Test that an interaction rule removes agents and frames report the agents alive."""
    system = CompositeSystem(
        elements=[GroupElement('walkers', element=WalkerElement, size=5, variable=True)],
        interactions=[CullerInteraction('cull', target='walkers')],
        space=LatticeSpace((5, 5), boundary='fixed'),
    )
    trajectory = simulate(system, until=3, random_state=0)
    assert trajectory.get_values('walkers.count').tolist() == [5.0, 4.0, 3.0, 2.0]
    assert 1 not in trajectory.get_agents('walkers', 1).ids.tolist()


@pytest.mark.parametrize(('metric', 'expected'), [('chebyshev', 8), ('manhattan', 4)])
def test_grid_neighbours(metric, expected):
    """Test that an agent in the middle of a 10 by 10 grid has 8 Moore or 4 von Neumann neighbouring cells."""
    system = CompositeSystem(
        elements=[GroupElement('probe', element=ProbeElement, size=1, location=(5, 5))],
        space=LatticeSpace((10, 10), boundary='fixed', metric=metric),
    )
    snapshot = simulate(system, until=1).get_agents('probe', 1)
    assert snapshot.values['cells'].tolist() == [expected]


def test_network_neighbours_follow_edges():
    """Test that agents on a cycle network see their two neighbours."""
    system = CompositeSystem(
        elements=[GroupElement('nodes', element=ProbeElement, size=6)],
        space=NetworkSpace(nx.cycle_graph(6)),
    )
    snapshot = simulate(system, until=1).get_agents('nodes', 1)
    assert snapshot.values['near'].tolist() == [2.0] * 6
    assert sorted(snapshot.locations.tolist()) == list(range(6))


def test_continuous_neighbours_within_radius():
    """Test that agents in a continuous space find the agents within a radius."""
    system = CompositeSystem(
        elements=[
            GroupElement('drifters', element=DrifterElement, size=3, location=[(1.0, 1.0), (1.5, 1.0), (8.0, 8.0)]),
        ],
        space=ContinuousSpace(extent=(10.0, 10.0)),
    )
    snapshot = simulate(system, until=1).get_agents('drifters', 1)
    assert snapshot.values['near'].tolist() == [1.0, 1.0, 0.0]


class TriangulatedElement(DrifterElement):
    """Agent in a continuous space that counts its Delaunay neighbours."""

    def count(self, agent, system):
        """Count the agents joined to the agent by an edge of the Delaunay triangulation."""
        agent.near = len(agent.get_neighbors(over=DelaunayNeighborhood()))


def test_continuous_neighbours_of_a_delaunay_triangulation():
    """Test that the centre of a square of agents has four Delaunay neighbours and every corner three."""
    corners = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0), (4.0, 4.0)]
    system = CompositeSystem(
        elements=[GroupElement('cells', element=TriangulatedElement, size=5, location=corners)],
        space=ContinuousSpace(extent=(10.0, 10.0)),
    )
    snapshot = simulate(system, until=1).get_agents('cells', 1)
    assert snapshot.values['near'].tolist() == [3.0, 3.0, 3.0, 3.0, 4.0]


def test_agent_runs_are_reproducible():
    """Test that equal random states give equal agent trajectories."""
    runs = [
        simulate(
            CompositeSystem(
                elements=[GroupElement('walkers', element=WalkerElement, size=5)],
                space=LatticeSpace((10, 10), boundary='fixed'),
            ),
            until=5,
            random_state=0,
        )
        for _ in range(2)
    ]
    first, second = (run.get_agents('walkers', 5) for run in runs)
    assert np.array_equal(first.locations, second.locations)
    assert first.values['moves'].tolist() == [5.0] * 5


def test_agent_frames_use_integer_steps():
    """Test that agent-based systems reject a non-integer frame duration."""
    with pytest.raises(ValidationError, match='positive integer'):
        simulate(
            CompositeSystem(elements=[GroupElement('herd', element=BreederElement, size=2, variable=True)]),
            until=1,
            dt=0.5,
        )


def test_boltzmann_wealth_keeps_wealth():
    """Test that the Boltzmann wealth system as an ordinary system keeps its total wealth and its agents' ids."""
    trajectory = simulate(_money_system(), until=100, random_state=0)
    totals = {int(trajectory.get_agents('agents', frame).values['wealth'].sum()) for frame in range(101)}
    assert totals == {50}
    assert trajectory.get_agents('agents', 5).ids.tolist() == list(range(1, 51))


def test_agent_parameter_changes_live():
    """Test that a parameter update reaches the rules of the agents."""
    schedule = {'agents.transfer': lambda t: 1.0 if t < 5 else 0.0}
    trajectory = simulate(_money_system(), until=10, schedule=schedule, random_state=0)
    frames = [trajectory.get_agents('agents', frame).values['wealth'] for frame in (7, 10)]
    assert np.array_equal(*frames)


def test_mesa_declares_its_support():
    """Test that the Mesa backend declares discrete time and rule laws."""
    supports = MesaSolver._get_support()
    assert supports.time == {DiscreteTime}
    assert not supports.get_missing('laws', {RuleLaw})


def test_mesa_validates_systems_against_its_support():
    """Test that validation reports a system of rate laws and continuous time for the Mesa backend."""
    system = CompositeSystem(
        elements=[PopulationElement('herd')],
        interactions=[ExponentialGrowthInteraction('growth', target='herd')],
    )
    issues = PassCompiler(solver=MesaSolver()).prepare(system).report_.issues
    assert [issue.code for issue in issues] == ['solver-unsupported']
    assert 'RateLaw' in issues[0].message
    assert PassCompiler(solver=MesaSolver()).prepare(_money_system()).report_.ok


class CattleElement(Element):
    """Agent that stays where it is and that a census interaction counts."""

    def _get_laws(self, s):
        """Return the rule of staying."""
        return [RuleLaw(self.stay)]

    def stay(self, agent, system):
        """Stay in place."""


class CensusInteraction(Interaction):
    """Interaction whose rule records its parameter, the groups of its roles, and the labels of their agents."""

    def __init__(
        self,
        label: str,
        *,
        counted: Annotated[str, RoleSpec()],
        others: Annotated[str, RoleSpec()],
        threshold: Annotated[float, ParameterSpec()] = 0.9,
        looked_up: str = 'counted',
    ) -> None:
        super().__init__(label)
        self.counted = counted
        self.others = others
        self.threshold = threshold
        self.looked_up = looked_up

    def _get_laws(self, s):
        """Return the census rule."""
        return [RuleLaw(self.record)]

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
    farm = CompositeSystem(
        'farm',
        elements=[
            GroupElement('cows', element=CattleElement, size=42),
            GroupElement('patches', element=CattleElement, size=43),
        ],
        interactions=[CensusInteraction('census', counted='cows', others='patches', looked_up=looked_up)],
    )
    system = CompositeSystem(systems=[farm], space=space)
    trajectory = simulate(system, until=3, schedule=schedule, random_state=0)
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


def _views(space, locations):
    """Return views of two agents placed at given locations of a space."""
    system = CompositeSystem(
        elements=[GroupElement('cows', element=CattleElement, size=2, location=locations)],
        space=space,
    )
    stepper = _stepped(system, until=0)
    return [AgentView(agent, stepper) for agent in stepper._get_group_agents('cows')]


def test_displacement_across_torus_seam():
    """Test that the displacement in a toroidal continuous space is the shortest image across the seam."""
    first, second = _views(ContinuousSpace(extent=(100.0, 1.0), boundary='periodic'), [(99.0, 0.5), (1.0, 0.5)])
    assert first.get_displacement_to(second).tolist() == [2.0, 0.0]
    assert first.get_distance_to(second) == 2.0
    assert first.get_distance_to((97.0, 0.5)) == 2.0


def test_location_key_matches_the_cells_around_an_agent():
    """Test that the key of a grid location equals the neighbouring cell another agent stands on."""
    first, second = _views(LatticeSpace((5, 5), metric='chebyshev', boundary='fixed'), [(1, 1), (2, 2)])
    assert first.location_key == (1, 1)
    assert second.location_key in first.get_neighbor_cells()


def test_location_key_is_hashable_in_every_space():
    """Test that the key of a position is a tuple of floats and the key of a network node is the node."""
    first, _ = _views(ContinuousSpace(extent=(10.0, 10.0)), [(1.5, 2.0), (3.0, 4.0)])
    assert first.location_key == (1.5, 2.0)
    assert {first.location_key}
    node, _ = _views(NetworkSpace(nx.path_graph(5)), [0, 3])
    assert node.location_key == 0


def test_network_distance_counts_hops():
    """Test that the distance on a network counts hops and that a displacement raises a space error."""
    first, second = _views(NetworkSpace(nx.path_graph(5)), [0, 3])
    assert first.get_distance_to(second) == 3.0
    with pytest.raises(SpaceError, match='network'):
        first.get_displacement_to(second)


def test_agent_view_names_are_reserved():
    """Test that every public name of the agent view is reserved for the states of agents."""
    public = {name for name in dir(AgentView) if not name.startswith('_')}
    assert public <= AGENT_RESERVED_NAMES


def test_agent_state_with_reserved_name_raises_error():
    """Test that an agent state named like an attribute of the agent view raises a label error at compilation."""

    class MoverElement(Element):
        """Mover element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            alive: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.alive = alive

        def _get_laws(self, s):
            """Return the laws of the mover element."""
            return [RuleLaw(lambda agent, system: None)]

    with pytest.raises(LabelError, match='declare the variables alive'):
        PassCompiler().prepare(CompositeSystem(elements=[GroupElement('movers', element=MoverElement, size=2)])).run()


def test_non_agent_state_may_take_agent_view_name():
    """Test that an element without rule laws keeps states named like attributes of the agent view."""

    class CellElement(Element):
        """Cell element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            alive: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.alive = alive

    system = CompositeSystem(elements=[GroupElement('cells', element=CellElement, size=2)])
    assert PassCompiler().prepare(system).run().problem_.states.names == [
        'cells.alive',
    ]


class ClockedElement(Element):
    """Agent that records the system time at every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        seen: Annotated[float, StateSpec()] = -1.0,
        n: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.seen = seen
        self.n = n

    def _get_laws(self, s):
        """Return the recording rule."""
        return [RuleLaw(self.record)]

    def record(self, agent, system):
        """Store the system time."""
        agent.seen = system.time


def test_rules_read_system_time():
    """Test that rules read the system time k during the step that produces frame k."""
    trajectory = simulate(CompositeSystem(elements=[GroupElement('clocks', element=ClockedElement, size=2)]), until=3)
    assert [trajectory.get_agents('clocks', step).values['seen'].tolist() for step in (1, 2, 3)] == [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]


def test_one_agent_takes_one_entry_values():
    """Test that a group of one agent reads a one-entry location and a one-entry value as its own."""
    system = CompositeSystem(
        elements=[GroupElement('one', element=ClockedElement, size=1, values={'n': [3.0]}, location=[(1, 1)])],
        space=LatticeSpace((5, 5), boundary='fixed'),
    )
    snapshot = simulate(system, until=0).get_agents('one', 0)
    assert snapshot.locations.tolist() == [[1, 1]]
    assert snapshot.values['n'].tolist() == [3.0]


def test_mesa_solver_run_sets_agents_and_snapshot():
    """Test that every run of the Mesa solver sets the alive agents by group and the snapshot of the step."""
    system = CompositeSystem(elements=[GroupElement('breeders', element=BreederElement, size=2, variable=True)])
    problem = PassCompiler().prepare(system).run().problem_
    solver = MesaSolver().set_params(random_state=np.random.default_rng(0), until=3.0).prepare(problem)
    assert len(solver.agents_['breeders']) == 2
    solver.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0))
    assert len(solver.agents_['breeders']) == 8
    assert solver.snapshot_.reporters['breeders.count'] == 8.0
    for name in ('agents_of', 'snapshot', 'create_agent', 'resolve_group', 'spec_of', 'group_agents'):
        assert not hasattr(solver, name)


class ForagerElement(Element):
    """Agent that gathers one unit of food every step and records whether it can read an undeclared state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        food: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.food = food

    def _get_laws(self, s):
        """Return the foraging rule."""
        return [RuleLaw(self.forage)]

    def forage(self, agent, system):
        """Gather one unit of food and try to read an undeclared state."""
        agent.food += 1
        try:
            _ = agent.wealth
        except AttributeError:
            UNDECLARED.append(True)


def test_agent_views_read_and_set_states_as_properties():
    """Test that a rule sets a state through its view and that an undeclared state raises Python's own error."""
    UNDECLARED.clear()
    system = CompositeSystem(elements=[GroupElement('foragers', element=ForagerElement, size=2)], time=DiscreteTime())
    trajectory = simulate(system, until=3, dt=1.0, random_state=0)
    assert trajectory.get_agents('foragers', 3).values['food'].tolist() == [3.0, 3.0]
    assert UNDECLARED
    assert all(UNDECLARED)


class SaverElement(Element):
    """Test helper Saver."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        money: Annotated[float, StateSpec(lags=3)] = 0.0,
        seen: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.money = money
        self.seen = seen

    def _get_laws(self, s):
        """Test helper laws."""
        return [RuleLaw(self.save)]

    def save(self, agent, system):
        """Test helper save."""
        agent.seen = float(len(agent.get_lags('money')))
        agent.money += 1


def test_memory_window():
    """Agents remember the last values of the states their type lists."""
    trajectory = simulate(
        CompositeSystem(elements=[GroupElement('savers', element=SaverElement, size=2)]),
        until=5,
        random_state=0,
    )
    assert trajectory.get_agents('savers', 5).values['seen'].tolist() == [3.0, 3.0]


class AccountElement(Element):
    """Agent that names a member of the group `stalls` and counts the stock of that member."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stall: Annotated[float, StateSpec(reference='stalls')] = -1.0,
        seen: Annotated[float, StateSpec()] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stall = stall
        self.seen = seen

    def _get_laws(self, s):
        """Return the reading rule."""
        return [RuleLaw(self.read)]

    def read(self, agent, system):
        """Record the stock of the member the reference names, or -1 for none."""
        stall = agent.get_referenced('stall')
        agent.seen = -1.0 if stall is None else stall.stock


def test_reference_states_name_members_of_another_group_by_id():
    """Test that initial positions in the referenced group become ids that `referenced` resolves to the members."""
    system = CompositeSystem(
        elements=[
            GroupElement('stalls', element=StallElement, size=2, values={'stock': [5.0, 7.0]}),
            GroupElement('accounts', element=AccountElement, size=3, values={'stall': [1, -1, 0]}),
        ],
    )
    snapshot = simulate(system, until=1, random_state=0).get_agents('accounts', 1)
    assert snapshot.values['seen'].tolist() == [7.0, -1.0, 5.0]
    assert snapshot.values['stall'].tolist() == [2.0, -1.0, 1.0]


def test_reference_to_a_group_the_system_lacks_names_no_member():
    """Test that a reference state names no member when the system has no referenced group."""
    system = CompositeSystem(elements=[GroupElement('accounts', element=AccountElement, size=2)])
    snapshot = simulate(system, until=1, random_state=0).get_agents('accounts', 1)
    assert snapshot.values['seen'].tolist() == [-1.0, -1.0]


def test_reference_beyond_the_referenced_group_raises():
    """Test that an initial position beyond the members of the referenced group raises an error."""
    system = CompositeSystem(
        elements=[
            GroupElement('stalls', element=StallElement, size=1),
            GroupElement('accounts', element=AccountElement, size=1, values={'stall': [3]}),
        ],
    )
    with pytest.raises(SimulationError, match='position 3'):
        simulate(system, until=1, random_state=0)


def test_memory_of_unremembered_state_raises():
    """Reading the memory of a state the type does not remember raises an error."""

    class ForgetfulElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            money: Annotated[float, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.money = money

        def _get_laws(self, s):
            return [RuleLaw(lambda agent, system: agent.get_lags('money'))]

    with pytest.raises(KeyError, match='remember'):
        simulate(CompositeSystem(elements=[GroupElement('agents', element=ForgetfulElement, size=1)]), until=1)


class TickerElement(Element):
    """Agent that logs its id in its own rule."""

    def _get_laws(self, s):
        """Return the logging rule."""
        return [RuleLaw(self.tick)]

    def tick(self, agent, system):
        """Log the agent with the element tag."""
        LOG.append(('element', agent.id))


class EchoInteraction(Interaction):
    """Interaction whose members log their id in their own activation, with the gain of the interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gain: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gain = gain

    def _get_laws(self, s):
        """Return the member rule."""
        return [RuleLaw(self.echo, members='target')]

    def echo(self, agent, system):
        """Log the agent with the gain of the interaction."""
        LOG.append((f'member {system.get_parameter("gain")}', agent.id))


def test_member_rules_run_in_the_activation_of_every_member_after_its_element_rules():
    """Test that a member rule of an interaction runs right after the rules of each agent, with its own view."""
    LOG.clear()
    group = GroupElement('tickers', element=TickerElement, size=5)
    system = CompositeSystem(
        elements=[group],
        interactions=[EchoInteraction('echo', target='tickers')],
        time=DiscreteTime(),
    )
    simulate(system, until=2, random_state=0)
    assert len(LOG) == 20
    for first, second in zip(LOG[::2], LOG[1::2], strict=True):
        assert (first[0], second) == ('element', ('member 2.0', first[1]))


class VisitedView(AgentView):
    """View of a visited agent, with its visits, its trait, and the weight of its group."""

    visits: float
    trait: float
    weight: float


class VisitedElement(Element):
    """Agent that counts its visits, records their times, and flips its trait at every visit."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        visits: Annotated[float, StateSpec()] = 0.0,
        trait: Annotated[float, StateSpec()] = 0.0,
        weight: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.visits = visits
        self.trait = trait
        self.weight = weight

    @classmethod
    def _get_view(cls) -> type[VisitedView]:
        """Return the view of a visited agent."""
        return VisitedView

    def _get_laws(self, s):
        """Return the visiting rule."""
        return [RuleLaw(self.visit, phase='visit')]

    def visit(self, agent, system):
        """Count the visit, record its time and the counts of the traits, and flip the trait."""
        agent.visits += 1
        agent.trait = 1.0 - agent.trait
        traits = [item.trait for item in system.get_agents('visited')]
        system.storage.setdefault('times', []).append(system.time)
        system.storage.setdefault('compositions', []).append(
            (system.get_composition('visited', 'trait'), {value: traits.count(value) for value in set(traits)}),
        )


def _stepped(
    system: CompositeSystem,
    until: int,
    random_state: int = 0,
    observables: tuple[ObservableSpec, ...] = (),
) -> MesaSolver:
    """Return the agent-based solver after it stepped a system one unit of time at a time until a time."""
    problem = simulate(system, until=0, random_state=random_state, observables=observables).problem
    solver = MesaSolver(random_state=random_state, until=until).prepare(problem)
    state, parameters = solver.initial_state_, problem.initial_parameters
    for time in range(until):
        state = solver.run(StepSpec(time, state, parameters, time + 1), final=time + 1 == until).state_
    return solver


def _visited_system(visits: float) -> CompositeSystem:
    """Return a group of four visited agents whose phase visits them with replacement."""
    return CompositeSystem(
        elements=[GroupElement('visited', element=VisitedElement, size=4, values={'weight': 2.0})],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('visit', activation='visits', visits=visits)])),
    )


def test_visits_draw_agents_with_replacement():
    """Test that a phase of visits activates agents drawn with replacement, visits times the size of the group."""
    trajectory = simulate(_visited_system(2.5), until=4, random_state=0)
    visits = trajectory.get_agents('visited', 4).values['visits']
    assert visits.sum() == 40
    assert len(set(visits.tolist())) > 1


def test_visits_advance_the_time_by_a_fraction_of_the_step():
    """Test that rules read the time `k + i / n` at visit `i` of `n` in step `k`."""
    solver = _stepped(_visited_system(1.0), until=2)
    assert solver.storage_['visited']['times'] == [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]


def test_composition_counts_the_values_of_a_state():
    """Test that the composition of a group keeps the counts of a state as agents change it."""
    compositions = _stepped(_visited_system(3.0), until=2, random_state=1).storage_['visited']['compositions']
    assert len(compositions) == 24
    assert all(kept == counted for kept, counted in compositions)


def test_periodic_phase_runs_every_period_steps_from_its_offset():
    """Test that a phase of period three and offset one runs in the steps 1, 4, and 7 only."""
    phase = PhaseSpec('visit', activation='sequential', period=3, offset=1)
    system = CompositeSystem(
        elements=[GroupElement('visited', element=VisitedElement, size=1)],
        time=DiscreteTime(schedule=PhasesScheduleSpec([phase])),
    )
    assert _stepped(system, until=8).storage_['visited']['times'] == [1.0, 4.0, 7.0]


class OccupancyReaderInteraction(Interaction):
    """Interaction that stores, at every step, the counts of the walkers by cell."""

    def _get_laws(self, s):
        """Return the reading rule."""
        return [RuleLaw(self.read)]

    def read(self, system):
        """Store the counts of the walkers by cell."""
        system.storage.setdefault('occupancies', []).append(system.get_occupancy('walkers'))


def test_occupancy_counts_the_agents_of_a_group_by_cell():
    """Test that the occupancy of a group counts its agents in every cell that holds one, by the key of the cell."""
    walkers = GroupElement('walkers', element=WalkerElement, size=3, location=[(0, 0), (0, 0), (2, 1)])
    system = CompositeSystem(
        elements=[walkers],
        interactions=[OccupancyReaderInteraction('reader')],
        space=LatticeSpace((4, 4), boundary='periodic'),
    )
    solver = _stepped(system, until=2)
    occupancies = solver.storage_['reader']['occupancies']
    locations = solver.snapshot_.groups['walkers'].locations.tolist()
    expected = {}
    for location in locations:
        expected[tuple(location)] = expected.get(tuple(location), 0) + 1
    assert occupancies[-1] == expected
    assert sum(occupancies[0].values()) == 3


class TotalReaderInteraction(Interaction):
    """Interaction that stores, at every step, the history of the observable it reads."""

    def _get_laws(self, s):
        """Return the reading rule."""
        return [RuleLaw(self.read, phase='visit')]

    def read(self, system):
        """Store the history of the total of the visits."""
        system.storage.setdefault('histories', []).append(system.get_history('total').tolist())


def test_history_reads_an_observable_at_earlier_steps():
    """Test that a rule reads the values of an observable of the run at the end of every earlier step."""
    system = _visited_system(1.0)
    system.interactions = [TotalReaderInteraction('reader')]
    observables = [ObservableSpec('total', group='visited', variable='visits', reduce='sum')]
    solver = _stepped(system, until=3, observables=tuple(observables))
    assert solver.storage_['reader']['histories'] == [[], [4.0], [4.0, 8.0]]


def _weight_of(agent: VisitedView) -> float:
    """Return the weight of an agent, a parameter of its group."""
    return agent.weight


def test_observables_read_agent_views():
    """Test that an observable of an agent group reads the views of its agents, with the parameters of the group."""
    observables = [ObservableSpec('weights', group='visited', expression=_weight_of, reduce='sum')]
    trajectory = simulate(_visited_system(1.0), until=1, observables=observables)
    assert trajectory.get_values('weights').tolist() == [8.0, 8.0]


class HolderElement(Element):
    """Agent on a node that counts the tokens on the nodes around it, and the agents of every group there."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        around: Annotated[float, StateSpec()] = 0.0,
        every: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.around = around
        self.every = every

    def _get_laws(self, s):
        """Return the counting rule."""
        return [RuleLaw(self.hold)]

    def hold(self, agent, system):
        """Count the tokens of the neighbouring nodes, and the agents of every group there."""
        agent.around = len(agent.get_neighbors(group='tokens'))
        agent.every = len(agent.get_neighbors())


def test_network_neighbours_of_one_group_read_the_member_sets_of_the_nodes_around():
    """Test that a holder on a path reads the tokens on its neighbouring nodes apart from the holders there."""
    system = CompositeSystem(
        elements=[
            GroupElement('holders', element=HolderElement, size=4, location=[0, 1, 2, 3]),
            GroupElement('tokens', element=Element, size=6, variable=True, location=[0, 0, 1, 2, 3, 3]),
        ],
        space=NetworkSpace(nx.path_graph(4)),
    )
    snapshot = simulate(system, until=1, random_state=0).get_agents('holders', 1)
    places = snapshot.locations.tolist()
    assert dict(zip(places, snapshot.values['around'].tolist(), strict=True)) == {0: 1.0, 1: 3.0, 2: 3.0, 3: 1.0}
    assert dict(zip(places, snapshot.values['every'].tolist(), strict=True)) == {0: 2.0, 1: 5.0, 2: 5.0, 3: 2.0}


class CoinElement(Element):
    """Agent that draws a named uniform value and the value its generator would give."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        drawn: Annotated[float, StateSpec()] = 0.0,
        expected: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.drawn = drawn
        self.expected = expected

    def _get_laws(self, s):
        """Return the drawing rule."""
        return [RuleLaw(self.toss)]

    def toss(self, agent, system):
        """Draw a named uniform value and the next value of a copy of the generator."""
        copy = np.random.Generator(type(system.rng.bit_generator)())
        copy.bit_generator.state = system.rng.bit_generator.state
        agent.expected = 2 + 3 * copy.random()
        agent.drawn = system.draw('toss', 'uniform', low=2, high=5)


def test_rule_draw_samples_from_the_generator_of_the_system():
    """Test that the named draw of a rule samples from the generator of the system and is recorded."""
    group = GroupElement('coins', element=CoinElement, size=5)
    system = CompositeSystem(elements=[group], time=DiscreteTime())
    trajectory = simulate(system, until=3, random_state=0, record_draws=True)
    agents = trajectory.get_agents('coins', 3).values
    np.testing.assert_allclose(agents['drawn'], agents['expected'])
    record = trajectory.draws.draws['coins.toss']
    assert record.values.size == 15
    assert np.all(record.arguments['low'] == 2)


def test_rule_draw_rejects_an_unknown_distribution():
    """Test that a named draw of an unknown distribution raises a named error."""

    class OddElement(CoinElement):
        def toss(self, agent, system):
            system.draw('toss', 'wobbly', a=1)

    system = CompositeSystem(elements=[GroupElement('coins', element=OddElement, size=2)], time=DiscreteTime())
    with pytest.raises(ValidationError, match="distribution 'wobbly'"):
        simulate(system, until=1, random_state=0)


def test_recorded_two_dimensional_trajectories_load():
    """Test that the trajectories recorded before three-dimensional spaces load with their frames and shapes."""
    recorded = json.loads((files(__package__) / RECORDED_BOIDS).read_text())
    assert sorted(int(frame) for frame in recorded) == [0, 1, 5, 10, 20]
    assert np.asarray(recorded['20']['locations']).shape == (10, 2)


def test_two_dimensional_space_keeps_recorded_trajectories():
    """Test that a system on a two-dimensional continuous space repeats the trajectories recorded before the change."""
    recorded = json.loads((files(__package__) / RECORDED_BOIDS).read_text())
    for _ in range(2):
        trajectory = simulate(BoidsSystem(10).prepare().system_, until=20, dt=1, random_state=0)
        for frame, values in recorded.items():
            snapshot = trajectory.get_agents('flock', int(frame))
            assert snapshot.ids.tolist() == values['ids']
            np.testing.assert_allclose(snapshot.locations, values['locations'], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(snapshot.values['vx'], values['vx'], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(snapshot.values['vy'], values['vy'], rtol=1e-9, atol=1e-9)


def test_three_dimensional_box_places_agents_inside():
    """Test that the system view reports three dimensions, and agents without a location start inside the box."""
    NeighbourProbeElement.seen = []
    group = GroupElement('probes', element=NeighbourProbeElement, size=200)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 5.0, 2.0)))
    locations = _simulate(system, until=1).get_agents('probes', 0).locations
    assert NeighbourProbeElement.seen[0][1] == (10.0, 5.0, 2.0)
    assert locations.shape == (200, 3)
    assert locations.min() >= 0.0
    assert np.all(locations.max(axis=0) <= (10.0, 5.0, 2.0))
    assert locations[:, 2].max() > 1.0


def test_neighbours_across_the_depth_boundary():
    """Test that two agents across the periodic third side find each other within a radius of 0.5."""

    class NearElement(NeighbourProbeElement):
        """Test helper that counts the neighbours within 0.5."""

        def read(self, agent, system):
            """Test helper rule."""
            agent.near = float(len(agent.get_neighbors(0.5)))

    group = GroupElement('probes', element=NearElement, size=2, location=[(5.0, 5.0, 0.2), (5.0, 5.0, 9.9)])
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic'))
    assert _simulate(system, until=1).get_agents('probes', 1).values['near'].tolist() == [1.0, 1.0]


def test_displacement_takes_the_minimum_image_in_depth():
    """Test that the displacement between agents across the third side wraps every coordinate."""

    class ShiftElement(NeighbourProbeElement):
        """Test helper that keeps the depth of the shortest vector to the other agent."""

        def read(self, agent, system):
            """Test helper rule."""
            other = next(item for item in system.get_agents('probes') if item.id != agent.id)
            agent.near = float(agent.get_displacement_to(other)[2])

    group = GroupElement('probes', element=ShiftElement, size=2, location=[(5.0, 5.0, 0.2), (5.0, 5.0, 9.9)])
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic'))
    near = _simulate(system, until=1).get_agents('probes', 1).values['near']
    np.testing.assert_allclose(near, [-0.3, 0.3])


def test_depth_counts_in_the_distance():
    """Test that agents that differ by 3 in the third coordinate are neighbours within 4 and not within 2."""
    NeighbourProbeElement.seen = []
    group = GroupElement('probes', element=NeighbourProbeElement, size=2, location=[(5.0, 5.0, 1.0), (5.0, 5.0, 4.0)])
    _simulate(CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0, 10.0))), until=1)
    assert [item[2:] for item in NeighbourProbeElement.seen] == [(0, 1), (0, 1)]


def test_gaussian_steps_spread_in_depth():
    """Test that the variance of the third coordinate grows under Gaussian steps and positions stay in the box."""
    group = GroupElement('walkers', element=Element, size=100, location=(5.0, 5.0, 5.0))
    walk = MovementInteraction('walk', target='walkers', step='gaussian', scale=0.1)
    system = CompositeSystem(
        elements=[group],
        interactions=[walk],
        space=ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic'),
    )
    trajectory = _simulate(system, until=100)
    variances = [trajectory.get_agents('walkers', frame).locations[:, 2].var() for frame in (0, 10, 100)]
    assert variances[0] < variances[1] < variances[2]
    for frame in (1, 50, 100):
        locations = trajectory.get_agents('walkers', frame).locations
        assert locations.min() >= 0.0
        assert locations.max() <= 10.0


def test_trajectory_reports_three_coordinates_per_agent():
    """Test that a trajectory holds one row of 3 coordinates per agent alive at frame 5, with the agent ids."""
    group = GroupElement('walkers', element=Element, size=7)
    walk = MovementInteraction('walk', target='walkers', step='gaussian', scale=0.1)
    system = CompositeSystem(
        elements=[group],
        interactions=[walk],
        space=ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic'),
    )
    snapshot = _simulate(system, until=5).get_agents('walkers', 5)
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
    group = GroupElement('probes', element=NeighbourProbeElement, size=2)
    problem = PassCompiler(solver=MesaSolver()).prepare(CompositeSystem(elements=[group], space=space)).run().problem_
    with pytest.raises(BackendError, match=f"continuous space 'space'.*{dimension}.*two or three dimensions"):
        MesaSolver().prepare(problem)
    with pytest.raises(BackendError, match='two or three dimensions'):
        _simulate(CompositeSystem(elements=[group], space=space), until=1)


@pytest.mark.parametrize(
    'location',
    [(1.0, 2.0), [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)], box(0.0, 0.0, 1.0, 1.0)],
)
def test_location_of_the_wrong_length_is_reported(location):
    """Test that the compile report holds an initial location that does not fit a three-dimensional space."""
    group = GroupElement('probes', element=NeighbourProbeElement, size=3, location=location)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
    report = PassCompiler().prepare(system).report_
    issue = next(item for item in report.issues if item.code == 'location-dimension')
    assert issue.path == 'probes'
    assert '2 coordinates' in issue.message
    assert '3 dimensions' in issue.message


def test_locations_that_fit_their_space_are_not_reported():
    """Test that fitting positions and polygons in a two-dimensional space leave the compile report clean."""
    for location in ((1.0, 2.0, 3.0), [(1.0, 2.0, 3.0)] * 2):
        group = GroupElement('probes', element=NeighbourProbeElement, size=2, location=location)
        system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
        assert PassCompiler().prepare(system).report_.ok
    group = GroupElement('probes', element=NeighbourProbeElement, size=2, location=[box(0.0, 0.0, 1.0, 1.0)] * 2)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
    assert PassCompiler().prepare(system).report_.ok


def test_move_to_of_the_wrong_length_raises_space_error():
    """Test that a move to a location that does not fit the space raises a named error during the run."""
    group = GroupElement('walkers', element=Element, size=1, location=(1.0, 1.0, 1.0))
    walk = MovementInteraction('walk', target='walkers', step=lambda agent, system: (2.0, 2.0))
    system = CompositeSystem(elements=[group], interactions=[walk], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
    with pytest.raises(SpaceError, match='has 2 coordinates, and the space has 3 dimensions'):
        _simulate(system, until=1)


def test_agents_per_census_region():
    """Test that members start inside the polygons given as their locations."""
    regions = [box(0.0, 0.0, 4.0, 4.0), box(5.0, 5.0, 9.0, 9.0), box(0.0, 5.0, 4.0, 9.0)]
    group = GroupElement('people', element=IdleElement, size=150, location=[regions[0]] * 100 + [regions[1]] * 50)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0), regions={'census': regions}))
    snapshot = _simulate(system).get_agents('people', 0)
    assert all(regions[0].contains(Point(point)) for point in snapshot.locations[:100])
    assert all(regions[1].contains(Point(point)) for point in snapshot.locations[100:])


def test_one_polygon_is_shared_by_every_member():
    """Test that one polygon as the location of a group places every member at its own point inside it."""
    region = box(1.0, 1.0, 2.0, 2.0)
    group = GroupElement('people', element=IdleElement, size=5, location=region)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
    locations = _simulate(system).get_agents('people', 0).locations
    assert all(region.contains(Point(point)) for point in locations)
    assert len({tuple(point) for point in locations}) == 5


def test_region_placement_is_reproducible():
    """Test that two runs with one random state start the agents at the same points, and another seed differs."""

    def start(random_state):
        group = GroupElement('people', element=IdleElement, size=20, location=box(0.0, 0.0, 4.0, 4.0))
        system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
        return _simulate(system, random_state=random_state).get_agents('people', 0).locations

    np.testing.assert_array_equal(start(0), start(0))
    assert not np.array_equal(start(0), start(1))


def test_placement_is_uniform_in_a_concave_polygon_with_a_hole():
    """Test that 20000 agents lie in an L with a hole, with the share of each unit square equal to its area share."""
    polygon, hole = _concave_polygon()
    group = GroupElement('people', element=IdleElement, size=20000, location=polygon)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(2.0, 2.0)))
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

    class ReaderElement(Element):
        """Test helper that reads the regions of its agent."""

        def _get_laws(self, namespace):
            """Test helper laws."""
            return [RuleLaw(read)]

    group = GroupElement('people', element=ReaderElement, size=30, location=places[0])
    move = MovementInteraction('move', target='people', step=to_favourite, schedule='member')
    system = CompositeSystem(
        elements=[group],
        interactions=[move],
        space=ContinuousSpace(extent=(10.0, 10.0), regions={'places': places}),
    )
    locations = _simulate(system, until=2).get_agents('people', 2).locations
    assert all(places[1].contains(Point(point)) for point in locations)
    assert 3 in found
    assert [0] in found
    assert any(1 in item for item in found if isinstance(item, list))


def test_created_agent_is_placed_in_a_polygon():
    """Test that a created agent starts inside the polygon of its location."""
    region = box(1.0, 1.0, 2.0, 2.0)

    class SpawnerElement(Element):
        """Test helper that creates one agent in a region."""

        def _get_laws(self, namespace):
            """Test helper laws."""
            return [RuleLaw(self.spawn)]

        def spawn(self, agent, system):
            """Test helper rule."""
            if system.count('people') == 1:
                system.create('people', location=region)

    group = GroupElement('people', element=SpawnerElement, size=1, location=(4.0, 4.0), variable=True)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
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
    group = GroupElement('people', element=IdleElement, size=1, location=(1.0, 1.0))
    move = MovementInteraction('move', target='people', step=step)
    space = ContinuousSpace(extent=(5.0, 5.0), regions={'places': [box(0.0, 0.0, 1.0, 1.0)]})
    with pytest.raises(error, match=message):
        _simulate(CompositeSystem(elements=[group], interactions=[move], space=space), until=1)


def test_polygon_is_no_location_of_a_lattice():
    """Test that a polygon as the location of an agent on a lattice raises a named error."""
    group = GroupElement('people', element=IdleElement, size=1, location=box(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(SpaceError, match='location of a continuous space'):
        _simulate(CompositeSystem(elements=[group], space=LatticeSpace((5, 5))))


def test_snapshots_keep_tuples_as_they_were():
    """Test that each snapshot keeps one tuple per agent, in a one-dimensional array of objects."""
    trajectory = _queue_run()
    first, last = trajectory.get_agents('queues', 0), trajectory.get_agents('queues', 3)
    assert first.values['orders'].shape == (2,)
    assert first.values['orders'].dtype == object
    assert first.values['orders'].tolist() == [(), (7,)]
    assert last.values['orders'].tolist() == [(2, 3), (2, 3)]


def test_composition_counts_agents_by_tuple():
    """Test that the composition of a group counts its agents by the tuples they hold."""
    seen = []

    class CountingElement(QueueElement):
        """Agent of a queue that records the composition of its group."""

        def _step(self, agent, system):
            """Record the composition, and then step."""
            seen.append(dict(system.get_composition('queues', 'orders')))
            super()._step(agent, system)

    group = GroupElement('queues', element=CountingElement, size=2, values={'orders': [(), (7,)]})
    simulate(CompositeSystem(elements=[group], time=DiscreteTime()), until=1, random_state=0)
    assert seen[0] == {(): 1, (7,): 1}
    assert sum(seen[1].values()) == 2


def test_history_keeps_reported_objects():
    """Test that the reports of a run keep an object at every frame."""
    values = _queue_run().get_values('queues.longest')
    assert values.dtype == object
    assert isinstance(values[-1], tuple)


def test_observable_keeps_a_hashable_value():
    """Test that an observable of a group by a function of its values may give an object."""
    observables = [ObservableSpec('lengths', group='queues', variable='orders', reduce=_lengths)]
    group = GroupElement('queues', element=QueueElement, size=2, values={'orders': [(), (7,)]})
    system = CompositeSystem(elements=[group], time=DiscreteTime())
    trajectory = simulate(system, until=2, observables=observables, random_state=0)
    assert trajectory.get_values('lengths')[0] == pytest.approx([0, 1])


def test_offer_before_acceptance():
    """Every acceptance refers to an offer made earlier in the same step."""
    system = CompositeSystem(
        elements=[
            GroupElement('workers', element=WorkerElement, size=5),
            GroupElement('employers', element=EmployerElement, size=3),
        ],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('offers'), PhaseSpec('acceptance')])),
    )
    trajectory = simulate(system, until=10, random_state=0)
    workers = trajectory.get_agents('workers', 10)
    assert np.all(workers.values['stale'] == 0)
    assert np.all(workers.values['accepted'] == trajectory.get_agents('employers', 10).values['offer'].max())


def test_undeclared_phase_raises():
    """A rule in a phase that the system does not declare raises an error."""
    system = CompositeSystem(elements=[GroupElement('employers', element=EmployerElement, size=2)])
    with pytest.raises(SimulationError, match='offers'):
        simulate(system, until=1)


def test_run_reads_the_phases_of_a_nested_system():
    """Test that rules run in a phase that the schedule of a nested system declares."""
    nested = CompositeSystem(
        label='market',
        elements=[GroupElement('employers', element=EmployerElement, size=2)],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('offers')])),
    )
    trajectory = simulate(CompositeSystem(systems=[nested]), until=3, random_state=0)
    assert np.all(trajectory.get_agents('market/employers', 3).values['offer'] == 4.0)


def test_phases_keep_unphased_systems():
    """Rules without a phase run in the default phase."""

    class CounterElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            count: Annotated[float, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.count = count

        def _get_laws(self, s):
            return [RuleLaw(self.tick)]

        def tick(self, agent, system):
            agent.count += 1

    system = CompositeSystem(
        elements=[GroupElement('counters', element=CounterElement, size=3)],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('extra', repeat=2)])),
    )
    trajectory = simulate(system, until=4, random_state=0)
    assert trajectory.get_agents('counters', 4).values['count'].tolist() == [4.0, 4.0, 4.0]


def test_sequential_and_by_type_order():
    """Sequential activation follows creation, and by-type activation follows the listed groups."""
    seen = []

    class FirstElement(Element):
        def _get_laws(self, s):
            return [RuleLaw(lambda agent, system: seen.append(('first', agent.id)), phase='go')]

    class SecondElement(Element):
        def _get_laws(self, s):
            return [RuleLaw(lambda agent, system: seen.append(('second', agent.id)), phase='go')]

    system = CompositeSystem(
        elements=[
            GroupElement('first', element=FirstElement, size=3),
            GroupElement('second', element=SecondElement, size=3),
        ],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('go', activation='by-type', groups=['second', 'first'])]),
        ),
    )
    simulate(system, until=1, random_state=0)
    assert [group for group, _ in seen] == ['second'] * 3 + ['first'] * 3
    seen.clear()
    sequential = CompositeSystem(
        elements=[
            GroupElement('first', element=FirstElement, size=3),
            GroupElement('second', element=SecondElement, size=3),
        ],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('go', 'sequential')])),
    )
    simulate(sequential, until=1, random_state=0)
    assert [identity for _, identity in seen] == sorted(identity for _, identity in seen)
