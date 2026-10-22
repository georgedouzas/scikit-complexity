"""Test agent-based models on Mesa: form inference, spaces, populations, reproducibility, and wrapped models."""

import time

import mesa
import networkx as nx
import numpy as np
import pytest

from skcomplex.core.base import (
    Form,
    IncompatibleLawsError,
    SimulationError,
    SpaceError,
    ValidationError,
    parameter,
    role,
    state,
)
from skcomplex.core.compile import compile_system
from skcomplex.core.model import (
    AgentType,
    ContinuousSpace,
    GridSpace,
    Interaction,
    MesaSystem,
    NetworkSpace,
    Rule,
    System,
)
from skcomplex.core.runtime import RealTime, Simulation
from skcomplex.core.solvers import AgentView
from skcomplex.models.biology.ecology import ExponentialGrowth, Population


class Breeder(AgentType):
    """Agent that creates one offspring every step."""

    def laws(self, s):
        """Return the breeding rule."""
        return [Rule(self.breed)]

    def breed(self, agent, model):
        """Create one offspring."""
        agent.create()


class Walker(AgentType):
    """Agent that moves to a random neighbouring cell and counts its moves."""

    moves: float = state(0.0)

    def laws(self, s):
        """Return the walking rule."""
        return [Rule(self.walk)]

    def walk(self, agent, model):
        """Move to a random neighbouring cell."""
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])
        agent.moves += 1


class Probe(AgentType):
    """Agent that records how many neighbouring cells and neighbours it has."""

    cells: float = state(0.0)
    near: float = state(0.0)

    def laws(self, s):
        """Return the probing rule."""
        return [Rule(self.probe)]

    def probe(self, agent, model):
        """Record the number of neighbouring cells and of neighbours."""
        agent.cells = len(agent.neighbor_cells())
        agent.near = len(agent.neighbors())


class Drifter(AgentType):
    """Agent in a continuous space that counts the agents within a radius."""

    near: float = state(0.0)
    radius: float = parameter(1.0)

    def laws(self, s):
        """Return the counting rule."""
        return [Rule(self.count)]

    def count(self, agent, model):
        """Count the agents within the radius."""
        agent.near = len(agent.neighbors(radius=agent.radius))


class Culler(Interaction):
    """Model rule that removes the agent with the smallest id once."""

    target: str = role()

    def laws(self, s):
        """Return the culling rule."""
        return [Rule(self.cull)]

    def cull(self, model):
        """Remove the oldest agent while more than two remain."""
        agents = sorted(model.agents(self.target), key=lambda agent: agent.id)
        if len(agents) > 2:
            agents[0].remove()


class MoneyAgent(mesa.Agent):
    """Plain Mesa agent of the Boltzmann wealth model."""

    def __init__(self, model):
        """Start with one unit of wealth."""
        super().__init__(model)
        self.wealth = 1

    def step(self):
        """Give the transfer to a random agent when there is wealth to give."""
        if self.wealth > 0:
            other = self.model.random.choice(self.model.agents)
            other.wealth += self.model.transfer
            self.wealth -= self.model.transfer


class MoneyModel(mesa.Model):
    """Plain Mesa Boltzmann wealth model."""

    def __init__(self, n=50, transfer=1, rng=None):
        """Create the agents."""
        super().__init__(rng=rng)
        self.transfer = transfer
        MoneyAgent.create_agents(self, n)

    def step(self):
        """Activate the agents in random order."""
        self.agents.shuffle_do('step')


def _gini(model):
    """Return the Gini coefficient of the wealth of the model's agents."""
    wealth = np.sort([agent.wealth for agent in model.agents])
    n = len(wealth)
    return float((2 * np.arange(1, n + 1) - n - 1) @ wealth / (n * wealth.sum()))


def _money_system():
    """Return the wrapped Mesa Boltzmann wealth model."""
    return MesaSystem(
        'economy',
        model=MoneyModel,
        parameters={'n': 50, 'transfer': 1},
        model_reporters={'gini': _gini},
        agent_reporters={'wealth': 'wealth'},
    )


def test_rules_give_agent_form():
    """Test that a system of agent groups with rules compiles to the agent-based form."""
    problem = compile_system(System(elements=[Walker.group('walkers', 3)], space=GridSpace(5, 5)))
    assert problem.form is Form.AGENT
    assert problem.agents[0].states == ('moves',)


def test_rules_mixed_with_rates_raise_error():
    """Test that rules and rates in one system raise the law kind error."""
    system = System(
        elements=[Walker.group('walkers', 3), Population('herd')],
        interactions=[ExponentialGrowth('growth', target='herd')],
        space=GridSpace(5, 5),
    )
    with pytest.raises(IncompatibleLawsError, match='rate: herd; rule: walkers'):
        compile_system(system)


def test_agents_are_born_every_step():
    """Test that one offspring per agent per step gives 10, 20, 40, and 80 agents."""
    trajectory = Simulation(System(elements=[Breeder.group('herd', 10)])).run(until=3).trajectory_
    assert trajectory['herd.count'].tolist() == [10.0, 20.0, 40.0, 80.0]
    assert len(trajectory.agents('herd', 3)) == 80


def test_model_rules_remove_agents():
    """Test that an interaction rule removes agents and frames report the agents alive."""
    system = System(
        elements=[Walker.group('walkers', 5)],
        interactions=[Culler('cull', target='walkers')],
        space=GridSpace(5, 5),
    )
    trajectory = Simulation(system, random_state=0).run(until=3).trajectory_
    assert trajectory['walkers.count'].tolist() == [5.0, 4.0, 3.0, 2.0]
    assert 1 not in trajectory.agents('walkers', 1).ids.tolist()


@pytest.mark.parametrize(('neighborhood', 'expected'), [('moore', 8), ('von_neumann', 4)])
def test_grid_neighbours(neighborhood, expected):
    """Test that an agent in the middle of a 10 by 10 grid has 8 Moore or 4 von Neumann neighbouring cells."""
    system = System(
        elements=[Probe('probe', location=(5, 5))],
        space=GridSpace(10, 10, neighborhood=neighborhood),
    )
    snapshot = Simulation(system).run(until=1).trajectory_.agents('probe', 1)
    assert snapshot.values['cells'].tolist() == [expected]


def test_network_neighbours_follow_edges():
    """Test that agents on a cycle network see their two neighbours."""
    system = System(elements=[Probe.group('nodes', 6)], space=NetworkSpace(nx.cycle_graph(6)))
    snapshot = Simulation(system).run(until=1).trajectory_.agents('nodes', 1)
    assert snapshot.values['near'].tolist() == [2.0] * 6
    assert sorted(snapshot.locations.tolist()) == list(range(6))


def test_continuous_neighbours_within_radius():
    """Test that agents in a continuous space find the agents within a radius."""
    system = System(
        elements=[Drifter.group('drifters', 3, location=[(1.0, 1.0), (1.5, 1.0), (8.0, 8.0)])],
        space=ContinuousSpace(10.0, 10.0),
    )
    snapshot = Simulation(system).run(until=1).trajectory_.agents('drifters', 1)
    assert snapshot.values['near'].tolist() == [1.0, 1.0, 0.0]


def test_agent_runs_are_reproducible():
    """Test that equal random states give equal agent trajectories."""
    runs = [
        Simulation(System(elements=[Walker.group('walkers', 5)], space=GridSpace(10, 10)), random_state=0).run(until=5)
        for _ in range(2)
    ]
    first, second = (run.trajectory_.agents('walkers', 5) for run in runs)
    assert np.array_equal(first.locations, second.locations)
    assert first.values['moves'].tolist() == [5.0] * 5


def test_agent_frames_use_integer_steps():
    """Test that agent-based models reject a non-integer frame duration."""
    with pytest.raises(ValidationError, match='positive integer'):
        Simulation(System(elements=[Breeder.group('herd', 2)]), dt=0.5).run(until=1)


def test_wrapped_mesa_model_reports_gini_and_keeps_wealth():
    """Test that a wrapped Boltzmann wealth model reports 101 Gini values and keeps its total wealth."""
    trajectory = Simulation(_money_system(), random_state=0).run(until=100).trajectory_
    assert len(trajectory['economy.gini']) == 101
    totals = {int(trajectory.agents('economy/agents', frame).values['wealth'].sum()) for frame in range(101)}
    assert totals == {50}


def test_wrapped_agent_values_carry_ids():
    """Test that the trajectory returns one wealth value per agent with matching ids."""
    snapshot = Simulation(_money_system(), random_state=0).run(until=10).trajectory_.agents('economy/agents', 5)
    assert snapshot.ids.tolist() == list(range(1, 51))
    assert len(snapshot.values['wealth']) == 50


def test_wrapped_parameter_changes_live():
    """Test that a live parameter update sets the attribute of the wrapped model."""
    simulation = Simulation(_money_system(), clock=RealTime(speed=20.0)).start(until=20)
    time.sleep(0.3)
    simulation.update_parameters({'economy.transfer': 2})
    simulation.wait()
    assert simulation.stepper_.model_.transfer == 2
    assert isinstance(simulation.stepper_.model_.transfer, int)


def test_mesa_system_must_be_root():
    """Test that a wrapped Mesa model nested in another system is rejected."""
    with pytest.raises(ValidationError, match='root system'):
        compile_system(System(systems=[_money_system()]))


SEEN: list[object] = []


class Cattle(AgentType):
    """Agent that stays where it is and that a census interaction counts."""

    def laws(self, s):
        """Return the rule of staying."""
        return [Rule(self.stay)]

    def stay(self, agent, model):
        """Stay in place."""


class Census(Interaction):
    """Interaction whose rule records its parameter, the groups of its roles, and the labels of their agents."""

    counted: str = role()
    others: str = role()
    threshold: float = parameter(0.9)
    looked_up: str = 'counted'

    def laws(self, s):
        """Return the census rule."""
        return [Rule(self.record)]

    def record(self, model):
        """Record the parameter, the counts and labels of the groups, and the space of the model."""
        counted = model.agents(self.looked_up)
        SEEN.append(
            {
                'threshold': model.parameter('threshold'),
                'counted': len(counted),
                'others': model.count('others'),
                'labels': {agent.group_label for agent in counted},
                'paths': {agent.group for agent in counted},
                'dimensions': model.dimensions,
                'torus': model.torus,
            },
        )


def _census(looked_up='counted', space=None, schedule=None):
    """Return the records of a census of two groups of 42 and 43 agents nested in the subsystem `farm`."""
    SEEN.clear()
    farm = System(
        'farm',
        elements=[Cattle.group('cows', 42), Cattle.group('patches', 43)],
        interactions=[Census('census', counted='cows', others='patches', looked_up=looked_up)],
    )
    system = System(systems=[farm], space=space)
    simulation = Simulation(system, schedule=schedule, random_state=0).run(until=3)
    return simulation, list(SEEN)


def test_rule_reads_parameter_of_nested_owner():
    """Test that a nested interaction reads its own parameter by name, also after a live update."""
    _, records = _census(schedule={'farm/census.threshold': lambda t: 0.9 if t < 2 else 0.5})
    assert [record['threshold'] for record in records] == [0.9, 0.9, 0.5]


def test_rule_binds_two_agent_groups_through_roles():
    """Test that an interaction binds two agent groups of different sizes and reads each through its role."""
    simulation, records = _census()
    assert simulation.problem_.form is Form.AGENT
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
    _, records = _census(space=GridSpace(30, 20, torus=True))
    assert records[0]['dimensions'] == (30, 20)
    assert records[0]['torus'] is True


def _views(space, locations):
    """Return views of two agents placed at given locations of a space."""
    system = System(elements=[Cattle.group('cows', 2, location=locations)], space=space)
    simulation = Simulation(system, random_state=0).run(until=0)
    stepper = simulation.stepper_
    return [AgentView(agent, stepper) for agent in stepper.group_agents('cows')]


def test_displacement_across_torus_seam():
    """Test that the displacement in a toroidal continuous space is the shortest image across the seam."""
    first, second = _views(ContinuousSpace(100.0, 1.0, torus=True), [(99.0, 0.5), (1.0, 0.5)])
    assert first.displacement_to(second).tolist() == [2.0, 0.0]
    assert first.distance_to(second) == 2.0
    assert first.distance_to((97.0, 0.5)) == 2.0


def test_network_distance_counts_hops():
    """Test that the distance on a network counts hops and that a displacement raises a space error."""
    first, second = _views(NetworkSpace(nx.path_graph(5)), [0, 3])
    assert first.distance_to(second) == 3.0
    with pytest.raises(SpaceError, match='network'):
        first.displacement_to(second)


def test_agent_view_names_are_reserved():
    """Test that every public name of the agent view is reserved for the states of agent types."""
    public = {name for name in dir(AgentView) if not name.startswith('_')}
    assert public <= AgentType.reserved_names


class Clocked(AgentType):
    """Agent that records the model time at every step."""

    seen: float = state(-1.0)
    n: float = state(0.0)

    def laws(self, s):
        """Return the recording rule."""
        return [Rule(self.record)]

    def record(self, agent, model):
        """Store the model time."""
        agent.seen = model.time


def test_rules_read_model_time():
    """Test that rules read the model time k during the step that produces frame k."""
    trajectory = Simulation(System(elements=[Clocked.group('clocks', 2)])).run(until=3).trajectory_
    assert [trajectory.agents('clocks', step).values['seen'].tolist() for step in (1, 2, 3)] == [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]


def test_one_agent_takes_one_entry_values():
    """Test that a group of one agent reads a one-entry location and a one-entry value as its own."""
    system = System(elements=[Clocked.group('one', 1, location=[(1, 1)], n=[3.0])], space=GridSpace(5, 5))
    snapshot = Simulation(system).run(until=0).trajectory_.agents('one', 0)
    assert snapshot.locations.tolist() == [[1, 1]]
    assert snapshot.values['n'].tolist() == [3.0]
