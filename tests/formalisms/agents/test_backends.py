"""Test agent-based models on the Mesa backend: recognition, spaces, populations, reproducibility, and support."""

from typing import Any

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.base import (
    BaseInteraction,
    Form,
    IncompatibleLawsError,
    LabelError,
    SimulationError,
    SpaceError,
    TimeKind,
    ValidationError,
    parameter,
    role,
    state,
)
from skcomplex.core.components import ElementComponent, GroupComponent
from skcomplex.core.space import ContinuousSpace, DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system, validate
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.agents import AgentView, MesaSolver, RuleLaw, runs_rules
from skcomplex.formalisms.agents._passes import AGENT_RESERVED_NAMES
from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationComponent


class BreederComponent(ElementComponent):
    """Agent that creates one offspring every step."""

    def laws(self, s):
        """Return the breeding rule."""
        return [RuleLaw(self.breed)]

    def breed(self, agent, model):
        """Create one offspring."""
        agent.create()


class WalkerComponent(ElementComponent):
    """Agent that moves to a random neighbouring cell and counts its moves."""

    moves: float = state(0.0)

    def laws(self, s):
        """Return the walking rule."""
        return [RuleLaw(self.walk)]

    def walk(self, agent, model):
        """Move to a random neighbouring cell."""
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])
        agent.moves += 1


class ProbeComponent(ElementComponent):
    """Agent that records how many neighbouring cells and neighbours it has."""

    cells: float = state(0.0)
    near: float = state(0.0)

    def laws(self, s):
        """Return the probing rule."""
        return [RuleLaw(self.probe)]

    def probe(self, agent, model):
        """Record the number of neighbouring cells and of neighbours."""
        agent.cells = len(agent.neighbor_cells())
        agent.near = len(agent.neighbors())


class DrifterComponent(ElementComponent):
    """Agent in a continuous space that counts the agents within a radius."""

    near: float = state(0.0)
    radius: float = parameter(1.0)

    def laws(self, s):
        """Return the counting rule."""
        return [RuleLaw(self.count)]

    def count(self, agent, model):
        """Count the agents within the radius."""
        agent.near = len(agent.neighbors(radius=agent.radius))


class CullerInteraction(BaseInteraction):
    """Model rule that removes the agent with the smallest id once."""

    target: str = role()

    def laws(self, s):
        """Return the culling rule."""
        return [RuleLaw(self.cull)]

    def cull(self, model):
        """Remove the oldest agent while more than two remain."""
        agents = sorted(model.agents(self.target), key=lambda agent: agent.id)
        if len(agents) > 2:
            agents[0].remove()


class MoneyAgentComponent(ElementComponent):
    """Agent of the Boltzmann wealth model, which gives the transfer to a random agent when it has wealth."""

    wealth: float = state(1.0)
    transfer: float = parameter(1.0)

    def laws(self, s):
        """Return the giving rule."""
        return [RuleLaw(self.give)]

    def give(self, agent, model):
        """Give the transfer to a random agent of the group."""
        if agent.wealth > 0:
            agents = model.agents(agent.group)
            other = agents[int(model.rng.integers(len(agents)))]
            other.wealth += agent.transfer
            agent.wealth -= agent.transfer


def _money_system():
    """Return the Boltzmann wealth model as an ordinary system of 50 agents."""
    return System('economy', components=[MoneyAgentComponent.group('agents', 50)], time=DiscreteTime())


def test_rule_laws_make_elements_agents():
    """Test that elements and groups with rule laws, and variable groups, run on the agent stepper."""
    plain = ElementComponent.define('PlainComponent', states={'x': 0.0})
    assert runs_rules(WalkerComponent.group('walkers', 3))
    assert runs_rules(WalkerComponent('walker'))
    assert runs_rules(GroupComponent('births', element=plain, size=2, variable=True))
    assert not runs_rules(plain.group('plain', 2))
    assert not runs_rules(PopulationComponent('herd'))


class StallComponent(ElementComponent):
    """Passive agent without laws, whose stock shoppers buy."""

    stock: float = state(3.0)


class ShopperComponent(ElementComponent):
    """Agent that buys one unit from the first stall that has stock."""

    def laws(self, s):
        """Return the buying rule."""
        return [RuleLaw(self.buy)]

    def buy(self, agent, model):
        """Buy one unit from the first stall with stock."""
        stall = model.agents('stalls')[0]
        if stall.stock > 0:
            stall.stock -= 1


def test_elements_without_laws_are_passive_agents():
    """Test that a group without laws next to agents with rules is a group of agents that the rules change."""
    system = System(components=[StallComponent.group('stalls', 1), ShopperComponent.group('shoppers', 2)])
    trajectory = Simulation(system, random_state=0).run(until=2).trajectory_
    assert [trajectory.agents('stalls', frame).values['stock'].tolist() for frame in (1, 2)] == [[1.0], [0.0]]


def test_rules_give_agent_form():
    """Test that a system of agent groups with rules compiles to the agent-based form."""
    problem = compile_system(
        System(components=[WalkerComponent.group('walkers', 3)], space=LatticeSpace((5, 5), boundary='fixed')),
    )
    assert problem.form == Form.AGENT
    assert problem.agents[0].states == ('moves',)


def test_rules_mixed_with_rates_raise_error():
    """Test that rules and rates in one system raise the law kind error."""
    system = System(
        components=[WalkerComponent.group('walkers', 3), PopulationComponent('herd')],
        interactions=[ExponentialGrowthInteraction('growth', target='herd')],
        space=LatticeSpace((5, 5), boundary='fixed'),
    )
    with pytest.raises(IncompatibleLawsError, match='rate: herd; rule: walkers'):
        compile_system(system)


def test_agents_are_born_every_step():
    """Test that one offspring per agent per step gives 10, 20, 40, and 80 agents."""
    trajectory = (
        Simulation(System(components=[BreederComponent.group('herd', 10, variable=True)])).run(until=3).trajectory_
    )
    assert trajectory.values('herd.count').tolist() == [10.0, 20.0, 40.0, 80.0]
    assert len(trajectory.agents('herd', 3).ids) == 80


def test_fixed_group_rejects_births():
    """Test that a rule creating a member of a group of fixed size raises an error that suggests a variable group."""
    with pytest.raises(SimulationError, match='variable=True'):
        Simulation(System(components=[BreederComponent.group('herd', 10)])).run(until=1)


def test_model_rules_remove_agents():
    """Test that an interaction rule removes agents and frames report the agents alive."""
    system = System(
        components=[WalkerComponent.group('walkers', 5, variable=True)],
        interactions=[CullerInteraction('cull', target='walkers')],
        space=LatticeSpace((5, 5), boundary='fixed'),
    )
    trajectory = Simulation(system, random_state=0).run(until=3).trajectory_
    assert trajectory.values('walkers.count').tolist() == [5.0, 4.0, 3.0, 2.0]
    assert 1 not in trajectory.agents('walkers', 1).ids.tolist()


@pytest.mark.parametrize(('neighborhood', 'expected'), [('moore', 8), ('von_neumann', 4)])
def test_grid_neighbours(neighborhood, expected):
    """Test that an agent in the middle of a 10 by 10 grid has 8 Moore or 4 von Neumann neighbouring cells."""
    system = System(
        components=[ProbeComponent.group('probe', 1, location=(5, 5))],
        space=LatticeSpace((10, 10), boundary='fixed', neighborhood=neighborhood),
    )
    snapshot = Simulation(system).run(until=1).trajectory_.agents('probe', 1)
    assert snapshot.values['cells'].tolist() == [expected]


def test_network_neighbours_follow_edges():
    """Test that agents on a cycle network see their two neighbours."""
    system = System(components=[ProbeComponent.group('nodes', 6)], space=NetworkSpace(nx.cycle_graph(6)))
    snapshot = Simulation(system).run(until=1).trajectory_.agents('nodes', 1)
    assert snapshot.values['near'].tolist() == [2.0] * 6
    assert sorted(snapshot.locations.tolist()) == list(range(6))


def test_continuous_neighbours_within_radius():
    """Test that agents in a continuous space find the agents within a radius."""
    system = System(
        components=[DrifterComponent.group('drifters', 3, location=[(1.0, 1.0), (1.5, 1.0), (8.0, 8.0)])],
        space=ContinuousSpace(10.0, 10.0),
    )
    snapshot = Simulation(system).run(until=1).trajectory_.agents('drifters', 1)
    assert snapshot.values['near'].tolist() == [1.0, 1.0, 0.0]


def test_agent_runs_are_reproducible():
    """Test that equal random states give equal agent trajectories."""
    runs = [
        Simulation(
            System(components=[WalkerComponent.group('walkers', 5)], space=LatticeSpace((10, 10), boundary='fixed')),
            random_state=0,
        ).run(
            until=5,
        )
        for _ in range(2)
    ]
    first, second = (run.trajectory_.agents('walkers', 5) for run in runs)
    assert np.array_equal(first.locations, second.locations)
    assert first.values['moves'].tolist() == [5.0] * 5


def test_agent_frames_use_integer_steps():
    """Test that agent-based models reject a non-integer frame duration."""
    with pytest.raises(ValidationError, match='positive integer'):
        Simulation(System(components=[BreederComponent.group('herd', 2, variable=True)]), dt=0.5).run(until=1)


def test_boltzmann_wealth_keeps_wealth():
    """Test that the Boltzmann wealth model as an ordinary system keeps its total wealth and its agents' ids."""
    trajectory = Simulation(_money_system(), random_state=0).run(until=100).trajectory_
    totals = {int(trajectory.agents('agents', frame).values['wealth'].sum()) for frame in range(101)}
    assert totals == {50}
    assert trajectory.agents('agents', 5).ids.tolist() == list(range(1, 51))


def test_agent_parameter_changes_live():
    """Test that a parameter update reaches the rules of the agents."""
    schedule = {'agents.transfer': lambda t: 1.0 if t < 5 else 0.0}
    trajectory = Simulation(_money_system(), schedule=schedule, random_state=0).run(until=10).trajectory_
    frames = [trajectory.agents('agents', frame).values['wealth'] for frame in (7, 10)]
    assert np.array_equal(*frames)


def test_mesa_declares_its_support():
    """Test that the Mesa backend declares discrete time, rule laws, and fixed and variable groups."""
    supports = MesaSolver.supports
    assert supports.time == {TimeKind.DISCRETE}
    assert not supports.missing('laws', {RuleLaw.law_kind})
    assert {'group', 'variable-group'} <= {str(kind) for kind in supports.components}


def test_mesa_validates_systems_against_its_support():
    """Test that validation reports a system of rate laws and continuous time for the Mesa backend."""
    system = System(
        components=[PopulationComponent('herd')],
        interactions=[ExponentialGrowthInteraction('growth', target='herd')],
    )
    issues = validate(system, solver='mesa').issues
    assert [issue.code for issue in issues] == ['solver-unsupported']
    assert 'rate' in issues[0].message
    assert validate(_money_system(), solver='mesa').ok


SEEN: list[object] = []


class CattleComponent(ElementComponent):
    """Agent that stays where it is and that a census interaction counts."""

    def laws(self, s):
        """Return the rule of staying."""
        return [RuleLaw(self.stay)]

    def stay(self, agent, model):
        """Stay in place."""


class CensusInteraction(BaseInteraction):
    """Interaction whose rule records its parameter, the groups of its roles, and the labels of their agents."""

    counted: str = role()
    others: str = role()
    threshold: float = parameter(0.9)

    def __init__(
        self,
        label: str,
        *,
        looked_up: str = 'counted',
        **values: Any,
    ) -> None:
        """Store the settings and pass the values on."""
        super().__init__(label, **values)
        self.looked_up = looked_up

    def laws(self, s):
        """Return the census rule."""
        return [RuleLaw(self.record)]

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
        components=[CattleComponent.group('cows', 42), CattleComponent.group('patches', 43)],
        interactions=[CensusInteraction('census', counted='cows', others='patches', looked_up=looked_up)],
    )
    system = System(components=[farm], space=space)
    simulation = Simulation(system, schedule=schedule, random_state=0).run(until=3)
    return simulation, list(SEEN)


def test_rule_reads_parameter_of_nested_owner():
    """Test that a nested interaction reads its own parameter by name, also after a live update."""
    _, records = _census(schedule={'farm/census.threshold': lambda t: 0.9 if t < 2 else 0.5})
    assert [record['threshold'] for record in records] == [0.9, 0.9, 0.5]


def test_rule_binds_two_agent_groups_through_roles():
    """Test that an interaction binds two agent groups of different sizes and reads each through its role."""
    simulation, records = _census()
    assert simulation.problem_.form == Form.AGENT
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
    system = System(components=[CattleComponent.group('cows', 2, location=locations)], space=space)
    simulation = Simulation(system, random_state=0).run(until=0)
    stepper = simulation.stepper_
    return [AgentView(agent, stepper) for agent in stepper._group_agents('cows')]


def test_displacement_across_torus_seam():
    """Test that the displacement in a toroidal continuous space is the shortest image across the seam."""
    first, second = _views(ContinuousSpace(100.0, 1.0, boundary='periodic'), [(99.0, 0.5), (1.0, 0.5)])
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
    """Test that every public name of the agent view is reserved for the states of agents."""
    public = {name for name in dir(AgentView) if not name.startswith('_')}
    assert public <= AGENT_RESERVED_NAMES


def test_agent_state_with_reserved_name_raises_error():
    """Test that an agent state named like an attribute of the agent view raises a label error at compilation."""
    MoverComponent = ElementComponent.define(
        'MoverComponent',
        states={'alive': 1.0},
        laws=lambda s: [RuleLaw(lambda agent, model: None)],
    )
    with pytest.raises(LabelError, match='declare the variables alive'):
        compile_system(System(components=[MoverComponent.group('movers', 2)]))


def test_non_agent_state_may_take_agent_view_name():
    """Test that an element without rule laws keeps states named like attributes of the agent view."""
    CellComponent = ElementComponent.define('CellComponent', states={'alive': 1.0})
    assert compile_system(System(components=[CellComponent.group('cells', 2)])).states.names == ['cells.alive']


class ClockedComponent(ElementComponent):
    """Agent that records the model time at every step."""

    seen: float = state(-1.0)
    n: float = state(0.0)

    def laws(self, s):
        """Return the recording rule."""
        return [RuleLaw(self.record)]

    def record(self, agent, model):
        """Store the model time."""
        agent.seen = model.time


def test_rules_read_model_time():
    """Test that rules read the model time k during the step that produces frame k."""
    trajectory = Simulation(System(components=[ClockedComponent.group('clocks', 2)])).run(until=3).trajectory_
    assert [trajectory.agents('clocks', step).values['seen'].tolist() for step in (1, 2, 3)] == [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]


def test_one_agent_takes_one_entry_values():
    """Test that a group of one agent reads a one-entry location and a one-entry value as its own."""
    system = System(
        components=[ClockedComponent.group('one', 1, location=[(1, 1)], n=[3.0])],
        space=LatticeSpace((5, 5), boundary='fixed'),
    )
    snapshot = Simulation(system).run(until=0).trajectory_.agents('one', 0)
    assert snapshot.locations.tolist() == [[1, 1]]
    assert snapshot.values['n'].tolist() == [3.0]


def test_mesa_solver_run_sets_agents_and_snapshot():
    """Test that every run of the Mesa solver sets the alive agents by group and the snapshot of the step."""
    system = System(components=[BreederComponent.group('breeders', 2, variable=True)])
    problem = compile_system(system)
    solver = MesaSolver().prepare(problem, np.random.default_rng(0), 3.0)
    assert len(solver.agents_['breeders']) == 2
    solver.run(0.0, problem.initial_state, problem.initial_parameters, 2.0)
    assert len(solver.agents_['breeders']) == 8
    assert solver.snapshot_.reporters['breeders.count'] == 8.0
    for name in ('agents_of', 'snapshot', 'create_agent', 'resolve_group', 'spec_of', 'group_agents'):
        assert not hasattr(solver, name)


class ForagerComponent(ElementComponent):
    """Agent that gathers one unit of food every step and records whether it can read an undeclared state."""

    food: float = state(0.0)

    def laws(self, s):
        """Return the foraging rule."""
        return [RuleLaw(self.forage)]

    def forage(self, agent, model):
        """Gather one unit of food and try to read an undeclared state."""
        agent.food += 1
        try:
            _ = agent.wealth
        except AttributeError:
            UNDECLARED.append(True)


UNDECLARED: list[bool] = []


def test_agent_views_read_and_set_states_as_properties():
    """Test that a rule sets a state through its view and that an undeclared state raises Python's own error."""
    UNDECLARED.clear()
    system = System(components=[ForagerComponent.group('foragers', 2)], time=DiscreteTime())
    trajectory = Simulation(system, dt=1.0, random_state=0).run(until=3).trajectory_
    assert trajectory.agents('foragers', 3).values['food'].tolist() == [3.0, 3.0]
    assert UNDECLARED
    assert all(UNDECLARED)
