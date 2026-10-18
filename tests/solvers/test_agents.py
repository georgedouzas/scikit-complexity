"""Test agent-based models on Mesa: form inference, spaces, populations, reproducibility, and wrapped models."""

import time

import mesa
import networkx as nx
import numpy as np
import pytest

from skcomplex.compile import compile_system
from skcomplex.core import Form, IncompatibleLawsError, ValidationError, parameter, role, state
from skcomplex.model import (
    AgentType,
    ContinuousSpace,
    GridSpace,
    Interaction,
    MesaSystem,
    NetworkSpace,
    Rule,
    System,
)
from skcomplex.models.ecology import ExponentialGrowth, Population
from skcomplex.runtime import RealTime, Simulation


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
    with pytest.raises(IncompatibleLawsError, match='Rate, Rule'):
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
