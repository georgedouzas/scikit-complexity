"""Test co-simulation: partitioning, held inputs, aggregates, agent inputs, scheduling, and convergence."""

import time

import numpy as np
import pytest

from skcomplex.compile import compile_system
from skcomplex.core import Form, IncompatibleLawsError, SolverError, ValidationError, parameter, role, state
from skcomplex.model import AgentType, Aggregate, Interaction, Rate, Rule, System
from skcomplex.models.ecology import ExponentialGrowth, LogisticGrowth, Mortality, Population, Predation
from skcomplex.models.economics import Cobweb, Market
from skcomplex.models.social import harvesting
from skcomplex.runtime import RealTime, Simulation
from skcomplex.solvers import CoSimulation, ScipyODE


class Breeder(AgentType):
    """Agent that creates one offspring every step."""

    def laws(self, s):
        """Return the breeding rule."""
        return [Rule(self.breed)]

    def breed(self, agent, model):
        """Create one offspring."""
        agent.create()


class Idle(AgentType):
    """Agent that keeps its wealth and records the soil it sees."""

    wealth: float = state(1.0)
    seen: float = state(0.0)

    def laws(self, s):
        """Return the observing rule."""
        return [Rule(self.observe)]

    def observe(self, agent, model):
        """Record the fertility of the soil at the last synchronisation point."""
        agent.seen = model.inputs['land/soil.n']


class Accumulation(Interaction):
    """Rate equal to an aggregate, `dn/dt = value`."""

    target: str = role()
    source: str = role()

    def laws(self, s):
        """Return the rate `value`."""
        return [Rate(s.target.n, s.source.value)]


class Tending(Interaction):
    """Interaction with a rate on a population and a rule on agents."""

    soil: str = role()
    farmers: str = role()
    rate: float = parameter(1.0)

    def laws(self, s):
        """Return one rate law and one rule."""
        return [Rate(s.soil.n, s.rate), Rule(self.tend)]

    def tend(self, model):
        """Do nothing for the agents."""
        del model


def _lotka_volterra():
    """Return a Lotka-Volterra system split into a prey and a predator subsystem."""
    return System(
        systems=[
            System(
                'prey',
                elements=[Population('rabbits', n=40)],
                interactions=[LogisticGrowth('growth', target='rabbits', r=1.1, K=1000)],
            ),
            System(
                'predators',
                elements=[Population('foxes', n=9)],
                interactions=[Mortality('death', target='foxes', d=0.4)],
            ),
        ],
        interactions=[Predation('hunt', predator='predators/foxes', prey='prey/rabbits', a=0.4, b=0.1)],
    )


def _counting_system(agents=3, reduce='count', variable=None):
    """Return land first, then a village of breeders with an aggregate that a land rate reads."""
    return System(
        systems=[
            System('land', elements=[Population('soil', n=0.0)]),
            System(
                'village',
                elements=[
                    Breeder.group('farmers', agents),
                    Aggregate('size', group='farmers', reduce=reduce, variable=variable),
                ],
            ),
        ],
        interactions=[Accumulation('accumulate', target='land/soil', source='village/size')],
    )


def test_agents_and_ode_split_into_two_partitions():
    """Test that agents and differential equations in sibling subsystems give an agent and an ODE partition."""
    problem = compile_system(harvesting())
    assert problem.form is Form.COUPLED
    assert [(item.name, item.problem.form) for item in problem.partitions] == [
        ('land', Form.ODE),
        ('village', Form.AGENT),
    ]


def test_laws_join_the_partition_of_their_target():
    """Test that one interaction's rate and rule join the partitions of their targets, the rate counted per farmer."""
    system = System(
        systems=[
            System('land', elements=[Population('soil', n=0.0)]),
            System('village', elements=[Idle.group('farmers', 2)]),
        ],
        interactions=[Tending('tend', soil='land/soil', farmers='village/farmers')],
    )
    land, village = compile_system(system).partitions
    assert (land.problem.form, village.problem.form) == (Form.ODE, Form.AGENT)
    assert [path for path, _ in village.problem.model_rules] == ['tend']
    assert Simulation(system, dt=1).run(until=2).trajectory_['land/soil.n'][-1] == pytest.approx(2 * 2)


def test_mixed_leaf_subsystem_names_itself():
    """Test that a subsystem without children that mixes kinds names itself and suggests splitting it."""
    system = System(
        systems=[
            System(
                'economy',
                elements=[Market('market'), Population('herd')],
                interactions=[Cobweb('cobweb', target='market'), ExponentialGrowth('growth', target='herd')],
            ),
        ],
    )
    with pytest.raises(IncompatibleLawsError, match=r'Subsystem economy .*separate subsystems'):
        compile_system(system)


def test_mix_across_subsystems_gives_partitions():
    """Test that a discrete subsystem next to an ODE subsystem gives a discrete and an ODE partition."""
    system = System(
        systems=[
            System('market', elements=[Market('wheat')], interactions=[Cobweb('cobweb', target='wheat')]),
            System('herd', elements=[Population('sheep')], interactions=[ExponentialGrowth('growth', target='sheep')]),
        ],
    )
    assert [item.problem.form for item in compile_system(system).partitions] == [Form.DISCRETE, Form.ODE]


def test_held_input_keeps_start_of_interval_value():
    """Test that an ODE reading an aggregate uses its value from the start of the synchronisation interval."""
    trajectory = Simulation(_counting_system(3), solver=CoSimulation(sync_dt=2), dt=2).run(until=2).trajectory_
    assert trajectory['land/soil.n'][-1] == pytest.approx(3 * 2)
    assert trajectory['village/size.value'].tolist() == [3.0, 12.0]


def test_aggregate_counts_agents():
    """Test that an aggregate counting 12 agents gives the ODE the value 12."""
    trajectory = Simulation(_counting_system(12), solver=CoSimulation(), dt=1).run(until=1).trajectory_
    assert trajectory['village/size.value'][0] == 12
    assert trajectory['land/soil.n'][-1] == pytest.approx(12)


@pytest.mark.parametrize(
    ('reduce', 'expected'),
    [('sum', 3.0), ('mean', 1.0), (lambda values: float(values.max()), 1.0)],
    ids=['sum', 'mean', 'custom'],
)
def test_aggregate_reductions_of_agent_states(reduce, expected):
    """Test that aggregates sum, average, and reduce an agent state with a custom function."""
    system = System(
        systems=[
            System('land', elements=[Population('soil')]),
            System(
                'village',
                elements=[
                    Idle.group('farmers', 3),
                    Aggregate('total', group='farmers', reduce=reduce, variable='wealth'),
                ],
            ),
        ],
        interactions=[Accumulation('accumulate', target='land/soil', source='village/total')],
    )
    trajectory = Simulation(system, dt=1).run(until=1).trajectory_
    assert trajectory['village/total.value'][0] == expected


def test_aggregate_reduces_element_group():
    """Test that an aggregate sums the states of an element group."""
    system = System(
        systems=[
            System(
                'herds',
                elements=[
                    Population.group('herd', 3, n=[1.0, 2.0, 3.0]),
                    Aggregate('total', group='herd', reduce='sum', variable='n'),
                ],
                interactions=[ExponentialGrowth('growth', target='herd')],
            ),
            System('market', elements=[Market('wheat')], interactions=[Cobweb('cobweb', target='wheat')]),
        ],
    )
    trajectory = Simulation(system, dt=1).run(until=1).trajectory_
    assert trajectory['herds/total.value'][0] == 6.0


def test_agent_rules_read_other_partitions():
    """Test that agent rules read the states of the ODE partition at the last synchronisation point."""
    system = System(
        systems=[
            System(
                'land',
                elements=[Population('soil', n=5.0)],
                interactions=[ExponentialGrowth('growth', target='soil', r=1.0)],
            ),
            System('village', elements=[Idle.group('farmers', 2)]),
        ],
    )
    trajectory = Simulation(system, dt=1).run(until=1).trajectory_
    assert trajectory.agents('village/farmers', 1).values['seen'].tolist() == pytest.approx([5.0 * np.e] * 2)


def test_sync_interval_must_divide_frame():
    """Test that a synchronisation interval that does not divide the frame names both values."""
    with pytest.raises(ValidationError, match=r'0\.03.*0\.1'):
        Simulation(_lotka_volterra(), solver=CoSimulation(split=True, sync_dt=0.03), dt=0.1).run(until=0.1)


def test_sync_interval_must_be_whole_steps_for_agents():
    """Test that agent partitions need a whole number of steps between synchronisation points."""
    with pytest.raises(ValidationError, match='whole number of steps'):
        Simulation(harvesting(), solver=CoSimulation(sync_dt=0.5), dt=1).run(until=1)


def test_gauss_seidel_reads_already_advanced_values():
    """Test that Gauss-Seidel reads the partitions advanced before and Jacobi reads the start of the interval."""
    village_first = System(
        systems=[
            System('village', elements=[Breeder.group('farmers', 3), Aggregate('size', group='farmers')]),
            System('land', elements=[Population('soil', n=0.0)]),
        ],
        interactions=[Accumulation('accumulate', target='land/soil', source='village/size')],
    )
    gauss_seidel = Simulation(village_first, solver=CoSimulation(), dt=1).run(until=1).trajectory_
    jacobi = Simulation(village_first, solver=CoSimulation(scheme='jacobi'), dt=1).run(until=1).trajectory_
    assert gauss_seidel['land/soil.n'][-1] == pytest.approx(6)
    assert jacobi['land/soil.n'][-1] == pytest.approx(3)


def test_unknown_scheme_raises_error():
    """Test that an unknown scheme is rejected."""
    with pytest.raises(SolverError, match='gauss-seidel, jacobi'):
        Simulation(harvesting(), solver=CoSimulation(scheme='newton')).run(until=1)


def test_single_form_needs_split():
    """Test that a problem of one form needs `split=True` to run as a co-simulation."""
    with pytest.raises(SolverError, match='split=True'):
        Simulation(_lotka_volterra(), solver=CoSimulation()).run(until=0.1)


def test_split_lotka_volterra_converges():
    """Test that the splitting error shrinks by a factor of at least 1.8 with each halving of `sync_dt`."""
    precise = ScipyODE(rtol=1e-10, atol=1e-12)
    joint = Simulation(_lotka_volterra(), solver=precise, dt=0.1).run(until=10).trajectory_.states[-1]
    errors = []
    for interval in (0.1, 0.05, 0.025):
        solver = CoSimulation(split=True, sync_dt=interval, solvers={'prey': precise, 'predators': precise})
        split = Simulation(_lotka_volterra(), solver=solver, dt=0.1).run(until=10).trajectory_.states[-1]
        errors.append(np.abs(split - joint).max())
    assert errors[0] >= 1.8 * errors[1]
    assert errors[1] >= 1.8 * errors[2]


def test_mixed_trajectory_holds_states_and_agents():
    """Test that a co-simulation trajectory returns ODE states per frame and agent states per agent."""
    trajectory = Simulation(harvesting(5), random_state=0).run(until=10).trajectory_
    assert trajectory['land/stock.n'].shape == (11,)
    snapshot = trajectory.agents('village/farmers', 10)
    assert len(snapshot.values['wealth']) == 5
    assert np.all(snapshot.values['wealth'] > 0)


def test_cosimulation_is_reproducible():
    """Test that equal random states give equal co-simulations with agents."""
    runs = [Simulation(_counting_system(3), random_state=0, dt=1).run(until=3).trajectory_ for _ in range(2)]
    assert np.array_equal(runs[0].states, runs[1].states)


def test_live_parameter_reaches_partitions():
    """Test that a live parameter change applies inside the partitions."""
    simulation = Simulation(harvesting(10, effort=0.05), clock=RealTime(speed=20.0)).start(until=30)
    time.sleep(0.6)
    simulation.update_parameters({'harvest.effort': 0.0})
    simulation.wait()
    stock = simulation.trajectory_['land/stock.n']
    assert stock[-1] > stock[10]
