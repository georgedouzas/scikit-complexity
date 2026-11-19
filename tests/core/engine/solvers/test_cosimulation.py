"""Test co-simulation: partitioning, held inputs, held reductions, agent inputs, scheduling, and convergence."""

import numpy as np
import pytest

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import count, maximum, mean, neighbors, total
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.compile._observables import ObservableEvaluator
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.solvers import CoSimulationSolver, MetropolisSolver, ScipyODESolver
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import IncompatibleLawsError, SolverError, ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.populations import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import PredationInteraction
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
from skcomplex.systems.economics.resources import HarvestingSystem

REDUCTIONS = {'sum': total, 'mean': mean, 'max': maximum}


def breed(agent, system):
    """Create one offspring of an agent."""
    del system
    agent.create()


def observe(agent, system):
    """Record the soil an agent sees at the last synchronisation point."""
    agent.seen = system.inputs['land/soil.x']


def tend(system):
    """Do nothing for the whole system."""
    del system


def _rule(label, target, rule):
    """Return a one-off interaction that runs a rule on every member of a group."""
    return Interaction(label, roles={'target': target}, laws=lambda s: [UpdateLaw(s.target, rule)])


def _counting(label, target, source):
    """Return a one-off interaction whose rate is the number of members of a group, `dn/dt = count(source)`."""
    return Interaction(
        label,
        roles={'target': target, 'source': source},
        laws=lambda s: [RateLaw(s.target.x, count(s.source))],
    )


def _wealth(label, target, source, reduction='sum'):
    """Return a one-off interaction whose rate reduces the wealth of a group, such as `dn/dt = total(source.wealth)`."""
    return Interaction(
        label,
        roles={'target': target, 'source': source},
        laws=lambda s: [RateLaw(s.target.x, REDUCTIONS[reduction](s.source.wealth))],
    )


def _farmers(size, label='farmers'):
    """Return a village of idle farmers who record the soil they see."""
    return System(
        'village',
        elements=[Element(label, values={'wealth': 1.0, 'seen': 0.0}, size=size)],
        interactions=[_rule('observe', label, observe)],
    )


def _lotka_volterra(solver='auto'):
    """Return a Lotka-Volterra system split into a prey and a predator subsystem."""
    return System(
        systems=[
            System(
                'prey',
                elements=[PopulationElement('rabbits', x=40)],
                interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.1, K=1000)],
            ),
            System(
                'predators',
                elements=[PopulationElement('foxes', x=9)],
                interactions=[MortalityInteraction('death', target='foxes', d=0.4)],
            ),
        ],
        interactions=[PredationInteraction('hunt', predator='predators/foxes', prey='prey/rabbits', a=0.4, b=0.1)],
        solver=solver,
    )


def _counting_system(agents=3, solver='auto', village_first=False):
    """Return land and a village of breeders whose count a land rate reads."""
    land = System('land', elements=[PopulationElement('soil', x=0.0)])
    village = System(
        'village',
        elements=[Element('farmers', size=agents)],
        interactions=[_rule('breed', 'farmers', breed)],
    )
    return System(
        systems=[village, land] if village_first else [land, village],
        interactions=[_counting('accumulate', 'land/soil', 'village/farmers')],
        solver=solver,
    )


def _ledger(interaction):
    """Return a ledger tally beside three growing herds, joined by an interaction."""
    return System(
        systems=[
            System('ledger', elements=[Element('tally', values={'k': 0.0})]),
            System(
                'herds',
                elements=[Element('herd', values={'x': [1.0, 2.0, 3.0]}, size=3)],
                interactions=[ExponentialGrowthInteraction('growth', target='herd')],
            ),
        ],
        interactions=[interaction],
    )


def test_agents_and_ode_split_into_two_partitions():
    """Test that agents and differential equations in sibling subsystems give an agent and an ODE partition."""
    problem = HarvestingSystem().prepare().system_._problem
    assert problem.form == FormKind.COUPLED
    assert [(item.name, item.problem.form) for item in problem.partitions] == [
        ('land', FormKind.ODE),
        ('village', FormKind.AGENT),
    ]


def test_laws_join_the_partition_of_their_target():
    """Test that one interaction's rate and system rule join the partitions of their targets, the rate per farmer."""
    tending = Interaction(
        'tend',
        roles={'soil': 'land/soil', 'farmers': 'village/farmers'},
        laws=lambda s: [RateLaw(s.soil.x, 1.0), UpdateLaw(s, tend)],
    )
    system = System(
        systems=[
            System('land', elements=[PopulationElement('soil', x=0.0)]),
            System('village', elements=[Element('farmers', values={'wealth': 1.0}, size=2)]),
        ],
        interactions=[tending],
    ).prepare()
    land, village = system._problem.partitions
    assert (land.problem.form, village.problem.form) == (FormKind.ODE, FormKind.AGENT)
    assert [path for path, _ in village.problem.system_rules] == ['tend']
    assert system.simulate(until=2, dt=1).get_values('land/soil.x')[-1] == pytest.approx(2 * 2)


def test_mixed_leaf_subsystem_names_itself():
    """Test that a subsystem without children that mixes kinds names itself and suggests splitting it."""
    system = System(
        systems=[
            System(
                'economy',
                elements=[MarketElement('market'), PopulationElement('herd')],
                interactions=[
                    CobwebInteraction('cobweb', target='market'),
                    ExponentialGrowthInteraction('growth', target='herd'),
                ],
            ),
        ],
    )
    with pytest.raises(IncompatibleLawsError, match=r'economy: No form takes the laws.*nested system'):
        PassCompiler().prepare(system._get_composite()).run()


def test_mix_across_subsystems_gives_partitions():
    """Test that a discrete subsystem next to an ODE subsystem gives a discrete and an ODE partition."""
    system = System(
        systems=[
            System(
                'market',
                elements=[MarketElement('wheat')],
                interactions=[CobwebInteraction('cobweb', target='wheat')],
            ),
            System(
                'herd',
                elements=[PopulationElement('sheep')],
                interactions=[ExponentialGrowthInteraction('growth', target='sheep')],
            ),
        ],
    ).prepare()
    assert [item.problem.form for item in system._problem.partitions] == [FormKind.DISCRETE, FormKind.ODE]


def test_held_input_keeps_start_of_interval_value():
    """Test that an ODE reading a count of agents uses its value from the start of the synchronisation interval."""
    system = _counting_system(3, solver=CoSimulationSolver(sync_dt=2)).prepare()
    trajectory = system.simulate(until=2, dt=2)
    assert trajectory.get_values('land/soil.x')[-1] == pytest.approx(3 * 2)
    assert trajectory.get_values('village/farmers_count.value').tolist() == [3.0, 12.0]


def test_held_reduction_counts_agents():
    """Test that an ODE law reading the count of 12 agents sees the value 12 until the next synchronisation point."""
    trajectory = _counting_system(12, solver=CoSimulationSolver()).prepare().simulate(until=1, dt=1)
    assert trajectory.get_values('village/farmers_count.value')[0] == 12
    assert trajectory.get_values('land/soil.x')[-1] == pytest.approx(12)


@pytest.mark.parametrize(('reduction', 'expected'), [('sum', 3.0), ('mean', 1.0), ('max', 1.0)])
def test_held_reductions_of_agent_states(reduction, expected):
    """Test that laws read the total, the mean, and the largest value of an agent state across partitions."""
    system = System(
        systems=[System('land', elements=[PopulationElement('soil')]), _farmers(3)],
        interactions=[_wealth('accumulate', 'land/soil', 'village/farmers', reduction)],
    )
    trajectory = system.prepare().simulate(until=1, dt=1)
    assert trajectory.get_values(f'village/farmers_{reduction}.value')[0] == expected
    assert trajectory.get_values('land/soil.x')[-1] == pytest.approx(1.0 + expected)


def test_held_reductions_of_one_group_share_an_element():
    """Test that two laws reading the same reduction of agents read one element, and other reductions get their own."""
    system = System(
        systems=[System('land', elements=[PopulationElement('soil'), PopulationElement('dust')]), _farmers(3)],
        interactions=[
            _wealth('first', 'land/soil', 'village/farmers'),
            _wealth('second', 'land/dust', 'village/farmers'),
            _counting('third', 'land/dust', 'village/farmers'),
        ],
    ).prepare()
    paths = [path for path, _ in ObservableEvaluator._get_held_reductions_of(system._problem)]
    assert paths == ['village/farmers_sum', 'village/farmers_count']


def test_reduction_of_element_group_reads_other_partition():
    """Test that a discrete law reads the total of an element group of an ODE partition at the synchronisation point."""
    tally = Interaction(
        'tally',
        roles={'target': 'ledger/tally', 'source': 'herds/herd'},
        laws=lambda s: [UpdateLaw(s.target.k, total(s.source.x))],
    )
    system = _ledger(tally).prepare()
    trajectory = system.simulate(until=1, dt=1)
    assert ObservableEvaluator._get_held_reductions_of(system._problem) == []
    assert trajectory.get_values('ledger/tally.k')[-1] == pytest.approx(6.0)


def test_held_observable_reduces_element_group():
    """Test that an observable reducing an element group with a function is held for a role of another partition."""
    copy = Interaction(
        'copy',
        roles={'target': 'ledger/tally', 'source': 'herds/largest'},
        laws=lambda s: [UpdateLaw(s.target.k, s.source.value)],
    )
    largest = Observable('herds/largest', group='herd', reduce=np.max, variable='x')
    trajectory = _ledger(copy).prepare([largest]).simulate(until=1, dt=1)
    assert trajectory.get_values('herds/largest.value')[0] == 3.0
    assert trajectory.get_values('ledger/tally.k')[-1] == pytest.approx(3.0)


def test_subsystem_partitions_hold_reductions_of_agents():
    """Test that compiling one partition per subsystem holds the reductions of agents that laws read."""
    problem = PassCompiler(partition='subsystems').prepare(HarvestingSystem()._get_composite()).run().problem_
    assert [partition.name for partition in problem.partitions] == ['land', 'village']


def test_agent_rules_read_other_partitions():
    """Test that agent rules read the states of the ODE partition at the last synchronisation point."""
    system = System(
        systems=[
            System(
                'land',
                elements=[PopulationElement('soil', x=5.0)],
                interactions=[ExponentialGrowthInteraction('growth', target='soil', r=1.0)],
            ),
            _farmers(2),
        ],
    )
    trajectory = system.prepare().simulate(until=1, dt=1)
    assert trajectory.get_agents('village/farmers', 1).values['seen'].tolist() == pytest.approx(
        [5.0 * np.e] * 2,
        rel=1e-5,
    )


def test_sync_interval_must_divide_frame():
    """Test that a synchronisation interval that does not divide the frame names both values."""
    system = _lotka_volterra(CoSimulationSolver(split=True, sync_dt=0.03)).prepare()
    with pytest.raises(ValidationError, match=r'0\.03.*0\.1'):
        system.simulate(until=0.1, dt=0.1)


def test_sync_interval_must_be_whole_steps_for_agents():
    """Test that agent partitions need a whole number of steps between synchronisation points."""
    with pytest.raises(ValidationError, match='whole number of steps'):
        HarvestingSystem(solver=CoSimulationSolver(sync_dt=0.5)).prepare().simulate(until=1, dt=1)


def test_gauss_seidel_reads_already_advanced_values():
    """Test that Gauss-Seidel reads the partitions advanced before and Jacobi reads the start of the interval."""
    gauss_seidel = _counting_system(3, CoSimulationSolver(), village_first=True).prepare().simulate(until=1, dt=1)
    jacobi = _counting_system(3, CoSimulationSolver(scheme='jacobi'), village_first=True)
    assert gauss_seidel.get_values('land/soil.x')[-1] == pytest.approx(6)
    assert jacobi.prepare().simulate(until=1, dt=1).get_values('land/soil.x')[-1] == pytest.approx(3)


def test_unknown_scheme_raises_error():
    """Test that an unknown scheme is rejected."""
    with pytest.raises(SolverError, match='gauss-seidel, jacobi'):
        HarvestingSystem(solver=CoSimulationSolver(scheme='newton')).prepare().simulate(until=1)


def test_single_form_needs_split():
    """Test that a problem of one form needs `split=True` to run as a co-simulation."""
    with pytest.raises(SolverError, match='split=True'):
        _lotka_volterra(CoSimulationSolver()).prepare().simulate(until=0.1)


def test_split_lotka_volterra_converges():
    """Test that the splitting error shrinks by a factor of at least 1.8 with each halving of `sync_dt`."""
    precise = ScipyODESolver(rtol=1e-10, atol=1e-12)
    joint = _lotka_volterra(precise).prepare().simulate(until=10, dt=0.1).states[-1]
    errors = []
    for interval in (0.1, 0.05, 0.025):
        solver = CoSimulationSolver(split=True, sync_dt=interval, solvers={'prey': precise, 'predators': precise})
        split = _lotka_volterra(solver).prepare().simulate(until=10, dt=0.1).states[-1]
        errors.append(np.abs(split - joint).max())
    assert errors[0] >= 1.8 * errors[1]
    assert errors[1] >= 1.8 * errors[2]


def test_mixed_trajectory_holds_states_and_agents():
    """Test that a co-simulation trajectory returns ODE states per frame and agent states per agent."""
    trajectory = HarvestingSystem(5).prepare().simulate(until=10, random_state=0)
    assert trajectory.get_values('land/stock.n').shape == (11,)
    snapshot = trajectory.get_agents('village/farmers', 10)
    assert len(snapshot.values['wealth']) == 5
    assert np.all(snapshot.values['wealth'] > 0)


def test_cosimulation_is_reproducible():
    """Test that equal random states give equal co-simulations with agents."""
    system = _counting_system(3).prepare()
    runs = [system.simulate(until=3, random_state=0, dt=1) for _ in range(2)]
    assert np.array_equal(runs[0].states, runs[1].states)


def test_live_parameter_reaches_partitions():
    """Test that a parameter change during a run applies inside the partitions."""
    trajectory = (
        HarvestingSystem(10, effort=0.05)
        .prepare()
        .simulate(
            until=30,
            schedule={'harvest.effort': lambda t: 0.05 if t < 12 else 0.0},
        )
    )
    stock = trajectory.get_values('land/stock.n')
    assert stock[-1] > stock[10]


def _ising(s):
    """Return the objective of an Ising spin, minus its aligned neighbours over the temperature."""
    return [ObjectiveLaw(-s.target.s * neighbors(s.target.other.s) / s.target.T, s.target.s)]


def _tick(s):
    """Return the rate of a clock, one per unit of time."""
    return [RateLaw(s.target.n, 1)]


@pytest.mark.parametrize(('temperature', 'ordered'), [(0.1, True), (100.0, False)])
def test_cosimulation_nested_system_keeps_its_metropolis_solver(temperature, ordered):
    """Test that a magnet nested beside a clock is sampled by its own Metropolis solver, ordered only when cold."""
    ring = Element('ring', values={'s': np.ones(16), 'T': temperature}, space=LatticeSpace(16))
    ising = Interaction('ising', roles={'target': 'ring'}, laws=_ising)
    tick = Interaction('tick', roles={'target': 'clock'}, laws=_tick)
    solver = MetropolisSolver(moves={'ring.s': 'flip'})
    system = System(
        systems=[
            System('magnet', elements=[ring], interactions=[ising], solver=solver),
            System('timer', elements=[Element('clock', values={'n': 0.0})], interactions=[tick]),
        ],
    ).prepare()
    trajectory = system.simulate(until=20, random_state=0)
    assert isinstance(system.solver_, CoSimulationSolver)
    assert trajectory.get_values('timer/clock.n')[-1] == pytest.approx(20.0)
    assert bool(trajectory.get_values('magnet/ring.s')[-1].mean() == 1.0) is ordered
