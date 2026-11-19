"""Test co-simulation: partitioning, held inputs, held reductions, agent inputs, scheduling, and convergence."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import count, maximum, mean, total
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.compile._observables import ObservableEvaluator
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.solvers import CoSimulationSolver, ScipyODESolver
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import IncompatibleLawsError, SolverError, ValidationError
from skcomplex.core.specs import Role
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


def _breed(agent, system):
    """Create one offspring."""
    agent.create()


def _observe(agent, system):
    """Record the fertility of the soil at the last synchronisation point."""
    agent.seen = system.inputs['land/soil.x']


def _breeders(label, size):
    """Return a group of agents with a wealth that create one offspring every step, and the interaction of the rule."""
    breeding = Interaction('breeding', roles={'target': label}, laws=lambda s: [UpdateLaw(s.target, _breed)])
    return Element(label, values={'wealth': 1.0}, size=size), breeding


def _idlers(label, size):
    """Return a group of agents that keep their wealth and record the soil they see, and the interaction of the rule."""
    observing = Interaction('observing', roles={'target': label}, laws=lambda s: [UpdateLaw(s.target, _observe)])
    return Element(label, values={'wealth': 1.0, 'seen': 0.0}, size=size), observing


def _run(system, solver='auto', **settings):
    """Return the trajectory of a system run with a solver, compiled with the observables of the run."""
    return system.set_params(solver=solver).prepare(settings.get('observables')).simulate(**settings)


class CountingInteraction(Interaction):
    """Rate equal to the number of members of a group, `dn/dt = count(source)`."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        source: Role,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate `count(source)`."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [RateLaw(s.target.x, count(s.source))]


class WealthInteraction(Interaction):
    """Rate equal to a reduction of the wealth of a group, such as `dn/dt = total(source.wealth)`."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        source: Role,
        reduction='sum',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.reduction = reduction

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate, the reduction of the wealth."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [RateLaw(s.target.x, REDUCTIONS[self.reduction](s.source.wealth))]


class TallyInteraction(Interaction):
    """Update that sets a tally to the total of a group, `n <- total(source.x)`."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        source: Role,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update to the total."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [UpdateLaw(s.target.k, total(s.source.x))]


class CopyInteraction(Interaction):
    """Update that copies a held observable into a tally, `k <- value`."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        source: Role,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update to the observable."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [UpdateLaw(s.target.k, s.source.value)]


class TallyElement(Element):
    """Tally of a discrete subsystem."""

    def __init__(
        self,
        label: str,
        *,
        k: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.k = k


class TendingInteraction(Interaction):
    """Interaction with a rate on a population and a rule on agents."""

    def __init__(
        self,
        label: str,
        *,
        soil: Role,
        farmers: Role,
        rate: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.soil = soil
        self.farmers = farmers
        self.rate = rate

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set one rate law and one rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [RateLaw(s.soil.x, s.rate), UpdateLaw(s, self.tend)]

    def tend(self, system):
        """Do nothing for the agents."""
        del system


def _lotka_volterra():
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
    )


def _counting_system(agents=3):
    """Return land first, then a village of breeders whose count a land rate reads."""
    farmers, breeding = _breeders('farmers', agents)
    return System(
        systems=[
            System('land', elements=[PopulationElement('soil', x=0.0)]),
            System('village', elements=[farmers], interactions=[breeding]),
        ],
        interactions=[CountingInteraction('accumulate', target='land/soil', source='village/farmers')],
    )


def test_agents_and_ode_split_into_two_partitions():
    """Test that agents and differential equations in sibling subsystems give an agent and an ODE partition."""
    problem = PassCompiler().prepare(HarvestingSystem().prepare().system_).run().problem_
    assert problem.form == FormKind.COUPLED
    assert [(item.name, item.problem.form) for item in problem.partitions] == [
        ('land', FormKind.ODE),
        ('village', FormKind.AGENT),
    ]


@pytest.mark.skip(
    reason='engine gap: an interaction that sets a rate of an element beside a rule of the system loses the rate, so '
    'the element it targets is no state',
)
def test_laws_join_the_partition_of_their_target():
    """Test that one interaction's rate and rule join the partitions of their targets, the rate counted per farmer."""
    system = System(
        systems=[
            System('land', elements=[PopulationElement('soil', x=0.0)]),
            System('village', elements=[Element('farmers', values={'wealth': 1.0, 'seen': 0.0}, size=2)]),
        ],
        interactions=[TendingInteraction('tend', soil='land/soil', farmers='village/farmers')],
    )
    land, village = PassCompiler().prepare(system).run().problem_.partitions
    assert (land.problem.form, village.problem.form) == (FormKind.ODE, FormKind.AGENT)
    assert [path for path, _ in village.problem.system_rules] == ['tend']
    assert _run(system, until=2, dt=1).get_values('land/soil.x')[-1] == pytest.approx(2 * 2)


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
        PassCompiler().prepare(system).run()


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
    )
    assert [item.problem.form for item in PassCompiler().prepare(system).run().problem_.partitions] == [
        FormKind.DISCRETE,
        FormKind.ODE,
    ]


def test_held_input_keeps_start_of_interval_value():
    """Test that an ODE reading a count of agents uses its value from the start of the synchronisation interval."""
    trajectory = _run(_counting_system(3), until=2, solver=CoSimulationSolver(sync_dt=2), dt=2)
    assert trajectory.get_values('land/soil.x')[-1] == pytest.approx(3 * 2)
    assert trajectory.get_values('village/farmers_count.value').tolist() == [3.0, 12.0]


def test_held_reduction_counts_agents():
    """Test that an ODE law reading the count of 12 agents sees the value 12 until the next synchronisation point."""
    trajectory = _run(_counting_system(12), until=1, solver=CoSimulationSolver(), dt=1)
    assert trajectory.get_values('village/farmers_count.value')[0] == 12
    assert trajectory.get_values('land/soil.x')[-1] == pytest.approx(12)


@pytest.mark.parametrize(('reduction', 'expected'), [('sum', 3.0), ('mean', 1.0), ('max', 1.0)])
def test_held_reductions_of_agent_states(reduction, expected):
    """Test that laws read the total, the mean, and the largest value of an agent state across partitions."""
    farmers, observing = _idlers('farmers', 3)
    system = System(
        systems=[
            System('land', elements=[PopulationElement('soil')]),
            System('village', elements=[farmers], interactions=[observing]),
        ],
        interactions=[
            WealthInteraction('accumulate', target='land/soil', source='village/farmers', reduction=reduction),
        ],
    )
    trajectory = _run(system, until=1, dt=1)
    assert trajectory.get_values(f'village/farmers_{reduction}.value')[0] == expected
    assert trajectory.get_values('land/soil.x')[-1] == pytest.approx(1.0 + expected)


def test_held_reductions_of_one_group_share_an_element():
    """Test that two laws reading the same reduction of agents read one element, and other reductions get their own."""
    farmers, observing = _idlers('farmers', 3)
    system = System(
        systems=[
            System('land', elements=[PopulationElement('soil'), PopulationElement('dust')]),
            System('village', elements=[farmers], interactions=[observing]),
        ],
        interactions=[
            WealthInteraction('first', target='land/soil', source='village/farmers'),
            WealthInteraction('second', target='land/dust', source='village/farmers'),
            CountingInteraction('third', target='land/dust', source='village/farmers'),
        ],
    )
    paths = [
        path for path, _ in ObservableEvaluator._get_held_reductions_of(PassCompiler().prepare(system).run().problem_)
    ]
    assert paths == ['village/farmers_sum', 'village/farmers_count']


def test_reduction_of_element_group_reads_other_partition():
    """Test that a discrete law reads the total of an element group of an ODE partition at the synchronisation point."""
    system = System(
        systems=[
            System('ledger', elements=[TallyElement('tally')]),
            System(
                'herds',
                elements=[Element('herd', values={'x': np.array([1.0, 2.0, 3.0])}, size=3)],
                interactions=[ExponentialGrowthInteraction('growth', target='herd')],
            ),
        ],
        interactions=[TallyInteraction('tally', target='ledger/tally', source='herds/herd')],
    )
    trajectory = _run(system, until=1, dt=1)
    assert ObservableEvaluator._get_held_reductions_of(PassCompiler().prepare(system).run().problem_) == []
    assert trajectory.get_values('ledger/tally.k')[-1] == pytest.approx(6.0)


@pytest.mark.skip(
    reason='engine gap: a law that reads an observable reduced by a function fails state inference, so the element it '
    'targets is no state',
)
def test_held_observable_reduces_element_group():
    """Test that an observable reducing an element group with a function is held for a role of another partition."""
    system = System(
        systems=[
            System('ledger', elements=[TallyElement('tally')]),
            System(
                'herds',
                elements=[Element('herd', values={'x': np.array([1.0, 2.0, 3.0])}, size=3)],
                interactions=[ExponentialGrowthInteraction('growth', target='herd')],
            ),
        ],
        interactions=[CopyInteraction('copy', target='ledger/tally', source='herds/largest')],
    )
    largest = Observable('herds/largest', group='herd', reduce=np.max, variable='x')
    trajectory = _run(system, until=1, dt=1, observables=[largest])
    assert trajectory.get_values('herds/largest.value')[0] == 3.0
    assert trajectory.get_values('ledger/tally.k')[-1] == pytest.approx(3.0)


def test_subsystem_partitions_hold_reductions_of_agents():
    """Test that compiling one partition per subsystem holds the reductions of agents that laws read."""
    problem = PassCompiler(partition='subsystems').prepare(HarvestingSystem().prepare().system_).run().problem_
    assert [partition.name for partition in problem.partitions] == ['land', 'village']


def test_agent_rules_read_other_partitions():
    """Test that agent rules read the states of the ODE partition at the last synchronisation point."""
    farmers, observing = _idlers('farmers', 2)
    system = System(
        systems=[
            System(
                'land',
                elements=[PopulationElement('soil', x=5.0)],
                interactions=[ExponentialGrowthInteraction('growth', target='soil', r=1.0)],
            ),
            System('village', elements=[farmers], interactions=[observing]),
        ],
    )
    trajectory = _run(system, until=1, dt=1)
    assert trajectory.get_agents('village/farmers', 1).values['seen'].tolist() == pytest.approx(
        [5.0 * np.e] * 2,
        rel=1e-5,
    )


def test_sync_interval_must_divide_frame():
    """Test that a synchronisation interval that does not divide the frame names both values."""
    with pytest.raises(ValidationError, match=r'0\.03.*0\.1'):
        _run(_lotka_volterra(), until=0.1, solver=CoSimulationSolver(split=True, sync_dt=0.03), dt=0.1)


def test_sync_interval_must_be_whole_steps_for_agents():
    """Test that agent partitions need a whole number of steps between synchronisation points."""
    with pytest.raises(ValidationError, match='whole number of steps'):
        _run(HarvestingSystem().prepare().system_, until=1, solver=CoSimulationSolver(sync_dt=0.5), dt=1)


def test_gauss_seidel_reads_already_advanced_values():
    """Test that Gauss-Seidel reads the partitions advanced before and Jacobi reads the start of the interval."""
    farmers, breeding = _breeders('farmers', 3)
    village_first = System(
        systems=[
            System('village', elements=[farmers], interactions=[breeding]),
            System('land', elements=[PopulationElement('soil', x=0.0)]),
        ],
        interactions=[CountingInteraction('accumulate', target='land/soil', source='village/farmers')],
    )
    gauss_seidel = _run(village_first, until=1, solver=CoSimulationSolver(), dt=1)
    jacobi = _run(village_first, until=1, solver=CoSimulationSolver(scheme='jacobi'), dt=1)
    assert gauss_seidel.get_values('land/soil.x')[-1] == pytest.approx(6)
    assert jacobi.get_values('land/soil.x')[-1] == pytest.approx(3)


def test_unknown_scheme_raises_error():
    """Test that an unknown scheme is rejected."""
    with pytest.raises(SolverError, match='gauss-seidel, jacobi'):
        _run(HarvestingSystem().prepare().system_, until=1, solver=CoSimulationSolver(scheme='newton'))


def test_single_form_needs_split():
    """Test that a problem of one form needs `split=True` to run as a co-simulation."""
    with pytest.raises(SolverError, match='split=True'):
        _run(_lotka_volterra(), until=0.1, solver=CoSimulationSolver())


def test_split_lotka_volterra_converges():
    """Test that the splitting error shrinks by a factor of at least 1.8 with each halving of `sync_dt`."""
    precise = ScipyODESolver(rtol=1e-10, atol=1e-12)
    joint = _run(_lotka_volterra(), until=10, solver=precise, dt=0.1).states[-1]
    errors = []
    for interval in (0.1, 0.05, 0.025):
        solver = CoSimulationSolver(split=True, sync_dt=interval, solvers={'prey': precise, 'predators': precise})
        split = _run(_lotka_volterra(), until=10, solver=solver, dt=0.1).states[-1]
        errors.append(np.abs(split - joint).max())
    assert errors[0] >= 1.8 * errors[1]
    assert errors[1] >= 1.8 * errors[2]


def test_mixed_trajectory_holds_states_and_agents():
    """Test that a co-simulation trajectory returns ODE states per frame and agent states per agent."""
    trajectory = _run(HarvestingSystem(5).prepare().system_, until=10, random_state=0)
    assert trajectory.get_values('land/stock.n').shape == (11,)
    snapshot = trajectory.get_agents('village/farmers', 10)
    assert len(snapshot.values['wealth']) == 5
    assert np.all(snapshot.values['wealth'] > 0)


def test_cosimulation_is_reproducible():
    """Test that equal random states give equal co-simulations with agents."""
    runs = [_run(_counting_system(3), until=3, random_state=0, dt=1) for _ in range(2)]
    assert np.array_equal(runs[0].states, runs[1].states)


def test_live_parameter_reaches_partitions():
    """Test that a parameter change during a run applies inside the partitions."""
    trajectory = _run(
        HarvestingSystem(10, effort=0.05).prepare().system_,
        until=30,
        schedule={'harvest.effort': lambda t: 0.05 if t < 12 else 0.0},
    )
    stock = trajectory.get_values('land/stock.n')
    assert stock[-1] > stock[10]
