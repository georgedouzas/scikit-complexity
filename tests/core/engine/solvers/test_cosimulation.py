"""Test co-simulation: partitioning, held inputs, held reductions, agent inputs, scheduling, and convergence."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.components.laws.utils import count, maximum, mean, total
from skcomplex.core.components.systems import System
from skcomplex.core.engine.compile._observables import ObservableEvaluator
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import CoSimulationSolver, ScipyODESolver
from skcomplex.core.errors import IncompatibleLawsError, SolverError, ValidationError
from skcomplex.core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.core.types import FormKind
from skcomplex.systems.biology.common.populations import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import PredationInteraction
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
from skcomplex.systems.economics.resources import HarvestingSystem


class BreederElement(Element):
    """Agent that creates one offspring every step."""

    def _get_laws(self, s):
        """Return the breeding rule."""
        return [RuleLaw(self.breed)]

    def breed(self, agent, system):
        """Create one offspring."""
        agent.create()


class IdleElement(Element):
    """Agent that keeps its wealth and records the soil it sees."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 1.0,
        seen: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth
        self.seen = seen

    def _get_laws(self, s):
        """Return the observing rule."""
        return [RuleLaw(self.observe)]

    def observe(self, agent, system):
        """Record the fertility of the soil at the last synchronisation point."""
        agent.seen = system.inputs['land/soil.x']


REDUCTIONS = {'sum': total, 'mean': mean, 'max': maximum}


class CountingInteraction(Interaction):
    """Rate equal to the number of members of a group, `dn/dt = count(source)`."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _get_laws(self, s):
        """Return the rate `count(source)`."""
        return [RateLaw(s.target.x, count(s.source))]


class WealthInteraction(Interaction):
    """Rate equal to a reduction of the wealth of a group, such as `dn/dt = total(source.wealth)`."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        reduction='sum',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.reduction = reduction

    def _get_laws(self, s):
        """Return the rate, the reduction of the wealth."""
        return [RateLaw(s.target.x, REDUCTIONS[self.reduction](s.source.wealth))]


class TallyInteraction(Interaction):
    """Update that sets a tally to the total of a group, `n <- total(source.x)`."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _get_laws(self, s):
        """Return the update to the total."""
        return [UpdateLaw(s.target.k, total(s.source.x))]


class CopyInteraction(Interaction):
    """Update that copies a held observable into a tally, `k <- value`."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _get_laws(self, s):
        """Return the update to the observable."""
        return [UpdateLaw(s.target.k, s.source.value)]


class TallyElement(Element):
    """Tally of a discrete subsystem."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k


class TendingInteraction(Interaction):
    """Interaction with a rate on a population and a rule on agents."""

    def __init__(
        self,
        label: str,
        *,
        soil: Annotated[str, RoleSpec()],
        farmers: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.soil = soil
        self.farmers = farmers
        self.rate = rate

    def _get_laws(self, s):
        """Return one rate law and one rule."""
        return [RateLaw(s.soil.x, s.rate), RuleLaw(self.tend)]

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
    return System(
        systems=[
            System('land', elements=[PopulationElement('soil', x=0.0)]),
            System('village', elements=[GroupElement('farmers', element=BreederElement, size=agents, variable=True)]),
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


def test_laws_join_the_partition_of_their_target():
    """Test that one interaction's rate and rule join the partitions of their targets, the rate counted per farmer."""
    system = System(
        systems=[
            System('land', elements=[PopulationElement('soil', x=0.0)]),
            System('village', elements=[GroupElement('farmers', element=IdleElement, size=2)]),
        ],
        interactions=[TendingInteraction('tend', soil='land/soil', farmers='village/farmers')],
    )
    land, village = PassCompiler().prepare(system).run().problem_.partitions
    assert (land.problem.form, village.problem.form) == (FormKind.ODE, FormKind.AGENT)
    assert [path for path, _ in village.problem.system_rules] == ['tend']
    assert simulate(system, until=2, dt=1).values('land/soil.x')[-1] == pytest.approx(2 * 2)


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
    trajectory = simulate(_counting_system(3), until=2, solver=CoSimulationSolver(sync_dt=2), dt=2)
    assert trajectory.values('land/soil.x')[-1] == pytest.approx(3 * 2)
    assert trajectory.values('village/farmers_count.value').tolist() == [3.0, 12.0]


def test_held_reduction_counts_agents():
    """Test that an ODE law reading the count of 12 agents sees the value 12 until the next synchronisation point."""
    trajectory = simulate(_counting_system(12), until=1, solver=CoSimulationSolver(), dt=1)
    assert trajectory.values('village/farmers_count.value')[0] == 12
    assert trajectory.values('land/soil.x')[-1] == pytest.approx(12)


@pytest.mark.parametrize(('reduction', 'expected'), [('sum', 3.0), ('mean', 1.0), ('max', 1.0)])
def test_held_reductions_of_agent_states(reduction, expected):
    """Test that laws read the total, the mean, and the largest value of an agent state across partitions."""
    system = System(
        systems=[
            System('land', elements=[PopulationElement('soil')]),
            System('village', elements=[GroupElement('farmers', element=IdleElement, size=3)]),
        ],
        interactions=[
            WealthInteraction('accumulate', target='land/soil', source='village/farmers', reduction=reduction),
        ],
    )
    trajectory = simulate(system, until=1, dt=1)
    assert trajectory.values(f'village/farmers_{reduction}.value')[0] == expected
    assert trajectory.values('land/soil.x')[-1] == pytest.approx(1.0 + expected)


def test_held_reductions_of_one_group_share_an_element():
    """Test that two laws reading the same reduction of agents read one element, and other reductions get their own."""
    system = System(
        systems=[
            System('land', elements=[PopulationElement('soil'), PopulationElement('dust')]),
            System('village', elements=[GroupElement('farmers', element=IdleElement, size=3)]),
        ],
        interactions=[
            WealthInteraction('first', target='land/soil', source='village/farmers'),
            WealthInteraction('second', target='land/dust', source='village/farmers'),
            CountingInteraction('third', target='land/dust', source='village/farmers'),
        ],
    )
    paths = [path for path, _ in ObservableEvaluator._held_reductions_of(PassCompiler().prepare(system).run().problem_)]
    assert paths == ['village/farmers_sum', 'village/farmers_count']


def test_reduction_of_element_group_reads_other_partition():
    """Test that a discrete law reads the total of an element group of an ODE partition at the synchronisation point."""
    system = System(
        systems=[
            System('ledger', elements=[TallyElement('tally')]),
            System(
                'herds',
                elements=[GroupElement('herd', element=PopulationElement, size=3, values={'x': [1.0, 2.0, 3.0]})],
                interactions=[ExponentialGrowthInteraction('growth', target='herd')],
            ),
        ],
        interactions=[TallyInteraction('tally', target='ledger/tally', source='herds/herd')],
    )
    trajectory = simulate(system, until=1, dt=1)
    assert ObservableEvaluator._held_reductions_of(PassCompiler().prepare(system).run().problem_) == []
    assert trajectory.values('ledger/tally.k')[-1] == pytest.approx(6.0)


def test_held_observable_reduces_element_group():
    """Test that an observable reducing an element group with a function is held for a role of another partition."""
    system = System(
        systems=[
            System('ledger', elements=[TallyElement('tally')]),
            System(
                'herds',
                elements=[GroupElement('herd', element=PopulationElement, size=3, values={'x': [1.0, 2.0, 3.0]})],
                interactions=[ExponentialGrowthInteraction('growth', target='herd')],
            ),
        ],
        interactions=[CopyInteraction('copy', target='ledger/tally', source='herds/largest')],
    )
    largest = ObservableSpec('herds/largest', group='herd', reduce=np.max, variable='x')
    trajectory = simulate(system, until=1, dt=1, observables=[largest])
    assert trajectory.values('herds/largest.value')[0] == 3.0
    assert trajectory.values('ledger/tally.k')[-1] == pytest.approx(3.0)


def test_subsystem_partitions_hold_reductions_of_agents():
    """Test that compiling one partition per subsystem holds the reductions of agents that laws read."""
    problem = PassCompiler(partition='subsystems').prepare(HarvestingSystem().prepare().system_).run().problem_
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
            System('village', elements=[GroupElement('farmers', element=IdleElement, size=2)]),
        ],
    )
    trajectory = simulate(system, until=1, dt=1)
    assert trajectory.agents('village/farmers', 1).values['seen'].tolist() == pytest.approx([5.0 * np.e] * 2, rel=1e-5)


def test_sync_interval_must_divide_frame():
    """Test that a synchronisation interval that does not divide the frame names both values."""
    with pytest.raises(ValidationError, match=r'0\.03.*0\.1'):
        simulate(_lotka_volterra(), until=0.1, solver=CoSimulationSolver(split=True, sync_dt=0.03), dt=0.1)


def test_sync_interval_must_be_whole_steps_for_agents():
    """Test that agent partitions need a whole number of steps between synchronisation points."""
    with pytest.raises(ValidationError, match='whole number of steps'):
        simulate(HarvestingSystem().prepare().system_, until=1, solver=CoSimulationSolver(sync_dt=0.5), dt=1)


def test_gauss_seidel_reads_already_advanced_values():
    """Test that Gauss-Seidel reads the partitions advanced before and Jacobi reads the start of the interval."""
    village_first = System(
        systems=[
            System('village', elements=[GroupElement('farmers', element=BreederElement, size=3, variable=True)]),
            System('land', elements=[PopulationElement('soil', x=0.0)]),
        ],
        interactions=[CountingInteraction('accumulate', target='land/soil', source='village/farmers')],
    )
    gauss_seidel = simulate(village_first, until=1, solver=CoSimulationSolver(), dt=1)
    jacobi = simulate(village_first, until=1, solver=CoSimulationSolver(scheme='jacobi'), dt=1)
    assert gauss_seidel.values('land/soil.x')[-1] == pytest.approx(6)
    assert jacobi.values('land/soil.x')[-1] == pytest.approx(3)


def test_unknown_scheme_raises_error():
    """Test that an unknown scheme is rejected."""
    with pytest.raises(SolverError, match='gauss-seidel, jacobi'):
        simulate(HarvestingSystem().prepare().system_, until=1, solver=CoSimulationSolver(scheme='newton'))


def test_single_form_needs_split():
    """Test that a problem of one form needs `split=True` to run as a co-simulation."""
    with pytest.raises(SolverError, match='split=True'):
        simulate(_lotka_volterra(), until=0.1, solver=CoSimulationSolver())


def test_split_lotka_volterra_converges():
    """Test that the splitting error shrinks by a factor of at least 1.8 with each halving of `sync_dt`."""
    precise = ScipyODESolver(rtol=1e-10, atol=1e-12)
    joint = simulate(_lotka_volterra(), until=10, solver=precise, dt=0.1).states[-1]
    errors = []
    for interval in (0.1, 0.05, 0.025):
        solver = CoSimulationSolver(split=True, sync_dt=interval, solvers={'prey': precise, 'predators': precise})
        split = simulate(_lotka_volterra(), until=10, solver=solver, dt=0.1).states[-1]
        errors.append(np.abs(split - joint).max())
    assert errors[0] >= 1.8 * errors[1]
    assert errors[1] >= 1.8 * errors[2]


def test_mixed_trajectory_holds_states_and_agents():
    """Test that a co-simulation trajectory returns ODE states per frame and agent states per agent."""
    trajectory = simulate(HarvestingSystem(5).prepare().system_, until=10, random_state=0)
    assert trajectory.values('land/stock.n').shape == (11,)
    snapshot = trajectory.agents('village/farmers', 10)
    assert len(snapshot.values['wealth']) == 5
    assert np.all(snapshot.values['wealth'] > 0)


def test_cosimulation_is_reproducible():
    """Test that equal random states give equal co-simulations with agents."""
    runs = [simulate(_counting_system(3), until=3, random_state=0, dt=1) for _ in range(2)]
    assert np.array_equal(runs[0].states, runs[1].states)


def test_live_parameter_reaches_partitions():
    """Test that a parameter change during a run applies inside the partitions."""
    trajectory = simulate(
        HarvestingSystem(10, effort=0.05).prepare().system_,
        until=30,
        schedule={'harvest.effort': lambda t: 0.05 if t < 12 else 0.0},
    )
    stock = trajectory.values('land/stock.n')
    assert stock[-1] > stock[10]
