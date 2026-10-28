"""Test co-simulation: partitioning, held inputs, held reductions, agent inputs, scheduling, and convergence."""

import time
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.errors import IncompatibleLawsError, SolverError, ValidationError
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.laws.utils import count, maximum, mean, total
from skcomplex.core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system, partition_system
from skcomplex.engine.compile._observables import ObservableEvaluator
from skcomplex.engine.problems import Form
from skcomplex.engine.runtime import RealTimeClock, Simulation
from skcomplex.engine.solvers import CoSimulationSolver, ScipyODESolver
from skcomplex.models.biology.ecology import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
    PredationInteraction,
)
from skcomplex.models.economics.markets import CobwebInteraction, MarketElement
from skcomplex.models.economics.resources import harvesting


class BreederElement(Element):
    """Agent that creates one offspring every step."""

    def _get_laws(self, s):
        """Return the breeding rule."""
        return [RuleLaw(self.breed)]

    def breed(self, agent, model):
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

    def observe(self, agent, model):
        """Record the fertility of the soil at the last synchronisation point."""
        agent.seen = model.inputs['land/soil.n']


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
        return [RateLaw(s.target.n, count(s.source))]


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
        return [RateLaw(s.target.n, REDUCTIONS[self.reduction](s.source.wealth))]


class TallyInteraction(Interaction):
    """Update that sets a tally to the total of a group, `n <- total(source.n)`."""

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
        return [UpdateLaw(s.target.k, total(s.source.n))]


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
        return [RateLaw(s.soil.n, s.rate), RuleLaw(self.tend)]

    def tend(self, model):
        """Do nothing for the agents."""
        del model


def _lotka_volterra():
    """Return a Lotka-Volterra system split into a prey and a predator subsystem."""
    return System(
        systems=[
            System(
                'prey',
                elements=[PopulationElement('rabbits', n=40)],
                interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.1, K=1000)],
            ),
            System(
                'predators',
                elements=[PopulationElement('foxes', n=9)],
                interactions=[MortalityInteraction('death', target='foxes', d=0.4)],
            ),
        ],
        interactions=[PredationInteraction('hunt', predator='predators/foxes', prey='prey/rabbits', a=0.4, b=0.1)],
    )


def _counting_system(agents=3):
    """Return land first, then a village of breeders whose count a land rate reads."""
    return System(
        systems=[
            System('land', elements=[PopulationElement('soil', n=0.0)]),
            System('village', elements=[GroupElement('farmers', element=BreederElement, size=agents, variable=True)]),
        ],
        interactions=[CountingInteraction('accumulate', target='land/soil', source='village/farmers')],
    )


def test_agents_and_ode_split_into_two_partitions():
    """Test that agents and differential equations in sibling subsystems give an agent and an ODE partition."""
    problem = compile_system(harvesting())
    assert problem.form == Form.COUPLED
    assert [(item.name, item.problem.form) for item in problem.partitions] == [
        ('land', Form.ODE),
        ('village', Form.AGENT),
    ]


def test_laws_join_the_partition_of_their_target():
    """Test that one interaction's rate and rule join the partitions of their targets, the rate counted per farmer."""
    system = System(
        systems=[
            System('land', elements=[PopulationElement('soil', n=0.0)]),
            System('village', elements=[GroupElement('farmers', element=IdleElement, size=2)]),
        ],
        interactions=[TendingInteraction('tend', soil='land/soil', farmers='village/farmers')],
    )
    land, village = compile_system(system).partitions
    assert (land.problem.form, village.problem.form) == (Form.ODE, Form.AGENT)
    assert [path for path, _ in village.problem.model_rules] == ['tend']
    assert Simulation(system, dt=1).check().run(until=2).trajectory_.values('land/soil.n')[-1] == pytest.approx(2 * 2)


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
        compile_system(system)


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
    assert [item.problem.form for item in compile_system(system).partitions] == [Form.DISCRETE, Form.ODE]


def test_held_input_keeps_start_of_interval_value():
    """Test that an ODE reading a count of agents uses its value from the start of the synchronisation interval."""
    trajectory = (
        Simulation(_counting_system(3), solver=CoSimulationSolver(sync_dt=2), dt=2)
        .check()
        .run(until=2)
        .trajectory_
    )
    assert trajectory.values('land/soil.n')[-1] == pytest.approx(3 * 2)
    assert trajectory.values('village/farmers_count.value').tolist() == [3.0, 12.0]


def test_held_reduction_counts_agents():
    """Test that an ODE law reading the count of 12 agents sees the value 12 until the next synchronisation point."""
    trajectory = Simulation(_counting_system(12), solver=CoSimulationSolver(), dt=1).check().run(until=1).trajectory_
    assert trajectory.values('village/farmers_count.value')[0] == 12
    assert trajectory.values('land/soil.n')[-1] == pytest.approx(12)


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
    trajectory = Simulation(system, dt=1).check().run(until=1).trajectory_
    assert trajectory.values(f'village/farmers_{reduction}.value')[0] == expected
    assert trajectory.values('land/soil.n')[-1] == pytest.approx(1.0 + expected)


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
    paths = [path for path, _ in ObservableEvaluator._held_reductions_of(compile_system(system))]
    assert paths == ['village/farmers_sum', 'village/farmers_count']


def test_reduction_of_element_group_reads_other_partition():
    """Test that a discrete law reads the total of an element group of an ODE partition at the synchronisation point."""
    system = System(
        systems=[
            System('ledger', elements=[TallyElement('tally')]),
            System(
                'herds',
                elements=[GroupElement('herd', element=PopulationElement, size=3, values={'n': [1.0, 2.0, 3.0]})],
                interactions=[ExponentialGrowthInteraction('growth', target='herd')],
            ),
        ],
        interactions=[TallyInteraction('tally', target='ledger/tally', source='herds/herd')],
    )
    trajectory = Simulation(system, dt=1).check().run(until=1).trajectory_
    assert ObservableEvaluator._held_reductions_of(compile_system(system)) == []
    assert trajectory.values('ledger/tally.k')[-1] == pytest.approx(6.0)


def test_held_observable_reduces_element_group():
    """Test that an observable reducing an element group with a function is held for a role of another partition."""
    system = System(
        systems=[
            System('ledger', elements=[TallyElement('tally')]),
            System(
                'herds',
                elements=[GroupElement('herd', element=PopulationElement, size=3, values={'n': [1.0, 2.0, 3.0]})],
                interactions=[ExponentialGrowthInteraction('growth', target='herd')],
            ),
        ],
        interactions=[CopyInteraction('copy', target='ledger/tally', source='herds/largest')],
    )
    largest = ObservableSpec('herds/largest', group='herd', reduce=np.max, variable='n')
    trajectory = Simulation(system, dt=1, observables=[largest]).check().run(until=1).trajectory_
    assert trajectory.values('herds/largest.value')[0] == 3.0
    assert trajectory.values('ledger/tally.k')[-1] == pytest.approx(3.0)


def test_partition_system_rejects_reductions_of_agents():
    """Test that compiling partitions without validation names the reduction of agents that it cannot hold."""
    with pytest.raises(ValidationError, match='only a co-simulation computes'):
        partition_system(harvesting())


def test_agent_rules_read_other_partitions():
    """Test that agent rules read the states of the ODE partition at the last synchronisation point."""
    system = System(
        systems=[
            System(
                'land',
                elements=[PopulationElement('soil', n=5.0)],
                interactions=[ExponentialGrowthInteraction('growth', target='soil', r=1.0)],
            ),
            System('village', elements=[GroupElement('farmers', element=IdleElement, size=2)]),
        ],
    )
    trajectory = Simulation(system, dt=1).check().run(until=1).trajectory_
    assert trajectory.agents('village/farmers', 1).values['seen'].tolist() == pytest.approx([5.0 * np.e] * 2, rel=1e-5)


def test_sync_interval_must_divide_frame():
    """Test that a synchronisation interval that does not divide the frame names both values."""
    with pytest.raises(ValidationError, match=r'0\.03.*0\.1'):
        (
            Simulation(_lotka_volterra(), solver=CoSimulationSolver(split=True, sync_dt=0.03), dt=0.1)
            .check()
            .run(until=0.1)
        )


def test_sync_interval_must_be_whole_steps_for_agents():
    """Test that agent partitions need a whole number of steps between synchronisation points."""
    with pytest.raises(ValidationError, match='whole number of steps'):
        Simulation(harvesting(), solver=CoSimulationSolver(sync_dt=0.5), dt=1).check().run(until=1)


def test_gauss_seidel_reads_already_advanced_values():
    """Test that Gauss-Seidel reads the partitions advanced before and Jacobi reads the start of the interval."""
    village_first = System(
        systems=[
            System('village', elements=[GroupElement('farmers', element=BreederElement, size=3, variable=True)]),
            System('land', elements=[PopulationElement('soil', n=0.0)]),
        ],
        interactions=[CountingInteraction('accumulate', target='land/soil', source='village/farmers')],
    )
    gauss_seidel = Simulation(village_first, solver=CoSimulationSolver(), dt=1).check().run(until=1).trajectory_
    jacobi = (
        Simulation(village_first, solver=CoSimulationSolver(scheme='jacobi'), dt=1)
        .check()
        .run(until=1)
        .trajectory_
    )
    assert gauss_seidel.values('land/soil.n')[-1] == pytest.approx(6)
    assert jacobi.values('land/soil.n')[-1] == pytest.approx(3)


def test_unknown_scheme_raises_error():
    """Test that an unknown scheme is rejected."""
    with pytest.raises(SolverError, match='gauss-seidel, jacobi'):
        Simulation(harvesting(), solver=CoSimulationSolver(scheme='newton')).check().run(until=1)


def test_single_form_needs_split():
    """Test that a problem of one form needs `split=True` to run as a co-simulation."""
    with pytest.raises(SolverError, match='split=True'):
        Simulation(_lotka_volterra(), solver=CoSimulationSolver()).check().run(until=0.1)


def test_split_lotka_volterra_converges():
    """Test that the splitting error shrinks by a factor of at least 1.8 with each halving of `sync_dt`."""
    precise = ScipyODESolver(rtol=1e-10, atol=1e-12)
    joint = Simulation(_lotka_volterra(), solver=precise, dt=0.1).check().run(until=10).trajectory_.states[-1]
    errors = []
    for interval in (0.1, 0.05, 0.025):
        solver = CoSimulationSolver(split=True, sync_dt=interval, solvers={'prey': precise, 'predators': precise})
        split = Simulation(_lotka_volterra(), solver=solver, dt=0.1).check().run(until=10).trajectory_.states[-1]
        errors.append(np.abs(split - joint).max())
    assert errors[0] >= 1.8 * errors[1]
    assert errors[1] >= 1.8 * errors[2]


def test_mixed_trajectory_holds_states_and_agents():
    """Test that a co-simulation trajectory returns ODE states per frame and agent states per agent."""
    trajectory = Simulation(harvesting(5), random_state=0).check().run(until=10).trajectory_
    assert trajectory.values('land/stock.n').shape == (11,)
    snapshot = trajectory.agents('village/farmers', 10)
    assert len(snapshot.values['wealth']) == 5
    assert np.all(snapshot.values['wealth'] > 0)


def test_cosimulation_is_reproducible():
    """Test that equal random states give equal co-simulations with agents."""
    runs = [Simulation(_counting_system(3), random_state=0, dt=1).check().run(until=3).trajectory_ for _ in range(2)]
    assert np.array_equal(runs[0].states, runs[1].states)


def test_live_parameter_reaches_partitions():
    """Test that a live parameter change applies inside the partitions."""
    simulation = Simulation(harvesting(10, effort=0.05), clock=RealTimeClock(speed=20.0)).check().start(until=30)
    time.sleep(0.6)
    simulation.update_parameters({'harvest.effort': 0.0})
    simulation.wait()
    stock = simulation.trajectory_.values('land/stock.n')
    assert stock[-1] > stock[10]
