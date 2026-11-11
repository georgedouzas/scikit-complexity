"""Test the `_observables` module of `skcomplex.core.engine.compile`."""

import dataclasses
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import I, exp

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, RuleLaw
from skcomplex.core.components.laws.utils import distributed_delay, inverse_helmholtz, mean, total
from skcomplex.core.components.spaces import LatticeSpace, StaticTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile._observables import ObservableEvaluator
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import RoleError, ValidationError
from skcomplex.core.specs import (
    ObservableSpec,
    ParameterSpec,
    RoleSpec,
    StateSpec,
    VisualSpec,
)
from skcomplex.systems import all_systems
from skcomplex.systems.biology.common.populations import PopulationElement
from skcomplex.systems.economics.resources import HarvestingSystem
from skcomplex.systems.engineering.mechatronics import QuarterCarSystem
from skcomplex.systems.physics.waves import LatticeGaugeFieldsSystem

from ....conftest import GrowingPopulationElement, RotorElement, _total, _two_populations
from .conftest import PhaseElement

FIELD_WEIGHTS = np.ones((4, 4)) - np.eye(4)


class ObservedValueElement(Element):
    """Observed value element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y


def _observed(system_class):
    """Return a system with its observables, and the paths of those that laws read at the current state."""
    system = system_class().prepare().system_
    observables = list(system_class().get_run_settings()['observables'])
    compiler = PassCompiler(observables=observables).prepare(system)
    if compiler.learning_ is not None:
        pytest.skip('A system that trains a nested system records no observables of its own.')
    compilation = compiler.compilation_
    nodes = {node.path: node for node in compilation.nodes}
    found, blocked = [], set()
    for observable in observables:
        scope = observable.label.rpartition('/')[0]
        group = nodes.get(f'{scope}/{observable.group}' if scope else observable.group)
        agents = group is not None and group.is_agent
        unread = observable.group is not None and (group is None or agents or not isinstance(observable.reduce, str))
        if scope in blocked or observable.element is not None or observable.function is not None or unread:
            blocked.add(scope)
            continue
        found.append(observable.label)
    return system, observables, found


class ReadInteraction(Interaction):
    """Rate of a probe at the modulus of the value it reads from a source."""

    def __init__(
        self,
        label: str,
        *,
        probe: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.probe = probe
        self.source = source

    def _get_laws(self, s):
        """Return the rate of the probe."""
        return [RateLaw(s.probe.y, abs(s.source.value))]


def _double(s):
    """Return twice the complex state of the oscillator."""
    return 2 * s.osc.z


def _size(s):
    """Return the modulus of the complex state of the oscillator."""
    return abs(s.osc.z)


class SaverElement(Element):
    """Agent whose wealth grows by one every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth

    def _get_laws(self, s):
        """Return the saving rule."""
        return [RuleLaw(self.save)]

    def save(self, agent, system):
        """Add one to the wealth."""
        del system
        agent.wealth += 1


class InflowInteraction(Interaction):
    """Inflow into the population of a target at the value of a source."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        source: Annotated[object, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _get_laws(self, s):
        """Return the inflow."""
        return [RateLaw(s.target.n, s.source.value)]


class DecayingElement(Element):
    """Decaying element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the decay."""
        return [RateLaw(s.x, -s.x)]


class SmoothedElement(Element):
    """Stock that relaxes toward 2 as it is perceived over a perception time."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 1.0,
        perception_time: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.perception_time = perception_time

    def _get_laws(self, s):
        """Return the relaxation toward the perceived stock."""
        return [RateLaw(s.n, 2.0 - _perceived(s))]


def _phase_vector(member):
    """Return the unit complex number of a member's phase."""
    return exp(I * member.theta)


def _modulus(s):
    """Return the modulus of the complex mean field."""
    return abs(s.z)


ORDER = [ObservableSpec('z', _phase_vector, group='osc'), ObservableSpec('order', _modulus)]


def _share(s):
    """Return the share of the first population in the total."""
    return s.a.n / s.total


def _volume(s):
    """Return twice the population of the market."""
    return 2 * s.a.n


def _herd(s):
    """Return the symbol of a group member, outside a reduction."""
    return s.herd.n


def _timed(s):
    """Return the population times the time."""
    return s.a.n * s.t


def _member_timed(member):
    """Return a member's population plus the time."""
    return member.n + member.t


def _local_field_energy(member):
    """Return a member's state times the weighted total of the states of the others."""
    return member.x * total(member.x, weights=FIELD_WEIGHTS)


def _deviation(member):
    """Return a member's deviation from the group mean."""
    return member.x - mean(member.x)


def _deviation_from_other(member):
    """Return a member's state minus the mean of a symbol that is not one of its group."""
    return member.x - mean(member.x, over=symbol('elsewhere'))


def _perceived(s):
    """Return the stock of an element smoothed over its perception time."""
    return distributed_delay(s.n, conserved=False, time=s.perception_time)


def _smoothed_population(s):
    """Return the population smoothed over one unit of time."""
    return distributed_delay(s.n, conserved=False, time=1.0)


def _oscillators():
    """Return oscillators whose order parameter `ORDER` builds from a complex mean."""
    return CompositeSystem(
        elements=[
            GroupElement(
                'osc',
                element=RotorElement,
                size=50,
                values={'theta': np.linspace(0, 1, 50), 'omega': np.linspace(0.5, 1.5, 50)},
            ),
        ],
    )


def _decaying_group(*observables, until=0.0):
    """Return the trajectory of a decaying group of 4 members with observables until a time."""
    system = CompositeSystem(
        elements=[GroupElement('u', element=DecayingElement, size=4, values={'x': np.arange(1.0, 5.0)})],
    )
    return simulate(system, until=until, dt=0.5, observables=list(observables))


@pytest.mark.parametrize('system_class', [system for _, system in all_systems()], ids=lambda system: system.__name__)
def test_reductions_agree_with_observables(system_class):
    """Test that laws reading a system's observables at a state read the values the observable evaluator records."""
    observed, observables, paths = _observed(system_class)
    if OperatorCompiler._contains_operator(observed):
        pytest.skip('An operator system cannot share a co-simulation with classical readers.')
    if isinstance(observed.time, StaticTime):
        pytest.skip('A static system has no time axis, so rate laws of a continuous system cannot read its values.')
    if 'distribution' in system_class.topics:
        pytest.skip('The observables of a distribution take its moments, which laws read only inside its reductions.')
    if not paths:
        pytest.skip('The system has no observable that laws read at the current state.')
    observed.label = 'system'
    probes = [ObservedValueElement(f'probe_{index}') for index in range(len(paths))]
    readers = [
        ReadInteraction(f'read_{index}', probe=f'probe_{index}', source=f'system/{path}')
        for index, path in enumerate(paths)
    ]
    nested = [dataclasses.replace(observable, label=f'system/{observable.label}') for observable in observables]
    system = CompositeSystem(elements=probes, systems=[observed], interactions=readers)
    problem = PassCompiler(backend=NumpyBackend(), observables=nested).prepare(system).run().problem_
    reading = next((item.problem for item in problem.partitions if 'probe_0' in item.owners), problem)
    x, p = problem.initial_state, problem.initial_parameters
    rates = reading.drift(0.0, x, p)
    frame = x if problem.algebraic is None else problem.algebraic(0.0, x, p)
    recorded = ObservableEvaluator(problem).prepare().evaluate(0.0, frame, p, agents_of=lambda group: [])
    for index, path in enumerate(paths):
        value = rates[problem.states.get_entry(f'probe_{index}.y').start]
        assert value == pytest.approx(abs(recorded[f'system/{path}']), rel=1e-12, abs=1e-12)


def test_laws_read_an_observable_of_time():
    """Test that a law reads an observable written with the time of the system, as the evaluator records it."""
    observed = QuarterCarSystem().prepare().system_
    observed.label = 'system'
    observables = [
        dataclasses.replace(item, label=f'system/{item.label}')
        for item in QuarterCarSystem().get_run_settings()['observables']
    ]
    reader = ReadInteraction('read', probe='probe', source='system/energy')
    system = CompositeSystem(elements=[ObservedValueElement('probe')], systems=[observed], interactions=[reader])
    problem = PassCompiler(backend=NumpyBackend(), observables=observables).prepare(system).run().problem_
    reading = next((item.problem for item in problem.partitions if 'probe' in item.owners), problem)
    x, p = problem.initial_state, problem.initial_parameters
    rates = reading.drift(1.0, x, p)
    recorded = ObservableEvaluator(problem).prepare().evaluate(1.0, x, p, agents_of=lambda group: [])
    assert recorded['system/energy'] > 0
    assert rates[problem.states.get_entry('probe.y').start] == pytest.approx(recorded['system/energy'], rel=1e-12)


def test_function_observable_reads_the_names_of_its_own_system():
    """Test that a function observable of a nested system reads its values by their names within that system."""
    system = LatticeGaugeFieldsSystem(size=4).prepare().system_
    observables = LatticeGaugeFieldsSystem(size=4).get_run_settings()['observables']
    alone = PassCompiler(backend=NumpyBackend(), observables=observables).prepare(system).run().problem_
    expected = ObservableEvaluator(alone).prepare().evaluate(0.0, alone.initial_state, alone.initial_parameters)
    system.label = 'system'
    nested = [dataclasses.replace(item, label=f'system/{item.label}') for item in observables]
    problem = (
        PassCompiler(backend=NumpyBackend(), observables=nested)
        .prepare(CompositeSystem(systems=[system]))
        .run()
        .problem_
    )
    recorded = ObservableEvaluator(problem).prepare().evaluate(0.0, problem.initial_state, problem.initial_parameters)
    assert recorded['system/mean_plaquette'] == pytest.approx(expected['mean_plaquette'])


class WealthyElement(Element):
    """Agent that keeps its wealth."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth

    def _get_laws(self, s):
        """Return a rule that keeps the wealth."""
        return [RuleLaw(self.keep)]

    def keep(self, agent, system):
        """Keep the wealth."""
        del agent, system


class MixedReductionInteraction(Interaction):
    """Rate that reads a reduction of agents mixed with other symbols."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.first = first
        self.second = second

    def _get_laws(self, s):
        """Return a rate that reads the reduction the interaction tests."""
        return [RateLaw(s.target.x, total(s.first.wealth + s.second.wealth))]


class ForeignReductionInteraction(Interaction):
    """Rate that reduces agents with a state of a single element."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        first: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.first = first

    def _get_laws(self, s):
        """Return a rate that reduces the wealth of the agents times the target."""
        return [RateLaw(s.target.x, total(s.first.wealth * s.target.x))]


def _village(interaction):
    """Return a land and a village of two agent groups, coupled by an interaction."""
    village = CompositeSystem(
        'village',
        elements=[
            GroupElement('a', element=WealthyElement, size=2),
            GroupElement('b', element=WealthyElement, size=2),
        ],
    )
    land = CompositeSystem('land', elements=[PopulationElement('soil')])
    return CompositeSystem(systems=[land, village], interactions=[interaction])


def test_held_reduction_reads_one_group():
    """Test that a reduction of the agents of two groups is reported as a role error."""
    interaction = MixedReductionInteraction('mix', target='land/soil', first='village/a', second='village/b')
    (issue,) = PassCompiler().prepare(_village(interaction)).report_.errors
    assert issue.error is RoleError
    assert 'village/a and village/b' in issue.message


def test_held_reduction_reads_only_agents():
    """Test that a reduction of agents that reads a state of a single element names the symbol it cannot read."""
    interaction = ForeignReductionInteraction('foreign', target='land/soil', first='village/a')
    (issue,) = PassCompiler().prepare(_village(interaction)).report_.errors
    assert issue.code == 'reduction'
    assert 'x_land_soil' in issue.message


def _field_streamfunction(s):
    """Return the streamfunction of a site, the inversion of its vorticity with walls at rest."""
    return inverse_helmholtz(s.u, 0, ('fixed', 0))


def _field_values(values):
    """Return the values of every site."""
    return np.array(values)


def test_observable_evaluator_lowers_field_operators():
    """Test that an observable of a field applies a field inversion at every site, as a law of the field would."""

    class SiteElement(Element):
        def __init__(self, label, u: Annotated[float, StateSpec()] = 0.0):
            super().__init__(label)
            self.u = u

    values = np.sin(np.pi * np.arange(1, 9) / 9)
    space = LatticeSpace(8, boundary='fixed')
    field = FieldElement('field', element=SiteElement, space=space, values={'u': values}, boundary={'u': ('fixed', 0)})
    observable = ObservableSpec('psi', _field_streamfunction, group='field', reduce=_field_values)
    problem = PassCompiler(observables=[observable]).prepare(CompositeSystem(elements=[field])).run().problem_
    evaluator = ObservableEvaluator(problem).prepare()
    psi = evaluator.evaluate(0.0, problem.initial_state, problem.initial_parameters)['psi']
    stencil = np.diff(np.concatenate([[0.0], psi, [0.0]]), 2)
    np.testing.assert_allclose(stencil, values, atol=1e-12)


def test_observable_evaluator_get_namespace_of():
    """Test `ObservableEvaluator._get_namespace_of` on a small case."""
    system = CompositeSystem(elements=[PopulationElement('rabbits')])
    problem = PassCompiler().prepare(system).run().problem_
    assert str(ObservableEvaluator._get_namespace_of(problem).rabbits.x) == 'x_rabbits'


def test_observable_evaluator_get_held_reductions_of():
    """Test `ObservableEvaluator._get_held_reductions_of` on a small case."""
    problem = PassCompiler().prepare(HarvestingSystem().prepare().system_).run().problem_
    assert [path for path, _ in ObservableEvaluator._get_held_reductions_of(problem)] == ['village/farmers_count']


def test_complex_observable_is_complex():
    """Test that an observable of a complex state is complex, and the observable of its modulus is real."""
    system = CompositeSystem(
        elements=[PhaseElement('osc', z=3 + 4j, omega=0.0)],
    )
    observables = [ObservableSpec('double', _double), ObservableSpec('size', _size)]
    trajectory = simulate(system, until=0.5, dt=0.5, observables=observables)
    assert trajectory.get_values('double')[0] == 6 + 8j
    assert trajectory.get_values('size')[0] == 5.0


def test_order_parameter_every_frame():
    """Test that the modulus of the mean of exp(i theta) is recorded at every frame between 0 and 1."""
    trajectory = simulate(_oscillators(), until=10, dt=0.5, observables=ORDER)
    order = trajectory.get_values('order')
    assert order.shape == (21,)
    assert np.all((order >= 0) & (order <= 1))
    expected = np.abs(np.exp(1j * np.asarray(trajectory.get_values('osc.theta'))).mean(axis=1))
    assert np.allclose(order, expected)


def test_expression_observable_of_elements():
    """Test that a system observable combines symbols of single elements and earlier observables."""
    observables = [ObservableSpec('total', _total), ObservableSpec('share', _share)]
    trajectory = simulate(_two_populations(), until=1, dt=0.5, observables=observables)
    assert np.allclose(trajectory.get_values('total'), trajectory.get_values('a.n') + trajectory.get_values('b.n'))
    assert np.allclose(trajectory.get_values('share'), trajectory.get_values('a.n') / trajectory.get_values('total'))


def test_observable_by_subsystem_path():
    """Test that an observable of a subsystem is recorded under the subsystem's path."""
    system = CompositeSystem(systems=[CompositeSystem('market', elements=[GrowingPopulationElement('a', n=3.0)])])
    trajectory = simulate(system, until=2, dt=1, observables=[ObservableSpec('market/volume', _volume)])
    assert trajectory.get_values('market/volume').tolist() == [6.0, 6.0, 6.0]


def test_agent_reduction_read_by_other_partition():
    """Test that the mean wealth of agents is recorded and read by an ODE in another partition."""
    system = CompositeSystem(
        systems=[
            CompositeSystem('village', elements=[GroupElement('savers', element=SaverElement, size=4)]),
            CompositeSystem('land', elements=[GrowingPopulationElement('soil', n=0.0)]),
        ],
        interactions=[InflowInteraction('inflow', target='land/soil', source='village/mean_wealth')],
    )
    observables = [ObservableSpec('village/mean_wealth', group='savers', variable='wealth')]
    trajectory = simulate(system, until=2, dt=1, observables=observables)
    assert trajectory.get_values('village/mean_wealth').tolist() == [1.0, 2.0, 3.0]
    assert trajectory.get_values('land/soil.n')[-1] == pytest.approx(2.0 + 3.0)


def test_observable_read_in_one_form_is_lowered():
    """Test that a law reading a symbolic observable of a system with one form reads it at the current state."""
    system = CompositeSystem(
        elements=[
            GroupElement('herd', element=GrowingPopulationElement, size=3, values={'n': [1.0, 2.0, 6.0]}),
            GrowingPopulationElement('soil'),
        ],
        interactions=[InflowInteraction('inflow', target='soil', source='mean_n')],
    )
    observables = [ObservableSpec('mean_n', group='herd', variable='n')]
    trajectory = simulate(system, until=1, dt=1.0, observables=observables)
    assert trajectory.get_values('soil.n')[-1] == pytest.approx(trajectory.get_values('soil.n')[0] + 3.0)


def test_observable_of_a_member_symbol_outside_a_reduction_is_rejected():
    """Test that an observable using a group member's symbol outside a reduction is rejected."""
    system = CompositeSystem(elements=[GroupElement('herd', element=GrowingPopulationElement, size=3)])
    with pytest.raises(AttributeError, match='herd'):
        simulate(system, until=1, observables=[ObservableSpec('bad', _herd)])


def test_simulation_checks_its_observables():
    """Test that a run rejects an observable whose expression is a lambda before compiling."""
    system = CompositeSystem(elements=[GrowingPopulationElement('a')])
    with pytest.raises(ValidationError, match="'total'"):
        simulate(system, until=1, observables=[ObservableSpec('total', lambda s: s.a.n)])


def test_observables_read_time():
    """Test that system observables and member expressions of reductions read the time as `s.t`."""
    system = CompositeSystem(
        elements=[
            GrowingPopulationElement('a', n=2.0),
            GroupElement('herd', element=GrowingPopulationElement, size=2, values={'n': [1.0, 3.0]}),
        ],
    )
    observables = [ObservableSpec('timed', _timed), ObservableSpec('member_timed', _member_timed, group='herd')]
    trajectory = simulate(system, until=2, dt=1.0, observables=observables)
    assert trajectory.get_values('timed').tolist() == [0.0, 2.0, 4.0]
    assert trajectory.get_values('member_timed').tolist() == [2.0, 3.0, 4.0]


def test_observable_reads_weighted_reduction_of_its_members():
    """Test that a member expression with a weighted total gives the local field energy at every frame."""
    trajectory = _decaying_group(ObservableSpec('energy', _local_field_energy, group='u', reduce='sum'), until=2)
    x = np.asarray(trajectory.get_values('u.x'))
    np.testing.assert_allclose(
        trajectory.get_values('energy'),
        np.einsum('fi,ij,fj->f', x, FIELD_WEIGHTS, x),
        rtol=1e-12,
    )


def test_observable_reads_plain_reduction_of_its_members():
    """Test that a member expression with a plain mean gives the variance of the group at every frame."""
    trajectory = _decaying_group(ObservableSpec('spread', _deviation, group='u', reduce='var'), until=2)
    np.testing.assert_allclose(
        trajectory.get_values('spread'),
        np.asarray(trajectory.get_values('u.x')).var(axis=1),
        rtol=1e-12,
    )


def test_observable_rejects_reduction_over_another_group():
    """Test that a reduction over a group other than the observable's raises the role error of laws."""
    with pytest.raises(RoleError, match=r'observable other reads a group other than u'):
        _decaying_group(ObservableSpec('other', _deviation_from_other, group='u', reduce='sum'))


def test_observable_of_an_element_records_its_expression_with_delays():
    """Test that an observable of an element's expression records it under the element and reads its delays."""
    observed = ObservableSpec('perceived', _perceived, element='tank')
    trajectory = simulate(CompositeSystem(elements=[SmoothedElement('tank')]), until=4, dt=0.5, observables=[observed])
    perceived = trajectory.get_values('tank.perceived')
    np.testing.assert_allclose(perceived, trajectory.get_values('tank.smoothing_delay1_0')[:, 0])
    assert perceived[0] == pytest.approx(1.0)
    assert perceived[-1] > perceived[0]


def test_observable_of_an_element_rejects_a_delay_no_law_uses():
    """Test that observing a delay that no law of the system reads raises an error that names the observable."""
    observed = ObservableSpec('smoothed', _smoothed_population, element='rabbits')
    with pytest.raises(ValidationError, match=r'rabbits\.smoothed'):
        simulate(
            CompositeSystem(elements=[GrowingPopulationElement('rabbits', n=2.0)]),
            until=1,
            observables=[observed],
        )
