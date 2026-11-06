"""Test the `_observables` module of `skcomplex.core.engine.compile`."""

import dataclasses
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, RuleLaw
from skcomplex.core.components.laws.utils import inverse_helmholtz, total
from skcomplex.core.components.spaces import LatticeSpace, StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile import ObservableEvaluator, PassCompiler
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.errors import RoleError
from skcomplex.core.specs import ObservableSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.systems import catalog, create_system, describe
from skcomplex.systems.biology.common.populations import PopulationElement


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


def _observed(name):
    """Return a system with its observables, and the paths of those that laws read at the current state."""
    system = create_system(name)
    observables = describe(name).observables_of(system)
    compilation = PassCompiler(observables=observables).prepare(system).compilation_
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


@pytest.mark.parametrize(
    'name',
    [listing.name for listing in catalog() if describe(listing.name).observables is not None],
)
def test_reductions_agree_with_observables(name):
    """Test that laws reading a system's observables at a state read the values the observable evaluator records."""
    observed, observables, paths = _observed(name)
    if OperatorCompiler._contains_operator(observed):
        pytest.skip('An operator system cannot share a co-simulation with classical readers.')
    if isinstance(observed.time, StaticTime):
        pytest.skip('A static system has no time axis, so rate laws of a continuous system cannot read its values.')
    if 'distribution' in describe(name).tags:
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
    system = System(elements=probes, systems=[observed], interactions=readers)
    problem = PassCompiler(backend=NumpyBackend(), observables=nested).prepare(system).run().problem_
    reading = next((item.problem for item in problem.partitions if 'probe_0' in item.owners), problem)
    x, p = problem.initial_state, problem.initial_parameters
    rates = reading.drift(0.0, x, p)
    frame = x if problem.algebraic is None else problem.algebraic(0.0, x, p)
    recorded = ObservableEvaluator(problem).prepare().evaluate(0.0, frame, p, agents_of=lambda group: [])
    for index, path in enumerate(paths):
        value = rates[problem.states.entry(f'probe_{index}.y').start]
        assert value == pytest.approx(abs(recorded[f'system/{path}']), rel=1e-12, abs=1e-12)


def test_laws_read_an_observable_of_time():
    """Test that a law reads an observable written with the time of the system, as the evaluator records it."""
    observed = create_system('quarter_car')
    observed.label = 'system'
    observables = [
        dataclasses.replace(item, label=f'system/{item.label}')
        for item in describe('quarter_car').observables_of(observed)
    ]
    reader = ReadInteraction('read', probe='probe', source='system/energy')
    system = System(elements=[ObservedValueElement('probe')], systems=[observed], interactions=[reader])
    problem = PassCompiler(backend=NumpyBackend(), observables=observables).prepare(system).run().problem_
    reading = next((item.problem for item in problem.partitions if 'probe' in item.owners), problem)
    x, p = problem.initial_state, problem.initial_parameters
    rates = reading.drift(1.0, x, p)
    recorded = ObservableEvaluator(problem).prepare().evaluate(1.0, x, p, agents_of=lambda group: [])
    assert recorded['system/energy'] > 0
    assert rates[problem.states.entry('probe.y').start] == pytest.approx(recorded['system/energy'], rel=1e-12)


def test_function_observable_reads_the_names_of_its_own_system():
    """Test that a function observable of a nested system reads its values by their names within that system."""
    system = create_system('lattice_gauge_fields', size=4)
    observables = describe('lattice_gauge_fields').observables_of(system)
    alone = PassCompiler(backend=NumpyBackend(), observables=observables).prepare(system).run().problem_
    expected = ObservableEvaluator(alone).prepare().evaluate(0.0, alone.initial_state, alone.initial_parameters)
    system.label = 'system'
    nested = [dataclasses.replace(item, label=f'system/{item.label}') for item in observables]
    problem = PassCompiler(backend=NumpyBackend(), observables=nested).prepare(System(systems=[system])).run().problem_
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
        return [RateLaw(s.target.n, total(s.first.wealth + s.second.wealth))]


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
        return [RateLaw(s.target.n, total(s.first.wealth * s.target.n))]


def _village(interaction):
    """Return a land and a village of two agent groups, coupled by an interaction."""
    village = System(
        'village',
        elements=[
            GroupElement('a', element=WealthyElement, size=2),
            GroupElement('b', element=WealthyElement, size=2),
        ],
    )
    land = System('land', elements=[PopulationElement('soil')])
    return System(systems=[land, village], interactions=[interaction])


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
    assert 'n_land_soil' in issue.message


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
    problem = PassCompiler(observables=[observable]).prepare(System(elements=[field])).run().problem_
    evaluator = ObservableEvaluator(problem).prepare()
    psi = evaluator.evaluate(0.0, problem.initial_state, problem.initial_parameters)['psi']
    stencil = np.diff(np.concatenate([[0.0], psi, [0.0]]), 2)
    np.testing.assert_allclose(stencil, values, atol=1e-12)
