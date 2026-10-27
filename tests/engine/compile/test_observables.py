"""Test the `_observables` module of `skcomplex.engine.compile`."""

from typing import Annotated

import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.errors import RoleError
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw, RuleLaw
from skcomplex.core.laws.utils import total
from skcomplex.core.specs import RoleSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import ObservableEvaluator, compile_system, observables_of, runs_rules, validate
from skcomplex.engine.compile.passes import contains_quantum
from skcomplex.models import catalog, create
from skcomplex.models.biology.ecology import PopulationElement


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


def _observed(entry_name):
    """Return the observables of a model that laws read at the current state, with their paths."""
    system = create(entry_name)
    groups = {
        scope: {item.label: item for item in [*system_of(system, scope).elements]}
        for scope, _ in observables_of(system)
    }
    found, blocked = [], set()
    for scope, observable in observables_of(system):
        group = groups[scope].get(observable.group) if observable.group is not None else None
        agents = isinstance(group, GroupElement) and runs_rules(group)
        if scope in blocked or (
            observable.group is not None and (group is None or agents or not isinstance(observable.reduce, str))
        ):
            blocked.add(scope)
            continue
        found.append(f'{scope}/{observable.label}' if scope else observable.label)
    return system, found


def system_of(system, scope):
    """Return the subsystem at a path."""
    for label in filter(None, scope.split('/')):
        system = next(child for child in system.systems if child.label == label)
    return system


def _readers(paths):
    """Return one reader of the modulus of every observable, added to a probe element."""
    readers = []
    for index, path in enumerate(paths):
        Read = type(
            f'Read{index}',
            (BaseInteraction,),
            {'probe': role(), 'source': role(), 'laws': lambda self, s: [RateLaw(s.probe.y, abs(s.source.value))]},
        )
        readers.append(Read(f'read_{index}', probe=f'probe_{index}', source=f'model/{path}'))
    return readers


@pytest.mark.parametrize(
    'entry',
    [entry for entry in catalog() if any(True for _ in observables_of(create(entry.name)))],
    ids=lambda entry: entry.name,
)
def test_reductions_agree_with_observables(entry):
    """Test that laws reading a model's observables at a state read the values the observable evaluator records."""
    model, paths = _observed(entry.name)
    if contains_quantum(model):
        pytest.skip('A quantum system cannot share a co-simulation with classical readers.')
    if not paths:
        pytest.skip('The model has no observable that laws read at the current state.')
    model.label = 'model'
    probes = [ObservedValueElement(f'probe_{index}') for index in range(len(paths))]
    problem = compile_system(System(elements=[*probes, *[model]], interactions=_readers(paths)), backend='numpy')
    reading = next((item.problem for item in problem.partitions if 'probe_0' in item.owners), problem)
    x, p = problem.initial_state, problem.initial_parameters
    rates = reading.drift(0.0, x, p)
    recorded = ObservableEvaluator(problem).prepare().evaluate(0.0, x, p, agents_of=lambda group: [])
    for index, path in enumerate(paths):
        value = rates[problem.states.entry(f'probe_{index}.y').start]
        assert value == pytest.approx(abs(recorded[f'model/{path}']), rel=1e-12, abs=1e-12)


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

    def keep(self, agent, model):
        """Keep the wealth."""
        del agent, model


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
    return System(elements=[land, village], interactions=[interaction])


def test_held_reduction_reads_one_group():
    """Test that a reduction of the agents of two groups is reported as a role error."""
    interaction = MixedReductionInteraction('mix', target='land/soil', first='village/a', second='village/b')
    (issue,) = validate(_village(interaction)).errors
    assert issue.error is RoleError
    assert 'village/a and village/b' in issue.message


def test_held_reduction_reads_only_agents():
    """Test that a reduction of agents that reads a state of a single element names the symbol it cannot read."""
    interaction = ForeignReductionInteraction('foreign', target='land/soil', first='village/a')
    (issue,) = validate(_village(interaction)).errors
    assert issue.code == 'reduction'
    assert 'n_land_soil' in issue.message
