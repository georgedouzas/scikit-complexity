"""Test the `_observables` module of `skcomplex.engine.compile`."""

import pytest

from skcomplex.core.components import ElementComponent, GroupComponent
from skcomplex.core.interactions import BaseInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.engine.compile import ObservableEvaluator, compile_system, observables_of
from skcomplex.formalisms.agents import runs_rules
from skcomplex.formalisms.quantum import contains_quantum
from skcomplex.models import catalog, create

ObservedValueComponent = ElementComponent.define('ObservedValueComponent', states={'y': 0.0})


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
        agents = isinstance(group, GroupComponent) and runs_rules(group)
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
        Read = BaseInteraction.define(
            f'Read{index}',
            roles=['probe', 'source'],
            laws=lambda s: [RateLaw(s.probe.y, abs(s.source.value))],
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
    probes = [ObservedValueComponent(f'probe_{index}') for index in range(len(paths))]
    problem = compile_system(System(components=[*probes, *[model]], interactions=_readers(paths)), backend='numpy')
    reading = next((item.problem for item in problem.partitions if 'probe_0' in item.owners), problem)
    x, p = problem.initial_state, problem.initial_parameters
    rates = reading.drift(0.0, x, p)
    recorded = ObservableEvaluator(problem).evaluate(0.0, x, p, agents_of=lambda group: [])
    for index, path in enumerate(paths):
        value = rates[problem.states.entry(f'probe_{index}.y').start]
        assert value == pytest.approx(abs(recorded[f'model/{path}']), rel=1e-12, abs=1e-12)
