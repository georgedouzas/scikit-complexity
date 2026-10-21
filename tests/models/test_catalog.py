"""Test the model catalog, the library's use of the public interface, and the conformance of every model."""

import importlib
import pickle  # nosec B403: the tests pickle and load only objects they create

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import ValidationError
from skcomplex.core.compile import ObservableEvaluator, compile_system, observables_of
from skcomplex.core.model import AgentType, Element, Group, Interaction, QuantumSystem, Rate, System
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import catalog, create, describe, register_model, run
from skcomplex.models.biology.ecology import ExponentialGrowth, Mortality, Population, Predation

CONFORMANCE_FRAMES = 10
FIELDS = [
    'biology',
    'chemistry',
    'computer_science',
    'earth',
    'economics',
    'engineering',
    'mathematics',
    'physics',
    'social',
]


def test_catalog_filters_by_field():
    """Test that the catalog lists the models of a field or a subfield with descriptions, references, and formalisms."""
    entries = catalog(field='biology/ecology')
    names = [entry.name for entry in entries]
    assert {'exponential_growth', 'logistic_growth', 'lotka_volterra', 'wolf_sheep'} <= set(names)
    assert all(entry.field == 'biology/ecology' and entry.description and entry.references for entry in entries)
    biology = catalog(field='biology')
    assert {entry.name for entry in entries} < {entry.name for entry in biology}
    assert {entry.field.partition('/')[0] for entry in biology} == {'biology'}
    lotka = describe('lotka_volterra')
    assert lotka.formalism == 'ordinary differential equations'
    assert lotka.parameters['a'] == 0.4
    assert not lotka.description.startswith('Return')


def test_every_model_has_one_field_path():
    """Test that every library model lives in one subfield, named by a field path of a known field."""
    library = [entry for entry in catalog() if not entry.field.startswith(('examples', 'test'))]
    assert len({entry.name for entry in library}) == len(library)
    fields = {entry.field.split('/')[0] for entry in library}
    assert fields == set(FIELDS)
    assert all(len(entry.field.split('/')) == 2 for entry in library)


@pytest.mark.parametrize(
    ('package', 'names'),
    [
        ('skcomplex.models.physics.fluids', ['burgers']),
        ('skcomplex.models.biology.ecology', ['lotka_volterra']),
        ('skcomplex.models.chemistry.pattern_formation', ['gray_scott']),
    ],
)
def test_import_by_field(package, names):
    """Test that a subfield package exports the factories of its subfield."""
    module = importlib.import_module(package)
    assert all(callable(getattr(module, name)) for name in names)


@pytest.mark.parametrize('package', ['fluids', 'lattices', 'fields', 'nonlinear'])
def test_technique_packages_are_gone(package):
    """Test that the technique-based model packages no longer exist."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f'skcomplex.models.{package}')


def test_catalog_filters_by_tag():
    """Test that the catalog keeps only the models with a tag."""
    assert {entry.name for entry in catalog(tag='agents')} >= {'wolf_sheep', 'schelling', 'voter_model'}


def test_create_matches_components():
    """Test that a model created by name runs like the same model built from its components."""
    created = create('lotka_volterra', a=0.2)
    built = System(
        elements=[Population('prey', n=40.0), Population('predators', n=9.0)],
        interactions=[
            ExponentialGrowth('growth', target='prey', r=1.1),
            Predation('predation', predator='predators', prey='prey', a=0.2, b=0.1),
            Mortality('death', target='predators', d=0.4),
        ],
    )
    first = Simulation(created, dt=0.1).run(until=5).trajectory_.states
    second = Simulation(built, dt=0.1).run(until=5).trajectory_.states
    assert np.array_equal(first, second)


def test_run_uses_default_settings():
    """Test that running a model by name uses its default duration and step."""
    simulation = run('sir', settings={'until': 20.0})
    assert simulation.trajectory_.times[-1] == 20.0
    assert simulation.dt_ == describe('sir').defaults['dt']


def test_registration_requires_defaults():
    """Test that a factory with an argument without default cannot be registered."""

    def needs_size(size: int) -> System:
        """Return nothing useful."""
        return System()

    with pytest.raises(ValidationError, match='needs defaults for size'):
        register_model(field='test')(needs_size)
    with pytest.raises(ValidationError, match='Unknown run settings'):
        register_model(field='test', defaults={'speed': 2})(needs_size)


@pytest.mark.parametrize('entry', catalog(), ids=lambda entry: entry.name)
def test_model_conforms(entry, tmp_path):
    """Test that a registered model is created, compiled, simulated, logged to Rerun, and cloned generically."""
    system = create(entry.name)
    problem = compile_system(system)
    settings = {key: value for key, value in entry.defaults.items() if key != 'until'}
    dt = settings.get('dt', 1.0)
    until = min(entry.defaults.get('until', CONFORMANCE_FRAMES * dt), CONFORMANCE_FRAMES * dt)
    observer = RerunObserver(entry.name, save_path=tmp_path / 'run.rrd')
    simulation = Simulation(system, observers=[observer], random_state=0, **settings).run(until=until)
    trajectory = simulation.trajectory_
    assert observer.logged_frames_ == len(trajectory.times)
    for name in problem.states.names:
        assert len(trajectory[name]) == len(trajectory.times)
    for scope, observable in observables_of(system):
        assert len(trajectory[f'{scope}/{observable.label}' if scope else observable.label]) == len(trajectory.times)
    copy = clone(system)
    assert copy is not system
    assert compile_system(copy).states.names == problem.states.names
    assert pickle.loads(pickle.dumps(system)) is not None  # noqa: S301  # nosec B301: loads data the test just dumped


PROBED_COLUMNS = 40


def _probe_pattern(problem, x, p, rng):
    """Check that the central differences of a drift in sampled columns lie inside the derived Jacobian pattern."""
    pattern = problem.jacobian.pattern()
    if pattern is None:
        return
    pattern = pattern.tocsc()
    rates = np.abs(problem.drift(0.0, x, p))
    size = x.size
    for column in rng.choice(size, min(size, PROBED_COLUMNS), replace=False):
        step = 1e-6 * max(1.0, abs(x[column]))
        shift = np.zeros(size)
        shift[column] = step
        with np.errstate(all='ignore'):
            derivative = (problem.drift(0.0, x + shift, p) - problem.drift(0.0, x - shift, p)) / (2 * step)
        found = np.flatnonzero(np.isfinite(derivative) & (np.abs(derivative) > 1e-8 + 1e-7 * rates))
        allowed = pattern[:, [column]].nonzero()[0]
        assert set(found) <= set(allowed), f'column {column} reaches rows {sorted(set(found) - set(allowed))}'


@pytest.mark.parametrize('entry', catalog(), ids=lambda entry: entry.name)
def test_jacobian_pattern_holds_finite_differences(entry):
    """Test that the Jacobian pattern of every model with an ODE form holds its finite differences at random states."""
    problem = compile_system(create(entry.name), backend='numpy')
    if problem.jacobian is None or problem.delays or problem.drift is None:
        pytest.skip('The model has no ODE form without delays.')
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = problem.initial_state * (1 + 0.1 * rng.random(problem.initial_state.size))
        x = x + 0.01 * rng.random(x.size)
        _probe_pattern(problem, x, problem.initial_parameters, rng)


ObservedValue = Element.define('ObservedValue', states={'y': 0.0})


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
        agents = isinstance(group, Group) and issubclass(group.element, AgentType)
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
        Read = Interaction.define(
            f'Read{index}',
            roles=['probe', 'source'],
            laws=lambda s: [Rate(s.probe.y, abs(s.source.value))],
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
    if isinstance(model, QuantumSystem):
        pytest.skip('A quantum system cannot share a co-simulation with classical readers.')
    if not paths:
        pytest.skip('The model has no observable that laws read at the current state.')
    model.label = 'model'
    probes = [ObservedValue(f'probe_{index}') for index in range(len(paths))]
    problem = compile_system(System(systems=[model], elements=probes, interactions=_readers(paths)), backend='numpy')
    reading = next((item.problem for item in problem.partitions if 'probe_0' in item.owners), problem)
    x, p = problem.initial_state, problem.initial_parameters
    rates = reading.drift(0.0, x, p)
    recorded = ObservableEvaluator(problem).evaluate(0.0, x, p, agents_of=lambda group: [])
    for index, path in enumerate(paths):
        value = rates[problem.states.entry(f'probe_{index}.y').start]
        assert value == pytest.approx(abs(recorded[f'model/{path}']), rel=1e-12, abs=1e-12)


def test_default_runs_of_a_stochastic_model_repeat():
    """Test that default catalog runs repeat, for random initial conditions and for stochastic dynamics."""
    gas = run('ideal_gas').trajectory_['gas.v']
    np.testing.assert_array_equal(gas, run('ideal_gas').trajectory_['gas.v'])
    assert not np.array_equal(gas, run('ideal_gas', random_state=None).trajectory_['gas.v'])
    counts = run('poisson_process').trajectory_['counters.count']
    np.testing.assert_array_equal(counts, run('poisson_process').trajectory_['counters.count'])
    fresh = run('poisson_process', settings={'random_state': None}).trajectory_['counters.count']
    assert not np.array_equal(counts, fresh)
