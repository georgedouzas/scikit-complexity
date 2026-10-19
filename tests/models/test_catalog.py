"""Test the model catalog, the library's use of the public interface, and the conformance of every model."""

import importlib
import pickle  # nosec B403: the tests pickle and load only objects they create

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import ValidationError
from skcomplex.core.compile import compile_system, observables_of
from skcomplex.core.model import System
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
