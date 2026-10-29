"""Test the catalog of the model library: listing, describing, creating, and running models by name."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.system import System
from skcomplex.engine.backends import NumpyBackend
from skcomplex.engine.compile import PassCompiler, compile_system
from skcomplex.engine.problems import ProblemSpec
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.errors import RegistryError, ValidationError
from skcomplex.models import ModelCatalog, catalog, create, describe, register_model, run
from skcomplex.models.biology.ecology import (
    ExponentialGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
    PredationInteraction,
)


def test_create_unknown_name_suggests_close_names():
    """Test that an unknown model name raises an error that suggests the closest registered names."""
    with pytest.raises(RegistryError, match=r"'pendulumm'\. Did you mean pendulum"):
        create('pendulumm')


def test_listings_reject_computed_registration_argument(tmp_path: Path):
    """Test that a field path computed at run time is rejected with the module and the argument."""
    module = tmp_path / 'physics' / 'mechanics' / '_computed.py'
    module.parent.mkdir(parents=True)
    module.write_text(
        'from skcomplex.models import register_model\n\n\n'
        "@register_model(field='physics/' + 'mechanics')\n"
        'def computed():\n'
        '    """Return nothing."""\n',
    )
    message = r"skcomplex\.models\.physics\.mechanics\._computed registers the argument 'field'"
    with pytest.raises(RegistryError, match=message):
        ModelCatalog(tmp_path).listings()


def test_listing_matches_registration():
    """Test that every listed model registers, once its module is imported, the metadata it was listed with."""
    for listing in catalog():
        if not listing.module.startswith('skcomplex.models.'):
            continue
        importlib.import_module(listing.module)
        entry = describe(listing.name)
        listed = (listing.name, listing.field, listing.description, listing.tags)
        assert (entry.name, entry.field, entry.description, entry.tags) == listed, listing.name


CONFORMANCE_FRAMES = 10
SLOW = {'cocoa'}


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


def test_catalog_filters_by_tag():
    """Test that the catalog keeps only the models with a tag."""
    assert {entry.name for entry in catalog(tag='agents')} >= {'wolf_sheep', 'schelling', 'voter_model'}


def test_create_matches_components():
    """Test that a model created by name runs like the same model built from its components."""
    created = create('lotka_volterra', a=0.2)
    built = System(
        elements=[PopulationElement('prey', n=40.0), PopulationElement('predators', n=9.0)],
        interactions=[
            ExponentialGrowthInteraction('growth', target='prey', r=1.1),
            PredationInteraction('predation', predator='predators', prey='prey', a=0.2, b=0.1),
            MortalityInteraction('death', target='predators', d=0.4),
        ],
    )
    first = Simulation(created, dt=0.1).check().run(until=5).trajectory_.states
    second = Simulation(built, dt=0.1).check().run(until=5).trajectory_.states
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


@pytest.mark.parametrize(
    'entry',
    [pytest.param(entry, marks=pytest.mark.slow) if entry.name in SLOW else entry for entry in catalog()],
    ids=lambda entry: entry.name,
)
def test_model_conforms(entry, tmp_path):
    """Test that a registered model is created, validated, compiled, simulated, logged, and cloned generically."""
    defaults = describe(entry.name).defaults
    system = create(entry.name)
    compiler = defaults.get('compiler', PassCompiler())
    assert clone(compiler).check().prepare(system).report_.ok
    problem = compile_system(system, compiler=compiler)
    settings = {key: value for key, value in defaults.items() if key != 'until'}
    dt = settings.get('dt', 1.0)
    until = min(defaults.get('until', CONFORMANCE_FRAMES * dt), CONFORMANCE_FRAMES * dt)
    observer = RerunObserver(entry.name, save_path=tmp_path / 'run.rrd')
    observables = describe(entry.name).observables_of(system)
    simulation = Simulation(system, observers=[observer], observables=observables, random_state=0, **settings).check()
    trajectory = simulation.run(until=until).trajectory_
    assert observer.logged_frames_ == len(trajectory.times)
    for name in problem.states.names:
        assert len(trajectory.values(name)) == len(trajectory.times)
    for scope, observable in problem.observables:
        assert len(trajectory.values(f'{scope}/{observable.label}' if scope else observable.label)) == len(
            trajectory.times,
        )
    copy = clone(system)
    assert copy is not system
    assert compile_system(copy, compiler=compiler).states.names == problem.states.names


@pytest.mark.parametrize('entry', catalog(), ids=lambda entry: entry.name)
def test_jacobian_pattern_holds_finite_differences(entry):
    """Test that the Jacobian pattern of every model with an ODE form holds its finite differences at random states."""
    compiler = describe(entry.name).defaults.get('compiler')
    problem = compile_system(create(entry.name), backend=NumpyBackend(), compiler=compiler)
    if not isinstance(problem, ProblemSpec) or problem.jacobian is None or problem.delays or problem.drift is None:
        pytest.skip('The model has no ODE form without delays.')
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = problem.initial_state * (1 + 0.1 * rng.random(problem.initial_state.size))
        x = x + 0.01 * rng.random(x.size)
        _probe_pattern(problem, x, problem.initial_parameters, rng)


@pytest.mark.slow
def test_default_runs_of_a_stochastic_model_repeat():
    """Test that default catalog runs repeat, for random initial conditions and for stochastic dynamics."""
    gas = run('ideal_gas').trajectory_.values('gas.v')
    np.testing.assert_array_equal(gas, run('ideal_gas').trajectory_.values('gas.v'))
    assert not np.array_equal(gas, run('ideal_gas', random_state=None).trajectory_.values('gas.v'))
    counts = run('poisson_process').trajectory_.values('counters.count')
    np.testing.assert_array_equal(counts, run('poisson_process').trajectory_.values('counters.count'))
    fresh = run('poisson_process', settings={'random_state': None}).trajectory_.values('counters.count')
    assert not np.array_equal(counts, fresh)
