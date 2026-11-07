"""Test the catalog of the catalog: listing, describing, creating, and running systems by name."""

import ast
import importlib
import multiprocessing
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.problems import ProblemSpec
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import RegistryError, ValidationError
from skcomplex.systems import SystemCatalog, catalog, create_simulation, create_system, describe, register_system
from skcomplex.systems.biology.common.populations import (
    ExponentialGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import PredationInteraction


def _undeclared_laws(system: System) -> list[str]:
    """Return the laws an interaction of a compiled system returned that its class does not list."""
    found = [
        f'{interaction.label}: {type(law).__name__}'
        for interaction in system.interactions
        for law in getattr(interaction, 'laws_', ())
        if interaction.get_requirements().laws and not isinstance(law, interaction.get_requirements().laws)
    ]
    return found + [item for child in system.systems for item in _undeclared_laws(child)]


def test_create_unknown_name_suggests_close_names():
    """Test that an unknown system name raises an error that suggests the closest registered names."""
    with pytest.raises(RegistryError, match=r"'pendulumm'\. Did you mean pendulum"):
        create_system('pendulumm')


def test_listings_reject_computed_registration_argument(tmp_path: Path):
    """Test that a field path computed at run time is rejected with the module and the argument."""
    module = tmp_path / 'physics' / 'mechanics' / '_computed.py'
    module.parent.mkdir(parents=True)
    module.write_text(
        'from skcomplex.systems import register_system\n\n\n'
        "@register_system(field='physics/' + 'mechanics')\n"
        'def computed():\n'
        '    """Return nothing."""\n',
    )
    message = r"skcomplex\.systems\.physics\.mechanics\._computed registers the argument 'field'"
    with pytest.raises(RegistryError, match=message):
        SystemCatalog(tmp_path).listings()


def test_create_simulation_takes_settings_that_depend_on_the_factory_arguments():
    """Test that a function of the factory arguments replaces the default run settings for these arguments."""
    entry = describe('active_gels')
    assert entry.settings_of()['until'] == entry.defaults['until']
    assert entry.settings_of(order='polar')['until'] == 200.0
    assert (
        create_simulation('active_gels', order='polar', shape=(4, 4)).prepare().run(until=2.0).trajectory_.times[-1]
        == 2.0
    )


def test_listing_imports_no_system_module():
    """Test that importing the catalog and listing it, whole or by field, loads no module of a field."""
    code = """
import sys
from skcomplex.systems import catalog
listed = len(catalog())
physics = catalog(field='physics/mechanics')
fields = {'biology', 'chemistry', 'computer_science', 'economics', 'engineering', 'mathematics', 'physics', 'social'}
loaded = [name for name in sys.modules if name.startswith('skcomplex.systems.') and name.split('.')[2] in fields]
assert listed > 500 and len(physics) > 0 and all(entry.field.startswith('physics/mechanics') for entry in physics)
assert loaded == [], loaded
assert not any(name == 'skcomplex.formalisms' or name.startswith('skcomplex.formalisms.') for name in sys.modules)
"""
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(code,))
    process.start()
    process.join()
    assert process.exitcode == 0


def test_listing_matches_registration():
    """Test that every listed system registers, once its module is imported, the metadata it was listed with."""
    for listing in catalog():
        if not listing.module.startswith('skcomplex.systems.'):
            continue
        importlib.import_module(listing.module)
        entry = describe(listing.name)
        listed = (listing.name, listing.field, listing.description, listing.tags)
        assert (entry.name, entry.field, entry.description, entry.tags) == listed, listing.name


CONFORMANCE_FRAMES = 10
SLOW = {'active_gels', 'cocoa', 'higher_order_random_lotka_volterra'}


FIELDS = [
    'biology',
    'chemistry',
    'computer_science',
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
    """Test that the catalog lists the systems of a field or subfield with descriptions, references, and formalisms."""
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


def test_every_system_has_one_field_path():
    """Test that every catalog system lives in one subfield, named by a field path of a known field."""
    library = [entry for entry in catalog() if not entry.field.startswith(('examples', 'test'))]
    assert len({entry.name for entry in library}) == len(library)
    fields = {entry.field.split('/')[0] for entry in library}
    assert fields == set(FIELDS)
    assert all(len(entry.field.split('/')) == 2 for entry in library)


def test_catalog_filters_by_tag():
    """Test that the catalog keeps only the systems with a tag."""
    assert {entry.name for entry in catalog(tag='agents')} >= {'wolf_sheep', 'schelling', 'voter_system'}


def test_create_matches_components():
    """Test that a system created by name runs like the same system built from its components."""
    created = create_system('lotka_volterra', a=0.2)
    built = System(
        elements=[PopulationElement('prey', n=40.0), PopulationElement('predators', n=9.0)],
        interactions=[
            ExponentialGrowthInteraction('growth', target='prey', r=1.1),
            PredationInteraction('predation', predator='predators', prey='prey', a=0.2, b=0.1),
            MortalityInteraction('death', target='predators', d=0.4),
        ],
    )
    first = Simulation(created, dt=0.1).prepare().run(until=5).trajectory_.states
    second = Simulation(built, dt=0.1).prepare().run(until=5).trajectory_.states
    assert np.array_equal(first, second)


def test_create_simulation_uses_default_settings():
    """Test that running a system by name uses its default duration and step."""
    simulation = create_simulation('sir').prepare().run(until=20.0)
    assert simulation.trajectory_.times[-1] == 20.0
    assert simulation.dt_ == describe('sir').defaults['dt']


def test_registration_requires_defaults():
    """Test that a factory with an argument without default cannot be registered."""

    def needs_size(size: int) -> System:
        """Return nothing useful."""
        return System()

    with pytest.raises(ValidationError, match='needs defaults for size'):
        register_system(field='test')(needs_size)
    with pytest.raises(ValidationError, match='Unknown run settings'):
        register_system(field='test', defaults={'speed': 2})(needs_size)


@pytest.mark.parametrize(
    'entry',
    [pytest.param(entry, marks=pytest.mark.slow) if entry.name in SLOW else entry for entry in catalog()],
    ids=lambda entry: entry.name,
)
def test_system_conforms(entry, tmp_path):
    """Test that a registered system is created, validated, compiled, simulated, logged, and cloned generically."""
    defaults = describe(entry.name).defaults
    system = create_system(entry.name)
    compiler = defaults.get('compiler', PassCompiler())
    assert clone(compiler).prepare(system).report_.ok
    problem = clone(compiler).prepare(system).run().problem_
    assert _undeclared_laws(system) == []
    settings = {key: value for key, value in defaults.items() if key != 'until'}
    dt = settings.get('dt', 1.0)
    until = min(defaults.get('until', CONFORMANCE_FRAMES * dt), CONFORMANCE_FRAMES * dt)
    observer = RerunObserver(entry.name, save_path=tmp_path / 'run.rrd')
    observables = describe(entry.name).observables_of(system)
    simulation = Simulation(system, observers=[observer], observables=observables, random_state=0, **settings).prepare()
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
    assert clone(compiler).prepare(copy).run().problem_.states.names == problem.states.names


@pytest.mark.parametrize('entry', catalog(), ids=lambda entry: entry.name)
def test_jacobian_pattern_holds_finite_differences(entry):
    """Test that the Jacobian pattern of every system with an ODE form holds its finite differences at random states."""
    compiler = clone(describe(entry.name).defaults.get('compiler', PassCompiler())).set_params(backend=NumpyBackend())
    problem = compiler.prepare(create_system(entry.name)).run().problem_
    if not isinstance(problem, ProblemSpec) or problem.jacobian is None or problem.delays or problem.drift is None:
        pytest.skip('The system has no ODE form without delays.')
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = problem.initial_state * (1 + 0.1 * rng.random(problem.initial_state.size))
        x = x + 0.01 * rng.random(x.size)
        _probe_pattern(problem, x, problem.initial_parameters, rng)


@pytest.mark.slow
def test_default_runs_of_a_stochastic_system_repeat():
    """Test that default catalog runs repeat, for random initial conditions and for stochastic dynamics."""
    gas = create_simulation('ideal_gas').prepare().run().trajectory_.values('gas.x_d1')
    np.testing.assert_array_equal(gas, create_simulation('ideal_gas').prepare().run().trajectory_.values('gas.x_d1'))
    assert not np.array_equal(
        gas,
        create_simulation('ideal_gas', random_state=None).prepare().run().trajectory_.values('gas.x_d1'),
    )
    counts = create_simulation('poisson_process').prepare().run().trajectory_.values('counters.count')
    np.testing.assert_array_equal(
        counts,
        create_simulation('poisson_process').prepare().run().trajectory_.values('counters.count'),
    )
    fresh = (
        create_simulation('poisson_process')
        .set_params(random_state=None)
        .prepare()
        .run()
        .trajectory_.values('counters.count')
    )
    assert not np.array_equal(counts, fresh)


# Fields whose component modules are not yet folded into their systems. The change that folds a field removes it here.
UNFOLDED = ('chemistry', 'computer_science', 'economics', 'engineering', 'mathematics', 'physics', 'social')


def _imported_modules(path: Path, field: Path) -> set[Path]:
    """Return the modules of a field that a module imports relatively, through a package's re-exports too."""
    found = set()
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        target = path.parents[node.level - 1].joinpath(*(node.module or '').split('.'))
        if target.with_suffix('.py').is_file():
            found.add(target.with_suffix('.py'))
        elif (target / '__init__.py').is_file():
            names = {alias.name for alias in node.names}
            for item in ast.parse((target / '__init__.py').read_text()).body:
                if isinstance(item, ast.ImportFrom) and item.level == 1 and names & {a.name for a in item.names}:
                    found.add(target.joinpath(*str(item.module).split('.')).with_suffix('.py'))
    return {module for module in found if module.is_relative_to(field) and module != path}


def _registers_system(path: Path) -> bool:
    """Return whether a module decorates a factory with `register_system`."""
    return any(
        isinstance(decorator, ast.Call) and getattr(decorator.func, 'id', None) == 'register_system'
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
    )


def _owners(field: Path) -> dict[Path, set[Path]]:
    """Return the system modules that reach every other private module of a field, through modules registering nothing.

    Public modules, such as the `types` module of a subfield, are part of the interface and serve no single system.
    """
    modules = [path for path in sorted(field.rglob('_*.py')) if path.name != '__init__.py']
    imports = {path: _imported_modules(path, field) for path in modules}
    systems = {path for path in modules if _registers_system(path)}
    owners: dict[Path, set[Path]] = {path: set() for path in modules if path not in systems}
    for system in systems:
        stack, seen = list(imports[system]), set()
        while stack:
            module = stack.pop()
            if module in seen or module in systems or module not in imports:
                continue
            seen.add(module)
            owners[module].add(system)
            stack.extend(imports[module])
    return owners


@pytest.mark.parametrize(
    'field',
    [
        (
            pytest.param(field, marks=pytest.mark.xfail(reason='Its components are not folded yet.', strict=True))
            if field in UNFOLDED
            else field
        )
        for field in FIELDS
    ],
)
def test_no_module_serves_one_system(field):
    """Test that no module of a field that registers no system serves only one system, which should hold it instead."""
    root = Path(str(files('skcomplex.systems'))) / field
    found = [
        f'{module.relative_to(root)} -> {next(iter(systems)).relative_to(root)}'
        for module, systems in _owners(root).items()
        if len(systems) == 1
    ]
    assert found == []
