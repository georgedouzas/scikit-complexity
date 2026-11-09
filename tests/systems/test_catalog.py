"""Test the listing of the ready-made systems, and that every system conforms to the generic interface."""

import ast
import multiprocessing
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.components.systems import BaseSystem, System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.problems import ProblemSpec
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import RegistryError, ValidationError
from skcomplex.systems import all_systems
from skcomplex.systems._catalog import _SystemClasses
from skcomplex.systems.biology.common.populations import (
    ExponentialGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import LotkaVolterraSystem, PredationInteraction
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.economics.finance import OrderBookSystem
from skcomplex.systems.mathematics.stochastic_processes import PoissonProcessSystem
from skcomplex.systems.physics.active_matter import ActiveGelsSystem
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
from skcomplex.systems.social.opinion_dynamics import AxelrodSystem


def _undeclared_laws(system: System) -> list[str]:
    """Return the laws an interaction of a compiled system returned that its class does not list."""
    found = [
        f'{interaction.label}: {type(law).__name__}'
        for interaction in system.interactions
        for law in getattr(interaction, 'laws_', ())
        if interaction.get_requirements().laws and not isinstance(law, interaction.get_requirements().laws)
    ]
    return found + [item for child in system.systems for item in _undeclared_laws(child)]


def test_run_settings_depend_on_the_model_parameters():
    """Test that a system class whose tuned run settings depend on its model parameters gives them for these."""
    assert ActiveGelsSystem().get_run_settings()['until'] == ActiveGelsSystem.until
    assert ActiveGelsSystem(order='polar').get_run_settings()['until'] == 200.0
    assert simulate(ActiveGelsSystem(order='polar', shape=(4, 4)), until=2.0).times[-1] == 2.0


def test_simulate_reads_the_run_settings_of_an_instance():
    """Test that the observables and invariants of an instance are those `simulate` records and checks."""
    observables = AxelrodSystem(width=4, height=4).get_run_settings()['observables']
    trajectory = simulate(AxelrodSystem(width=4, height=4), until=1, random_state=0)
    assert observables
    assert all(len(trajectory.values(spec.label)) == len(trajectory.times) for spec in observables)
    assert OrderBookSystem().get_run_settings()['invariants'] == {'total_money': 1e-6, 'total_shares': 1e-6}


def test_importing_the_systems_imports_no_system_module():
    """Test that importing the systems package loads no module of a field."""
    code = """
import sys
import skcomplex.systems
fields = {'biology', 'chemistry', 'computer_science', 'economics', 'engineering', 'mathematics', 'physics', 'social'}
loaded = [name for name in sys.modules if name.startswith('skcomplex.systems.') and name.split('.')[2] in fields]
assert loaded == [], loaded
"""
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(code, {}))
    process.start()
    process.join()
    assert process.exitcode == 0


def test_all_systems_lists_system_classes_by_field_and_topics():
    """Test that the system classes of a field or with topics are listed by class name, sorted, and imported alone."""
    code = """
import sys
from skcomplex.systems import all_systems
from skcomplex.core.components.systems import BaseSystem
listed = all_systems(field='chemistry/pattern_formation')
assert [name for name, _ in listed] == ['BrusselatorSystem', 'GrayScottSystem', 'SchnakenbergSystem'], listed
assert all(issubclass(system, BaseSystem) and system.__name__ == name for name, system in listed)
loaded = [name for name in sys.modules if name.startswith('skcomplex.systems.chemistry.kinetics')]
assert loaded == [], loaded
assert [name for name, _ in all_systems(field='chemistry', topics=('oscillation', 'pde'))] == ['BrusselatorSystem']
"""
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(code, {}))
    process.start()
    process.join()
    assert process.exitcode == 0


def test_all_systems_filters_by_capability_tags():
    """Test that filtering by capability tags keeps the system classes whose tags hold every wanted value."""
    stochastic = all_systems(field='chemistry', tags={'randomness': 'stochastic'})
    assert [name for name, _ in stochastic] == ['ChemicalLangevinEquationSystem', 'ChemicalLangevinNetworkSystem']
    lattice = all_systems(field='chemistry', tags={'space': 'lattice', 'laws': ['update']})
    assert [name for name, _ in lattice] == ['CyclicCellularAutomatonSystem', 'HodgepodgeSystem']
    with pytest.raises(ValidationError, match="The tag 'spaces' is unknown"):
        all_systems(field='chemistry', tags={'spaces': 'lattice'})


def test_listing_rejects_a_system_class_without_suffix_or_with_computed_field(tmp_path: Path):
    """Test that a class giving `field` needs the suffix `System` and a literal field path."""
    module = tmp_path / 'physics' / 'mechanics' / '_classes.py'
    module.parent.mkdir(parents=True)
    module.write_text("class Pendulum:\n    field = 'physics/mechanics'\n")
    with pytest.raises(RegistryError, match='Name it with the suffix `System`'):
        _SystemClasses(tmp_path).classes()
    module.write_text("class PendulumSystem:\n    field = 'physics/' + 'mechanics'\n")
    with pytest.raises(RegistryError, match="defines the class variable 'field' with a value computed at run time"):
        _SystemClasses(tmp_path).classes()


CONFORMANCE_FRAMES = 10
SLOW = {'ActiveGelsSystem', 'CocoaSystem', 'HigherOrderRandomLotkaVolterraSystem'}


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


def test_all_systems_filters_by_field():
    """Test that the systems of a field or subfield are listed, each with its field, references, and description."""
    ecology = all_systems(field='biology/ecology')
    names = [name for name, _ in ecology]
    assert {'ExponentialGrowthSystem', 'LogisticGrowthSystem', 'LotkaVolterraSystem', 'WolfSheepSystem'} <= set(names)
    assert all(system.field == 'biology/ecology' and system.references and system.__doc__ for _, system in ecology)
    biology = all_systems(field='biology')
    assert set(names) < {name for name, _ in biology}
    assert {system.field.partition('/')[0] for _, system in biology} == {'biology'}


def test_every_system_has_one_field_path():
    """Test that every ready-made system lives in one subfield, named by a field path of a known field."""
    library = all_systems()
    assert len({name for name, _ in library}) == len(library)
    assert {system.field.split('/')[0] for _, system in library} == set(FIELDS)
    assert all(len(system.field.split('/')) == 2 for _, system in library)


def test_all_systems_filters_by_topic():
    """Test that listing by a topic keeps only the systems with it."""
    assert {name for name, _ in all_systems(topics='agents')} >= {'WolfSheepSystem', 'SchellingSystem', 'VoterSystem'}


def test_system_class_matches_components():
    """Test that a system class runs like the same system built from its components."""
    built = System(
        elements=[PopulationElement('prey', n=40.0), PopulationElement('predators', n=9.0)],
        interactions=[
            ExponentialGrowthInteraction('growth', target='prey', r=1.1),
            PredationInteraction('predation', predator='predators', prey='prey', a=0.2, b=0.1),
            MortalityInteraction('death', target='predators', d=0.4),
        ],
    )
    first = simulate(LotkaVolterraSystem(a=0.2).prepare().system_, until=5, dt=0.1).states
    second = simulate(built, until=5, dt=0.1).states
    assert np.array_equal(first, second)


def test_simulate_uses_the_tuned_settings():
    """Test that simulating a system class uses its tuned time between frames."""
    trajectory = simulate(SIRSystem(), until=20.0)
    assert trajectory.times[-1] == 20.0
    assert trajectory.times[1] - trajectory.times[0] == SIRSystem.dt


@pytest.mark.parametrize(
    'system',
    [pytest.param(system, marks=pytest.mark.slow) if name in SLOW else system for name, system in all_systems()],
    ids=lambda system: system.__name__,
)
def test_system_conforms(system: type[BaseSystem], tmp_path: Path):
    """Test that a ready-made system is built, validated, compiled, simulated, logged, and cloned generically."""
    built = system()
    settings = built.get_run_settings()
    dt = settings['dt'] or 1.0
    tuned = settings['until']
    until = CONFORMANCE_FRAMES * dt if tuned is None else min(tuned, CONFORMANCE_FRAMES * dt)
    observer = RerunObserver(system.__name__, save_path=tmp_path / 'run.rrd')
    trajectory = simulate(built, until=until, observers=[observer], random_state=0)
    problem = trajectory.problem
    assert _undeclared_laws(built.system_) == []
    assert observer.logged_frames_ == len(trajectory.times)
    for name in problem.states.names:
        assert len(trajectory.values(name)) == len(trajectory.times)
    for scope, observable in problem.observables:
        assert len(trajectory.values(f'{scope}/{observable.label}' if scope else observable.label)) == len(
            trajectory.times,
        )
    copy = clone(built)
    assert copy is not built
    assert simulate(copy, until=0, random_state=0).problem.states.names == problem.states.names


@pytest.mark.parametrize('system', [system for _, system in all_systems()], ids=lambda system: system.__name__)
def test_jacobian_pattern_holds_finite_differences(system: type[BaseSystem]):
    """Test that the Jacobian pattern of every system with an ODE form holds its finite differences at random states."""
    problem = simulate(system(), until=0, backend=NumpyBackend(), random_state=0).problem
    if not isinstance(problem, ProblemSpec) or problem.jacobian is None or problem.delays or problem.drift is None:
        pytest.skip('The system has no ODE form without delays.')
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = problem.initial_state * (1 + 0.1 * rng.random(problem.initial_state.size))
        x = x + 0.01 * rng.random(x.size)
        _probe_pattern(problem, x, problem.initial_parameters, rng)


@pytest.mark.slow
def test_seeded_runs_of_a_stochastic_system_repeat():
    """Test that seeded runs repeat, for random initial conditions and for stochastic dynamics."""
    gas = simulate(IdealGasSystem(), random_state=0).values('gas.x_d1')
    np.testing.assert_array_equal(gas, simulate(IdealGasSystem(), random_state=0).values('gas.x_d1'))
    assert not np.array_equal(gas, simulate(IdealGasSystem(random_state=None), random_state=0).values('gas.x_d1'))
    counts = simulate(PoissonProcessSystem(), random_state=0).values('counters.count')
    np.testing.assert_array_equal(counts, simulate(PoissonProcessSystem(), random_state=0).values('counters.count'))
    assert not np.array_equal(counts, simulate(PoissonProcessSystem(), random_state=None).values('counters.count'))


# Fields whose component modules are not yet folded into their systems. The change that folds a field removes it here.
UNFOLDED: tuple[str, ...] = ()


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


def _defines_field(node: ast.ClassDef) -> bool:
    """Return whether a class body gives the class variable `field`, which makes the class a system class."""
    return any(
        isinstance(item, ast.Assign) and any(getattr(target, 'id', None) == 'field' for target in item.targets)
        for item in node.body
    )


def _defines_system(path: Path) -> bool:
    """Return whether a module defines a system class."""
    return any(isinstance(node, ast.ClassDef) and _defines_field(node) for node in ast.parse(path.read_text()).body)


def _owners(field: Path) -> dict[Path, set[Path]]:
    """Return the system modules that reach every other private module of a field, through modules defining none.

    Public modules, such as the `types` module of a subfield, are part of the interface and serve no single system. A
    module of `common` that another module of `common` imports belongs to a shared family, such as the force tables of
    an external engine, and serves no single system either.
    """
    modules = [path for path in sorted(field.rglob('_*.py')) if path.name != '__init__.py']
    imports = {path: _imported_modules(path, field) for path in modules}
    systems = {path for path in modules if _defines_system(path)}
    common = field / 'common'
    family = {
        module
        for path in modules
        if path.is_relative_to(common)
        for module in imports[path]
        if module.is_relative_to(common)
    }
    owners: dict[Path, set[Path]] = {path: set() for path in modules if path not in systems | family}
    for system in systems:
        stack, seen = list(imports[system]), set()
        while stack:
            module = stack.pop()
            if module in seen or module in systems or module not in imports:
                continue
            seen.add(module)
            if module in owners:
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
    """Test that no module of a field that defines no system serves only one system, which should hold it instead."""
    root = Path(str(files('skcomplex.systems'))) / field
    found = [
        f'{module.relative_to(root)} -> {next(iter(systems)).relative_to(root)}'
        for module, systems in _owners(root).items()
        if len(systems) == 1
    ]
    assert found == []


@pytest.mark.parametrize('field', FIELDS)
def test_no_system_module_imports_another(field):
    """Test that no module defining systems imports another module defining systems of its subfield."""
    root = Path(str(files('skcomplex.systems'))) / field
    systems = [path for path in sorted(root.rglob('_*.py')) if _defines_system(path)]
    found = [
        f'{path.relative_to(root)} -> {module.relative_to(root)}'
        for path in systems
        for module in sorted(_imported_modules(path, root))
        if module.parent == path.parent and _defines_system(module)
    ]
    assert found == []
