"""Validate systems against the requirements of their primitives and the support of solvers and backends."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ..base import (
    BackendError,
    Categorical,
    Domain,
    DomainKind,
    Form,
    HilbertFamily,
    HilbertSpace,
    IncompatibleLawsError,
    LabelError,
    LawKind,
    NoiseKind,
    ParameterError,
    Registry,
    Requires,
    RoleError,
    SkcomplexError,
    SolverError,
    SpaceError,
    SpaceKind,
    Structure,
    Supports,
    Tensor,
    TimeKind,
    ValidationError,
    VariableKind,
    declarations_of,
)
from ..components import Wiener
from ..interactions import Match
from ..laws import (
    Auxiliary,
    Boundary,
    Dissipator,
    Drive,
    Emission,
    Equation,
    Event,
    Extremal,
    Flux,
    Generator,
    GraphProduction,
    Invariant,
    Jump,
    LocalRule,
    Noise,
    Production,
    Rate,
    Relaxation,
    Rule,
    Sample,
    Substitution,
    Transfer,
    Transition,
    Update,
)
from ..space import LatticeSpace, NetworkSpace, Space
from ..system import Observable, System
from ._backends import BACKENDS, load_backends
from ._compiler import CensusRecord, _held_observables, _process_covariance, _with_held_observables, law_census
from ._layout import Entry
from ._observables import with_observables
from ._quantum import contains_quantum
from ._rewriting import is_rewriting
from ._tree import flatten_system

OPTIONAL_SOLVERS = {'lammps': 'lammps'}
COVARIANCE_TOLERANCE = 1e-12


def _solver_registry() -> Registry:
    """Return the registry of solvers, imported late since the solvers import the compiler."""
    from ..solvers import SOLVERS  # noqa: PLC0415

    load_backends()

    return SOLVERS


def law_types() -> tuple[Any, ...]:
    """Return every law type of the core.

    Returns:
        types:
            The law types, in declaration order.
    """
    return (
        Rate,
        Noise,
        Jump,
        Update,
        Transition,
        Emission,
        Boundary,
        Rule,
        Equation,
        Event,
        LocalRule,
        Drive,
        Extremal,
        Relaxation,
        Sample,
        Match,
        Flux,
        Transfer,
        Auxiliary,
        Invariant,
        Generator,
        Dissipator,
        Production,
        Substitution,
        GraphProduction,
    )


def _subclasses(base: type) -> list[type]:
    """Return every subclass of a class, recursively."""
    found: list[type] = []
    subclasses: list[type] = base.__subclasses__()
    for subclass in subclasses:
        found += [subclass, *_subclasses(subclass)]
    return found


def missing_declarations() -> list[str]:
    """Return the primitives, solvers, and backends that do not declare their requirements or support.

    Every law type declares its law kind and its requirements, every space its kind, every registered solver and
    backend its support. Validation reads only these declarations, so a primitive without one cannot be validated.

    Returns:
        missing:
            One sentence per missing declaration.

    Examples:
        >>> from skcomplex.core.compile import missing_declarations
        >>> missing_declarations()
        []
    """
    missing = []
    for law in law_types():
        if not isinstance(getattr(law, 'law_kind', None), LawKind):
            missing.append(f'The law {law.__name__} declares no law kind.')
        if not isinstance(getattr(law, 'requires', None), Requires):
            missing.append(f'The law {law.__name__} declares no requirements.')
    missing.extend(
        f'The space {space.__name__} declares no space kind.'
        for space in _subclasses(Space)
        if not isinstance(getattr(space, 'space_kind', None), SpaceKind)
    )
    solvers = _solver_registry()
    for key in solvers:
        solver = solvers.resolve(key)
        if not isinstance(getattr(solver, 'supports', None), Supports):
            _, name = cast('tuple[Form, str]', key)
            missing.append(f'The solver {name!r} ({solver.__name__}) declares no support.')
    for key in BACKENDS:
        backend = BACKENDS.resolve(key)
        if not isinstance(getattr(backend, 'supports', None), Supports):
            missing.append(f'The backend {key!r} ({backend.__name__}) declares no support.')
    return list(dict.fromkeys(missing))


ERROR = 'error'
WARNING = 'warning'
FIXES: dict[type[SkcomplexError], tuple[str, str]] = {
    LabelError: ('label', 'Give every child of a system a unique label of letters, digits, and underscores.'),
    RoleError: ('role', 'Bind every role to the path of an element, group, or member of the system.'),
    SpaceError: ('space-missing', 'Give the system a space whose kind the component supports.'),
    ParameterError: ('parameter', 'Give the parameter a value inside its domain.'),
}
REPLACEMENTS = {
    DomainKind.CATEGORICAL: 'a jump, transition, or update law',
    DomainKind.INTEGER: 'a jump or update law',
    DomainKind.BOOLEAN: 'a jump or update law',
    DomainKind.HILBERT: 'a generator or dissipator law',
}


@dataclass(frozen=True)
class Issue:
    """Problem that validation found in a system, with where it is and how to fix it.

    Args:
        code:
            The stable code of the problem, such as `'law-domain'`.

        path:
            The slash path of the component or system the problem is in.

        message:
            What is wrong and why.

        fix:
            The change that makes the system valid.

        severity:
            `'error'` for problems that stop compilation, `'warning'` otherwise.

        error:
            The class of the error raised for the problem: `ValidationError` or one of its subclasses for the problems
            validation finds, and the class of the original error for a component that fails to declare.
    """

    code: str
    path: str
    message: str
    fix: str
    severity: str = ERROR
    error: type[SkcomplexError] = field(default=ValidationError, repr=False, compare=False)

    def __str__(self) -> str:
        """Return the issue as one line with its path, code, message, and fix."""
        return f'{self.path}: {self.message} {self.fix} [{self.code}]'


@dataclass
class ValidationReport:
    """Every problem of a system, its solver, and its backend.

    Args:
        issues:
            The problems, in the order they were found.

    Examples:
        >>> from skcomplex.core.compile import validate
        >>> from skcomplex.core.system import System
        >>> validate(System()).ok
        True
    """

    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        """Return the problems that stop compilation."""
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        """Return the problems that do not stop compilation."""
        return [issue for issue in self.issues if issue.severity == WARNING]

    @property
    def ok(self) -> bool:
        """Check whether the report holds no error."""
        return not self.errors

    def raise_errors(self) -> None:
        """Raise one error that lists every error of the report, after emitting its warnings.

        Raises:
            ValidationError:
                The report holds an error.
        """
        for issue in self.warnings:
            warnings.warn(str(issue), ValidationWarning, stacklevel=3)
        if self.errors:
            lines = '\n'.join(f'- {issue}' for issue in self.errors)
            error_msg = f'The system is not valid, with {len(self.errors)} error(s):\n{lines}'
            classes = {issue.error for issue in self.errors}
            raise (classes.pop() if len(classes) == 1 else ValidationError)(error_msg)


class ValidationWarning(UserWarning):
    """Warning of a problem that does not stop compilation."""


def domain_kind(entry: Entry, *, vector: bool = False) -> DomainKind:
    """Return the kind of value a state of the layout holds.

    Args:
        entry:
            The layout entry of the state.

        vector:
            Whether the state is declared as a vector of the space's dimension. The shape of an entry does not tell,
            since a field holds one value per site.

    Returns:
        kind:
            The domain kind.
    """
    domain = entry.domain
    if vector or entry.is_complex or domain is Domain.COMPLEX:
        return DomainKind.VECTOR if vector else DomainKind.COMPLEX
    if isinstance(domain, Structure):
        return DomainKind(domain.kind)
    classes = {Categorical: DomainKind.CATEGORICAL, Tensor: DomainKind.TENSOR}
    classes |= {HilbertSpace: DomainKind.HILBERT, HilbertFamily: DomainKind.HILBERT}
    if type(domain) in classes:
        return classes[type(domain)]
    kinds: dict[object, DomainKind] = {
        Domain.INTEGER: DomainKind.INTEGER,
        Domain.NONNEGATIVE_INTEGER: DomainKind.INTEGER,
        Domain.BOOLEAN: DomainKind.BOOLEAN,
        Domain.ANGLE: DomainKind.ANGLE,
    }
    return kinds.get(domain, DomainKind.REAL)


def _space_kind(space: Space | None) -> SpaceKind:
    """Return the kind of a space, or `none` without one."""
    return SpaceKind.NONE if space is None else space.space_kind


def _names(terms: Iterable[object]) -> str:
    """Return terms as a sorted, comma-separated list."""
    return ', '.join(sorted(str(term) for term in terms))


def _solver_classes() -> dict[Any, list[str]]:
    """Return every registered solver class with its names, without the `'auto'` aliases."""
    classes: dict[Any, list[str]] = {}
    solvers = _solver_registry()
    for key in solvers:
        _, name = cast('tuple[Form, str]', key)
        if name != 'auto':
            classes.setdefault(solvers.resolve(key), []).append(name)
    return classes


def _fits(kinds: set[LawKind]) -> bool:
    """Check whether one registered solver of a single form runs every law kind of a set.

    Co-simulation is left out, since it runs partitions that each need a solver of their own.
    """
    return any(kinds <= solver.supports.laws for solver in _solver_classes() if solver.form is not Form.COUPLED)


def _structure_issues(system: System) -> list[Issue]:
    """Return the problems of the system tree that stop the census: wrong child types and bad labels."""
    try:
        flatten_system(system)
    except SkcomplexError as error:
        code, fix = FIXES.get(
            type(error),
            ('structure', 'Pass elements, groups, interactions, and systems as children.'),
        )
        return [Issue(code, system.label, str(error), fix, error=type(error))]
    return []


def _declared_times(system: System, scope: str = '', inherited: TimeKind | None = None) -> dict[str, TimeKind | None]:
    """Return the declared time of every system of a tree by path, a nested system inheriting its parent's."""
    time = system.time.kind if system.time is not None else inherited
    times = {scope: time}
    for child in system.systems:
        times.update(_declared_times(child, f'{scope}/{child.label}' if scope else child.label, time))
    return times


def _binding_issues(system: System) -> list[Issue]:
    """Return the interactions whose binding lacks its roles and parameters, or that the space does not offer."""
    nodes = flatten_system(system)
    by_path = {node.path: node for node in nodes}
    issues = []
    for node in nodes:
        interaction = node.component
        over = getattr(type(interaction), 'over', None)
        if over is None or over == 'members':
            continue
        roles = declarations_of(type(interaction), VariableKind.ROLE)
        if over == 'pairs':
            missing = [name for name in ('first', 'second') if name not in roles]
            missing += [] if 'cutoff' in declarations_of(type(interaction), VariableKind.PARAMETER) else ['cutoff']
            if missing:
                message = 'The binding over pairs needs the roles first and second and the parameter cutoff.'
                fix = f'Declare {", ".join(missing)} on {type(interaction).__name__}.'
                issues.append(Issue('binding', node.path, message, fix))
            continue
        target = by_path.get(f'{node.scope}{getattr(interaction, "target", "")}')
        space = target.space if target is not None else node.space
        offered = (
            isinstance(space, LatticeSpace | NetworkSpace)
            if over == 'neighbours'
            else bool(
                getattr(space, 'hyperedges', ()),
            )
        )
        if not offered:
            needs = 'a lattice or network space' if over == 'neighbours' else 'a network space with hyperedges'
            message = f'The binding over {over} needs {needs}, but the bound members live on {space!r}.'
            issues.append(Issue('binding', node.path, message, f'Give the bound group {needs}.'))
    return issues


def _law_issues(records: list[CensusRecord], times: dict[str, TimeKind | None]) -> list[Issue]:
    """Return the laws whose targets, space, or time their requirements do not accept."""
    issues = []
    for record in records:
        time = times.get(record.path.rpartition('/')[0])
        law, requires = record.law, record.law.requires
        name = f'{type(law).__name__} law of {record.path}'
        for entry in record.targets:
            kind = domain_kind(entry, vector=entry.name in record.vectors)
            if not requires.accepts('domains', kind):
                fix = f'Use {REPLACEMENTS[kind]}.' if kind in REPLACEMENTS else 'Target a state of a supported domain.'
                message = (
                    f'The {name} targets {entry.name}, a {kind} state, but {law.law_kind} laws target '
                    f'{_names(requires.domains)} states.'
                )
                issues.append(Issue('law-domain', record.path, message, fix))
        space = _space_kind(record.space)
        if not requires.accepts('space', space):
            message = f'The {name} needs a {_names(requires.space)} space, but the component lives on {space}.'
            issues.append(Issue('law-space', record.path, message, 'Give the system a space of a supported kind.'))
        if time is not None and not requires.accepts('time', time):
            message = f'The {name} is a {law.law_kind.law_class} law, which needs {_names(requires.time)} time.'
            kinds = {
                str(item.law_kind) for item in law_types() if item.requires.time and item.requires.accepts('time', time)
            }
            fix = f'Declare {_names(requires.time)} time, or use a law of {time} time: {_names(kinds)}.'
            issues.append(Issue('law-time', record.path, message, fix))
    return issues


def _mixed_issues(system: System, owners: dict[str, set[LawKind]], scope: str = '') -> list[Issue]:
    """Return the systems whose own laws no single solver runs, naming their components by law class."""
    under = {owner: kinds for owner, kinds in owners.items() if owner.startswith(scope)}
    if _fits(set().union(*under.values())):
        return []
    own = {owner: kinds for owner, kinds in under.items() if '/' not in owner[len(scope) :]}
    if not system.systems or not _fits(set().union(*own.values())):
        classes: dict[str, list[str]] = {}
        for owner, kinds in (own if system.systems else under).items():
            for kind in kinds:
                classes.setdefault(f'{kind}', []).append(owner)
        listed = '; '.join(f'{kind}: {", ".join(sorted(set(paths)))}' for kind, paths in sorted(classes.items()))
        message = f'No solver runs the laws of this system together ({listed}).'
        fix = 'Move the components of each law kind into their own nested system, so they run as coupled partitions.'
        return [Issue('laws-mixed', scope.rstrip('/') or system.label, message, fix, error=IncompatibleLawsError)]
    issues = []
    for child in system.systems:
        issues += _mixed_issues(child, owners, f'{scope}{child.label}/')
    return issues


def _solver_issues(solver: object, records: list[CensusRecord], system: System) -> list[Issue]:
    """Return the requirements of a system that a chosen solver does not support."""
    classes = _solver_classes()
    if isinstance(solver, str):
        found = [cls for cls, names in classes.items() if solver in names]
        if not found:
            available = ', '.join(sorted({name for names in classes.values() for name in names}))
            optional = OPTIONAL_SOLVERS.get(solver)
            fix = f'Install the `{optional}` extra.' if optional else f'Choose one of {available}.'
            code = 'solver-unavailable' if optional else 'solver-unknown'
            return [Issue(code, system.label, f'No solver is registered under {solver!r}.', fix, error=SolverError)]
        stepper = found[0]
    else:
        stepper = type(solver)
    needs = {
        'time': {kind for kind in _declared_times(system).values() if kind is not None},
        'laws': {record.law.law_kind for record in records},
        'domains': {
            domain_kind(entry, vector=entry.name in record.vectors) for record in records for entry in record.targets
        },
        'space': {_space_kind(record.space) for record in records},
        'components': {record.component for record in records},
    }
    issues = []
    for category, terms in needs.items():
        missing = stepper.supports.missing(category, terms)
        if missing:
            capable = sorted(
                name for cls, names in classes.items() if not cls.supports.missing(category, terms) for name in names
            )
            message = f'The solver {stepper.__name__} does not support the {category} {", ".join(missing)}.'
            fix = f'Use one of the solvers {", ".join(capable)}.' if capable else 'No registered solver supports them.'
            issues.append(Issue('solver-unsupported', system.label, message, fix, error=SolverError))
    return issues


def _wiener_nodes(system: System) -> list[tuple[str, Wiener]]:
    """Return the path and the element of every Wiener element of a system."""
    return [(node.path, node.component) for node in flatten_system(system) if isinstance(node.component, Wiener)]


def _matrix_problem(matrix: np.ndarray) -> str | None:
    """Return why a square covariance matrix is not symmetric positive semidefinite, or `None` when it is."""
    asymmetry = np.abs(matrix - matrix.T)
    if np.max(asymmetry) > COVARIANCE_TOLERANCE * max(1.0, float(np.max(np.abs(matrix)))):
        row, column = np.unravel_index(int(np.argmax(asymmetry)), matrix.shape)
        return f'Its covariance is not symmetric: entry ({row}, {column}) is {matrix[row, column]}.'
    values = np.linalg.eigvalsh(matrix)
    if values[0] < -COVARIANCE_TOLERANCE * max(1.0, float(np.max(np.abs(values)))):
        return f'Its covariance is not positive semidefinite: it has the negative eigenvalue {values[0]:.6g}.'
    return None


def _covariance_problem(wiener: Wiener) -> str | None:
    """Return why the dimension or the covariance of a Wiener element is invalid, or `None` when both are valid."""
    dimension = wiener.dimension
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        return f'Its dimension {dimension!r} is not a positive integer.'
    if wiener.covariance is None:
        return None
    try:
        matrix = np.asarray(wiener.covariance, dtype=float)
    except (TypeError, ValueError):
        return f'Its covariance {wiener.covariance!r} is not a matrix of numbers.'
    if matrix.shape != (dimension, dimension):
        return f'Its covariance has the shape {matrix.shape}, not ({dimension}, {dimension}).'
    return _matrix_problem(matrix)


def _wiener_issues(system: System) -> list[Issue]:
    """Return the Wiener elements outside continuous time or with an invalid covariance."""
    times = _declared_times(system)
    issues = []
    for path, wiener in _wiener_nodes(system):
        time = times.get(path.rpartition('/')[0])
        if time is not None and not wiener.requires.accepts('time', time):
            message = f'The Wiener element {path} declares Wiener processes, which exist only in continuous time.'
            issues.append(Issue('wiener-time', path, message, 'Declare continuous time for its system.'))
        problem = _covariance_problem(wiener)
        if problem is not None:
            message = f'The Wiener element {path} is invalid. {problem}'
            fix = 'Give a symmetric positive semidefinite covariance with one row and column per process.'
            issues.append(Issue('wiener-covariance', path, message, fix))
    return issues


def _law_name(record: CensusRecord) -> str:
    """Return the name of the law of a census record."""
    return f'{type(record.law).__name__} law of {record.path}'


def _process_name(key: Any) -> str:  # noqa: ANN401
    """Return the name of a process key."""
    kind, path, item = key
    return (
        f'the local process {item!r} of {path}' if kind == 'local' else f'process {item} of the Wiener element {path}'
    )


def _noise_records(records: list[CensusRecord]) -> list[CensusRecord]:
    """Return the records of noise laws."""
    return [record for record in records if isinstance(record.law, Noise)]


def _handle_issues(records: list[CensusRecord]) -> list[Issue]:
    """Return the laws that use the handle of a Wiener process outside the process of a noise law."""
    issues = []
    for record in records:
        if record.handles:
            message = (
                f'The {_law_name(record)} uses the Wiener handle {", ".join(record.handles)} in a value, but a handle '
                'is only the process of a noise law.'
            )
            fix = 'Name the handle as the process of a noise law, `Noise(target, value, process=s.<role>.dW)`.'
            issues.append(Issue('wiener-handle', record.path, message, fix))
    return issues


def _convention_issues(records: list[CensusRecord]) -> list[Issue]:
    """Return the processes that noise laws of different conventions share."""
    sharing: dict[object, list[CensusRecord]] = {}
    for record in _noise_records(records):
        for key in dict.fromkeys(record.processes):
            if key is not None:
                sharing.setdefault(key, []).append(record)
    issues = []
    for process, sharers in sharing.items():
        conventions = {cast('Noise', record.law).convention for record in sharers}
        if len(conventions) > 1:
            laws = ' and '.join(f'the {cast("Noise", item.law).convention} {_law_name(item)}' for item in sharers)
            message = f'{laws[0].upper()}{laws[1:]} share {_process_name(process)} with different conventions.'
            fix = 'Declare one convention for every noise law of the process.'
            issues.append(Issue('noise-convention-mixed', sharers[0].path, message, fix))
    return issues


def _sharer(record: CensusRecord, noisy: list[CensusRecord], wieners: dict[str, Wiener]) -> CensusRecord | None:
    """Return a noise law of another component that drives a state the amplitude of a law reads through its process."""
    for key, reads in zip(record.processes, record.reads, strict=True):
        if key is None:
            continue
        for item in noisy:
            if item.path == record.path:
                continue
            for state, process in zip(item.driven, item.processes, strict=True):
                if process is not None and state in reads and _process_covariance(key, process, wieners):
                    return item
    return None


def _scope_issues(records: list[CensusRecord], wieners: dict[str, Wiener]) -> list[Issue]:
    """Return the noise laws of a convention whose drift correction needs an amplitude of another component."""
    noisy = _noise_records(records)
    issues = []
    for record in noisy:
        if cast('Noise', record.law).convention == 'ito':
            continue
        other = _sharer(record, noisy, wieners)
        if other is not None:
            message = (
                f'The {_law_name(record)} depends on a state that the {_law_name(other)} drives through a shared '
                'process, so its drift correction needs an amplitude outside its component.'
            )
            fix = 'Use the Itô convention, or write both laws in one component.'
            issues.append(Issue('noise-correction-scope', record.path, message, fix))
    return issues


def _noise_needs(records: list[CensusRecord], wieners: dict[str, Wiener]) -> dict[NoiseKind, CensusRecord]:
    """Return the noise structures a system uses, each with the first law that needs it."""
    needs: dict[NoiseKind, CensusRecord] = {}
    states: dict[object, set[tuple[str, int]]] = {}
    first: dict[object, CensusRecord] = {}
    spread: set[object] = set()
    for record in _noise_records(records):
        for key, state in zip(record.processes, record.driven, strict=True):
            if key is None:
                needs.setdefault(NoiseKind.DIAGONAL, record)
                continue
            states.setdefault(key, set()).add(state)
            first.setdefault(key, record)
            if key[0] == 'wiener' and record.members > 1:
                spread.add(key)
            if key[0] == 'wiener' and wieners[key[1]].covariance is not None:
                needs.setdefault(NoiseKind.CORRELATED, record)
    for process, driven in states.items():
        shared = len(driven) > 1 or process in spread
        needs.setdefault(NoiseKind.SHARED if shared else NoiseKind.DIAGONAL, first[process])
    return needs


def _noise_runners(solver: object, backend: object) -> list[tuple[str, Any]]:
    """Return the solver and the backend that run the noise of a system, leaving out unknown names and backends."""
    runners: list[tuple[str, Any]] = []
    if solver == 'auto':
        runners.append(('solver', _solver_registry().resolve((Form.SDE, 'auto'))))
    elif isinstance(solver, str):
        runners += [('solver', cls) for cls, names in _solver_classes().items() if solver in names][:1]
    else:
        runners.append(('solver', type(solver)))
    if isinstance(backend, str) and backend in set(BACKENDS):
        runners.append(('backend', BACKENDS.resolve(backend)))
    elif not isinstance(backend, str):
        runners.append(('backend', type(backend)))
    return runners


def _noise_fix(term: NoiseKind) -> str:
    """Return the fix that lists the solvers and backends that support a noise structure."""
    solvers = sorted(
        name
        for cls, names in _solver_classes().items()
        if LawKind.NOISE in cls.supports.laws and term in cls.supports.noise
        for name in names
    )
    backends = sorted(str(key) for key in BACKENDS if term in BACKENDS.resolve(key).supports.noise)
    return f'Use one of the solvers {", ".join(solvers)}, with one of the backends {", ".join(backends)}.'


def _noise_support_issues(
    records: list[CensusRecord],
    wieners: dict[str, Wiener],
    solver: object,
    backend: object,
) -> list[Issue]:
    """Return the noise structures of a system that the chosen solver or backend does not support."""
    needs = _noise_needs(records, wieners)
    if not needs:
        return []
    issues = []
    for role, runner in _noise_runners(solver, backend):
        supported = getattr(runner, 'supports', Supports()).noise
        for term, record in needs.items():
            if term not in supported:
                message = f'The {_law_name(record)} needs {term} noise, which the {role} {runner.__name__} lacks.'
                issues.append(Issue('noise-unsupported', record.path, message, _noise_fix(term), error=SolverError))
    return issues


def _noise_issues(system: System, records: list[CensusRecord], solver: object, backend: object) -> list[Issue]:
    """Return the problems of the Wiener elements and the noise laws of a system."""
    wieners = dict(_wiener_nodes(system))
    issues = _wiener_issues(system) + _handle_issues(records)
    if any(issue.code == 'wiener-covariance' for issue in issues):
        return issues
    issues += _convention_issues(records)
    issues += _scope_issues(records, wieners)
    return issues + _noise_support_issues(records, wieners, solver, backend)


def validate(
    system: System,
    *,
    solver: object = 'auto',
    backend: object = 'auto',
    observables: Sequence[Observable] = (),
) -> ValidationReport:
    """Return every problem of a system, of its solver, and of its backend, before compilation.

    Validation compares the requirements every law declares with the states it targets, the space it lives on, and
    the system's time, checks that one solver runs the laws of every partition, and checks the chosen solver and
    backend against their declared support. It reports every problem at once, each with its path and fix.

    Args:
        system:
            The system.

        solver:
            `'auto'`, a registered solver name, or a stepper instance.

        backend:
            `'auto'`, a registered backend name, or a backend instance.

        observables:
            The observables of the run, labelled by their path from the root, which roles may bind to.

    Returns:
        report:
            The problems found.

    Examples:
        >>> from skcomplex.core.compile import validate
        >>> from skcomplex.models import create
        >>> validate(create('lotka_volterra')).ok
        True
        >>> [issue.code for issue in validate(create('lotka_volterra'), solver='gillespie').issues]
        ['solver-unsupported']
    """
    report = ValidationReport(_structure_issues(system))
    if report.issues or is_rewriting(system) or contains_quantum(system):
        return report
    try:
        observed = with_observables(system, observables)
    except ValidationError as error:
        report.issues.append(Issue('observable-path', system.label, str(error), 'Label observables by their path.'))
        return report
    held, _ = _with_held_observables(observed, _held_observables(observed))
    records, errors = law_census(held)
    for path, failure in errors:
        code, fix = FIXES.get(type(failure), ('value', 'Correct the component as the message describes.'))
        report.issues.append(Issue(code, path, str(failure), fix, error=type(failure)))
    report.issues += _binding_issues(system)
    report.issues += _law_issues(records, _declared_times(system))
    owners: dict[str, set[LawKind]] = {}
    for record in records:
        owners.setdefault(record.owner, set()).add(record.law.law_kind)
    report.issues += _mixed_issues(system, owners)
    if solver != 'auto':
        report.issues += _solver_issues(solver, records, system)
    report.issues += _noise_issues(system, records, solver, backend)
    if isinstance(backend, str) and backend != 'auto' and backend not in set(BACKENDS):
        message = f'No backend is registered under {backend!r}.'
        fix = f'Choose one of {_names(BACKENDS)}.'
        report.issues.append(Issue('backend-unknown', system.label, message, fix, error=BackendError))
    return report
