"""Validate systems against the requirements of their primitives and the support of solvers and backends."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

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
from ..model import (
    Auxiliary,
    Boundary,
    Dissipator,
    Drive,
    Emission,
    Equation,
    Event,
    Extremal,
    Flux,
    GraphProduction,
    Hamiltonian,
    Invariant,
    Jump,
    LatticeSpace,
    LocalRule,
    Match,
    MesaSystem,
    NetworkSpace,
    Noise,
    Observable,
    Production,
    Rate,
    Relaxation,
    Rule,
    Sample,
    Space,
    Substitution,
    System,
    Transfer,
    Transition,
    Update,
)
from ._backends import BACKENDS
from ._compiler import CensusRecord, _held_observables, _with_held_observables, law_census
from ._layout import Entry
from ._observables import with_observables
from ._quantum import contains_quantum
from ._rewriting import is_rewriting
from ._tree import flatten_system

OPTIONAL_SOLVERS = {'lammps': 'lammps'}


def _solver_registry() -> Registry:
    """Return the registry of solvers, imported late since the solvers import the compiler."""
    from ..solvers import SOLVERS  # noqa: PLC0415

    return SOLVERS


def law_types() -> tuple[type, ...]:
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
        Hamiltonian,
        Dissipator,
        Production,
        Substitution,
        GraphProduction,
    )


def _subclasses(base: type) -> list[type]:
    """Return every subclass of a class, recursively."""
    found = []
    for subclass in base.__subclasses__():
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
            missing.append(f'The solver {key[1]!r} ({solver.__name__}) declares no support.')
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
        >>> from skcomplex.core.model import System
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
    return {
        Domain.INTEGER: DomainKind.INTEGER,
        Domain.NONNEGATIVE_INTEGER: DomainKind.INTEGER,
        Domain.BOOLEAN: DomainKind.BOOLEAN,
        Domain.ANGLE: DomainKind.ANGLE,
    }.get(domain, DomainKind.REAL)


def _space_kind(space: object) -> SpaceKind:
    """Return the kind of a space, or `none` without one."""
    return SpaceKind.NONE if space is None else space.space_kind


def _names(terms: Iterable[object]) -> str:
    """Return terms as a sorted, comma-separated list."""
    return ', '.join(sorted(str(term) for term in terms))


def _solver_classes() -> dict[type, list[str]]:
    """Return every registered solver class with its names, without the `'auto'` aliases."""
    classes: dict[type, list[str]] = {}
    solvers = _solver_registry()
    for key in solvers:
        if key[1] != 'auto':
            classes.setdefault(solvers.resolve(key), []).append(key[1])
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
    time = system.time.kind if getattr(system, 'time', None) is not None else inherited
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
    if isinstance(system, MesaSystem):
        return ValidationReport()
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
    for path, error in errors:
        code, fix = FIXES.get(type(error), ('value', 'Correct the component as the message describes.'))
        report.issues.append(Issue(code, path, str(error), fix, error=type(error)))
    report.issues += _binding_issues(system)
    report.issues += _law_issues(records, _declared_times(system))
    owners: dict[str, set[LawKind]] = {}
    for record in records:
        owners.setdefault(record.owner, set()).add(record.law.law_kind)
    report.issues += _mixed_issues(system, owners)
    if solver != 'auto':
        report.issues += _solver_issues(solver, records, system)
    if isinstance(backend, str) and backend != 'auto' and backend not in set(BACKENDS):
        message = f'No backend is registered under {backend!r}.'
        fix = f'Choose one of {_names(BACKENDS)}.'
        report.issues.append(Issue('backend-unknown', system.label, message, fix, error=BackendError))
    return report
