"""Validate systems against the requirements of their primitives and the support of solvers and backends."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ...core.base import (
    BackendError,
    BaseSpace,
    DomainKind,
    Form,
    LabelError,
    LawClass,
    LawKind,
    NoiseKind,
    NumericDomain,
    OrderKind,
    ParameterError,
    Requires,
    RoleError,
    SkcomplexError,
    SolverError,
    SpaceError,
    SpaceKind,
    SpaceTooLargeError,
    Supports,
    TimeKind,
    UnregisteredLawError,
    ValidationError,
    VariableKind,
    declarations_of,
    law_classes,
)
from ...core.components import ElementComponent, GroupComponent, Wiener
from ...core.laws import JumpLaw, NoiseLaw, neighborhoods_in, values_of
from ...core.laws._neighbourhoods import NeighborReduction
from ...core.space import Adjacency, Cone, NetworkSpace, SiteSpace
from ...core.system import Observable, System
from ._backends import BACKENDS, BaseBackend
from ._census import Census, CensusRecord
from ._compilation import Compilation
from ._issues import ERROR, WARNING, Issue, ValidationWarning, names_of
from ._layout import Entry
from ._neighbourhoods import cone_heading, position_state
from ._noise import process_covariance
from ._observables import _with_observables, held_observables, with_held_observables
from ._pipeline import _registered_passes, registered_checks, run_passes
from ._planner import _Plan, solver_classes, solver_registry
from ._system_passes import claiming_pass
from ._tree import Node, declared_times
from .passes._core import CORE_PASSES

OPTIONAL_SOLVERS = {'lammps': 'lammps'}
COVARIANCE_TOLERANCE = 1e-12
TIME_CLASSES = {
    TimeKind.CONTINUOUS: 'ContinuousTime()',
    TimeKind.DISCRETE: 'DiscreteTime()',
    TimeKind.STATIC: 'StaticTime()',
}
OWN_NEIGHBORHOOD = Requires(space={SpaceKind.LATTICE, SpaceKind.NETWORK})
FIXES: dict[type[SkcomplexError], tuple[str, str]] = {
    LabelError: ('label', 'Give every child of a system a unique label of letters, digits, and underscores.'),
    RoleError: ('role', 'Bind every role to the path of an element, group, or member of the system.'),
    SpaceError: ('space-missing', 'Give the system a space whose kind the component supports.'),
    ParameterError: ('parameter', 'Give the parameter a value inside its domain.'),
    UnregisteredLawError: (
        'law-unregistered',
        'Import the formalism that registers the law class, or register it in a formalism.',
    ),
}
REPLACEMENTS = {
    DomainKind.CATEGORICAL: 'a jump, transition, or update law',
    DomainKind.INTEGER: 'a jump or update law',
    DomainKind.BOOLEAN: 'a jump or update law',
    DomainKind.HILBERT: 'a generator or dissipator law',
}


def _subclasses(base: type) -> list[type]:
    """Return every subclass of a class, recursively."""
    found: list[type] = []
    subclasses: list[type] = base.__subclasses__()
    for subclass in subclasses:
        found += [subclass, *_subclasses(subclass)]
    return found


def _domain_kind(entry: Entry, *, vector: bool = False) -> DomainKind:
    """Return the kind of value a state of the layout holds."""
    domain = entry.domain
    if vector or entry.is_complex or domain is NumericDomain.COMPLEX:
        return DomainKind.VECTOR if vector else DomainKind.COMPLEX
    return domain.domain_kind


def _space_kind(space: BaseSpace | None) -> SpaceKind:
    """Return the kind of a space, or `none` without one."""
    return SpaceKind.NONE if space is None else space.space_kind


def _layer_issues(node: Any, space: BaseSpace | None) -> list[Issue]:  # noqa: ANN401
    """Return the issue of an interaction over neighbours that names a layer its network does not have."""
    layer = getattr(node.component, 'layer', None)
    labels = getattr(space, 'layer_labels', (None,))
    if layer is None or layer in labels:
        return []
    known = ', '.join(repr(label) for label in labels)
    message = f'The interaction runs on the layer {layer!r}, which the network does not have. Its layers are {known}.'
    fix = (
        f'Name one of the layers {known}, or add the layer with `NetworkSpace(graph, layers=[Layer({layer!r}, ...)])`.'
    )
    return [Issue('binding', node.path, message, fix)]


def _binding_issues(nodes: list[Node]) -> list[Issue]:
    """Return the interactions whose binding lacks its roles and parameters, or that the space does not offer."""
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
        bound = [by_path.get(f'{node.scope}{getattr(interaction, name, "")}') for name in roles]
        members = [item for item in bound if item is not None and getattr(item.component, 'on', None) != 'edges']
        target = next(iter(members), None)
        space = target.space if target is not None else node.space
        offered = (
            isinstance(space, SiteSpace | NetworkSpace)
            if over == 'neighbors'
            else bool(
                getattr(space, 'hyperedges', ()),
            )
        )
        if not offered:
            needs = 'a lattice or network space' if over == 'neighbors' else 'a network space with hyperedges'
            message = f'The binding over {over} needs {needs}, but the bound members live on {space!r}.'
            issues.append(Issue('binding', node.path, message, f'Give the bound group {needs}.'))
            continue
        issues += _layer_issues(node, space)
    return issues


def _bound_issues(nodes: list[Node]) -> list[Issue]:
    """Return the adaptive parameters whose initial values lie outside their bounds."""
    issues = []
    for node in nodes:
        for name, declaration in declarations_of(node.component_type, VariableKind.STATE).items():
            if declaration.bounds is None:
                continue
            lower, upper = declaration.bounds
            value = np.asarray(node.value_of(name), dtype=float)
            if np.all((value >= lower) & (value <= upper)):
                continue
            path = f'{node.path}.{name}'
            message = (
                f'The adaptive parameter {path} starts at {value.tolist()}, outside its bounds ({lower}, {upper}).'
            )
            fix = f'Give {path} an initial value inside its bounds ({lower}, {upper}).'
            issues.append(Issue('parameter-bounds', path, message, fix, error=ParameterError))
    return issues


def _space_issues(system: System, nodes: list[Node]) -> list[Issue]:
    """Return the visuals that need options their space lacks, and the spaces whose own check fails."""
    issues = []
    spaces: dict[int, tuple[str, BaseSpace]] = {}
    for node in nodes:
        space = getattr(node.component, 'space', None) or node.space
        visual = getattr(node.component, 'visual', None)
        if space is not None:
            spaces.setdefault(id(space), (node.path, space))
        if space is not None and 'image' in space.lacks and getattr(visual, 'kind', None) == 'image':
            message = (
                f'The image visual of {node.path} needs a regular lattice, but its space is {type(space).__name__}.'
            )
            fix = f'Use the {space.default_visual!r} visual.' if isinstance(space, SiteSpace) else 'Use another visual.'
            issues.append(Issue('visual-unsupported', node.path, message, fix))
    if system.space is not None:
        spaces.setdefault(id(system.space), (system.label, system.space))
    for path, space in spaces.values():
        try:
            getattr(space, 'check', lambda: None)()
        except SpaceTooLargeError as error:
            fix = 'Ask for a smaller ball, or raise max_sites.'
            issues.append(Issue('space-too-large', path, str(error), fix, error=SpaceTooLargeError))
        except ValueError as error:
            fix = f'Give the {type(space).__name__} a shape, a boundary, and a geometry that it supports.'
            holder = path.rpartition('/')[0] or system.label
            issues.append(Issue('space-invalid', holder, str(error), fix, error=SpaceError))
    return issues


def _law_issues(records: list[CensusRecord], times: dict[str, TimeKind | None]) -> list[Issue]:
    """Return the laws whose targets, space, or time their requirements do not accept."""
    issues = []
    for record in records:
        time = times.get(record.path.rpartition('/')[0])
        law, requires = record.law, record.law.requires
        name = f'{type(law).__name__} of {record.path}'
        for entry in record.targets:
            kind = _domain_kind(entry, vector=entry.name in record.vectors)
            if not requires.accepts('domains', kind):
                fix = f'Use {REPLACEMENTS[kind]}.' if kind in REPLACEMENTS else 'Target a state of a supported domain.'
                message = (
                    f'The {name} targets {entry.name}, a {kind} state, but {law.law_kind} laws target '
                    f'{names_of(requires.domains)} states.'
                )
                issues.append(Issue('law-domain', record.path, message, fix))
        space = _space_kind(record.space)
        if not requires.accepts('space', space):
            message = f'The {name} needs a {names_of(requires.space)} space, but the component lives on {space}.'
            issues.append(Issue('law-space', record.path, message, 'Give the system a space of a supported kind.'))
        if time is not None and not requires.accepts('time', time):
            message = f'The {name} is a {law.time_class} law, which needs {names_of(requires.time)} time.'
            kinds = {
                str(item.law_kind)
                for item in law_classes()
                if item.requires.time and item.requires.accepts('time', time)
            }
            classes = ' or '.join(f'`{TIME_CLASSES[kind]}`' for kind in sorted(requires.time))
            fix = (
                f'Declare {names_of(requires.time)} time with {classes}, or use a law of {time} time: '
                f'{names_of(kinds)}.'
            )
            issues.append(Issue('law-time', record.path, message, fix))
    return issues


def _reductions_of(value: object) -> list[NeighborReduction]:
    """Return every neighbourhood reduction of a law value, nested ones included."""
    found = []
    stack = neighborhoods_in(value)
    while stack:
        reduction = stack.pop()
        found.append(reduction)
        stack += neighborhoods_in(reduction.argument)
        stack += neighborhoods_in(reduction.weight) if reduction.weight is not None else []
    return found


def _space_fix(requires: Requires) -> str:
    """Return the fix of a neighbourhood that the space does not offer."""
    if SpaceKind.CONTINUOUS in requires.space:
        return (
            'Give the group a continuous space, such as `ContinuousSpace(10.0, 10.0)`, or reduce over the neighbours '
            'of its network or lattice with `over=None`, or over a layer with `Adjacency(layer=label)`.'
        )
    return (
        'Give the group a network or lattice space, or reduce over `Radius(r)`, `Nearest(k)`, or `Cone(r, angle)` '
        'in a continuous space, or over a layer of a network with `Adjacency(layer=label)`.'
    )


def _layer_reduction_issue(name: str, path: str, over: object, space: object) -> Issue | None:
    """Return the issue of a reduction over a layer that its network does not have, or `None`."""
    if not isinstance(over, Adjacency) or not isinstance(space, NetworkSpace) or over.layer in space.layer_labels:
        return None
    known = ', '.join(repr(label) for label in space.layer_labels)
    message = (
        f'The {name} reduces over the layer {over.layer!r}, which the network does not have. Its layers are {known}.'
    )
    fix = f'Reduce over one of the layers {known}, or add the layer to the network space.'
    return Issue('binding', path, message, fix)


def _neighborhood_issues(records: list[CensusRecord], nodes: list[Node]) -> list[Issue]:
    """Return the neighbourhood reductions whose space does not offer them, and the cones without a heading."""
    by_path = {node.path: node for node in nodes}
    issues = []
    for record in records:
        name = f'{type(record.law).__name__} of {record.path}'
        space = _space_kind(record.space)
        values = values_of(record.law, include=('rate',) if isinstance(record.law, JumpLaw) else ('value',))
        reductions = {str(item.call): item for value in values for item in _reductions_of(value)}
        for reduction in reductions.values():
            over = reduction.over
            requires = OWN_NEIGHBORHOOD if over is None else over.requires
            described = 'the neighbours of its space' if over is None else repr(over)
            if not requires.accepts('space', space):
                message = (
                    f'The {name} reduces over {described}, which needs a {names_of(requires.space)} space, but the '
                    f'component lives on {space}.'
                )
                issues.append(Issue('law-space', record.path, message, _space_fix(requires)))
                continue
            layered = _layer_reduction_issue(name, record.path, over, record.space)
            if layered is not None:
                issues.append(layered)
                continue
            node = by_path.get(record.owner)
            if node is None or space is not SpaceKind.CONTINUOUS or over is None:
                continue
            states = declarations_of(node.component_type, VariableKind.STATE)
            if position_state(states) is None:
                message = (
                    f'The {name} reduces over {over!r}, which measures distances between positions, but '
                    f'{record.owner} declares no vector position state.'
                )
                fix = 'Declare the position with `state(vector=True, position=True)`.'
                issues.append(Issue('law-domain', record.path, message, fix))
                continue
            if isinstance(over, Cone):
                try:
                    cone_heading(over, states, getattr(record.space, 'n_dim', 0))
                except ValidationError as error:
                    fix = (
                        "Declare a vector velocity, `state(vector=True, velocity_of='x')`, or name a heading "
                        "state, such as `Cone(r, angle, heading='theta')`."
                    )
                    issues.append(Issue('law-domain', record.path, f'The {name}: {error}', fix))
    return issues


def _solver_issues(solver: object, records: list[CensusRecord], system: System) -> list[Issue]:
    """Return the requirements of a system that a chosen solver does not support."""
    classes = solver_classes()
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
        'time': {kind for kind in declared_times(system).values() if kind is not None},
        'laws': {record.law.law_kind for record in records},
        'domains': {
            _domain_kind(entry, vector=entry.name in record.vectors) for record in records for entry in record.targets
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


def _order_issues(nodes: list[Node]) -> list[Issue]:
    """Return the states of a fractional order when no registered solver integrates fractional orders."""
    if any(OrderKind.FRACTIONAL in cls.supports.orders and cls.form is not Form.COUPLED for cls in solver_classes()):
        return []
    fractional = [
        (node.path, name, declaration.order)
        for node in nodes
        if isinstance(node.component, ElementComponent | GroupComponent)
        for name, declaration in declarations_of(node.component_type, VariableKind.STATE).items()
        if declaration.is_fractional
    ]
    fix = 'Import the formalism of fractional orders, `skcomplex.formalisms.fractional`.'
    return [
        Issue(
            'order-unsupported',
            path,
            f'The state {path}.{name} declares the fractional order {order}, but no registered solver integrates '
            'fractional orders.',
            fix,
            error=SolverError,
        )
        for path, name, order in fractional
    ]


def _wiener_nodes(nodes: list[Node]) -> list[tuple[str, Wiener]]:
    """Return the path and the element of every Wiener element of a system."""
    return [(node.path, node.component) for node in nodes if isinstance(node.component, Wiener)]


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


def _wiener_issues(system: System, nodes: list[Node]) -> list[Issue]:
    """Return the Wiener elements outside continuous time or with an invalid covariance."""
    times = declared_times(system)
    issues = []
    for path, wiener in _wiener_nodes(nodes):
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
    return f'{type(record.law).__name__} of {record.path}'


def _process_name(key: Any) -> str:  # noqa: ANN401
    """Return the name of a process key."""
    kind, path, item = key
    return (
        f'the local process {item!r} of {path}' if kind == 'local' else f'process {item} of the Wiener element {path}'
    )


def _noise_records(records: list[CensusRecord]) -> list[CensusRecord]:
    """Return the records of noise laws."""
    return [record for record in records if isinstance(record.law, NoiseLaw)]


def _handle_issues(records: list[CensusRecord]) -> list[Issue]:
    """Return the laws that use the handle of a Wiener process outside the process of a noise law."""
    issues = []
    for record in records:
        if record.handles:
            message = (
                f'The {_law_name(record)} uses the Wiener handle {", ".join(record.handles)} in a value, but a handle '
                'is only the process of a noise law.'
            )
            fix = 'Name the handle as the process of a noise law, `NoiseLaw(target, value, process=s.<role>.dW)`.'
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
        conventions = {cast('NoiseLaw', record.law).convention for record in sharers}
        if len(conventions) > 1:
            laws = ' and '.join(f'the {cast("NoiseLaw", item.law).convention} {_law_name(item)}' for item in sharers)
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
                if process is not None and state in reads and process_covariance(key, process, wieners):
                    return item
    return None


def _scope_issues(records: list[CensusRecord], wieners: dict[str, Wiener]) -> list[Issue]:
    """Return the noise laws of a convention whose drift correction needs an amplitude of another component."""
    noisy = _noise_records(records)
    issues = []
    for record in noisy:
        if cast('NoiseLaw', record.law).convention == 'ito':
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
        if cast('NoiseLaw', record.law).conserved:
            needs.setdefault(NoiseKind.CONSERVED, record)
            continue
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
        runners.append(('solver', solver_registry().resolve((Form.SDE, 'auto'))))
    elif isinstance(solver, str):
        runners += [('solver', cls) for cls, names in solver_classes().items() if solver in names][:1]
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
        for cls, names in solver_classes().items()
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


def _noise_issues(compilation: Compilation, records: list[CensusRecord]) -> list[Issue]:
    """Return the problems of the Wiener elements and the noise laws of a system."""
    nodes = cast('list[Node]', compilation.nodes)
    solver, backend = compilation.solver, compilation.backend
    wieners = dict(_wiener_nodes(nodes))
    issues = _wiener_issues(compilation.system, nodes) + _handle_issues(records)
    if any(issue.code == 'wiener-covariance' for issue in issues):
        return issues
    issues += _convention_issues(records)
    issues += _scope_issues(records, wieners)
    return issues + _noise_support_issues(records, wieners, solver, backend)


def _counted(
    system: System,
    solver: object,
    backend: object,
    observables: Sequence[Observable],
) -> tuple[Compilation | None, list[Issue]]:
    """Return the compilation of a system through the census of its laws, or the issues that stop it before.

    The compilation stops after flattening for a system that a whole-system pass compiles.
    """
    try:
        observed = _with_observables(system, observables)
    except ValidationError as error:
        return None, [Issue('observable-path', system.label, str(error), 'Label observables by their path.')]
    passes = _registered_passes(CORE_PASSES)
    compilation = Compilation(observed, cast('str | BaseBackend', backend), solver)
    try:
        run_passes(compilation, passes, until='nodes')
    except SkcomplexError as error:
        code, fix = FIXES.get(
            type(error),
            ('structure', 'Pass elements, groups, interactions, and systems as children.'),
        )
        return None, [Issue(code, system.label, str(error), fix, error=type(error))]
    if claiming_pass(observed) is not None:
        return compilation, []
    held = held_observables(observed, cast('list[Node]', compilation.nodes))
    if held:
        tree, paths = with_held_observables(observed, held)
        compilation = Compilation(tree, cast('str | BaseBackend', backend), solver, held=tuple(paths))
    return run_passes(compilation, passes, until='census'), []


def _plan_issues(compilation: Compilation) -> list[Issue]:
    """Return the issues of the partitions of a system, lowering its laws as far as their classes decide its form."""
    try:
        run_passes(compilation, _registered_passes(CORE_PASSES), until='plan')
    except SkcomplexError as error:
        code, fix = FIXES.get(type(error), ('value', 'Correct the component as the message describes.'))
        return [Issue(code, compilation.system.label, str(error), fix, error=type(error))]
    return list(cast('_Plan', compilation.plan).issues)


def _census_issues(compilation: Compilation) -> list[Issue]:
    """Return the issues of the components that fail to declare or collect, unless their binding is the problem."""
    census = cast('Census', compilation.census)
    bindings = _binding_issues(cast('list[Node]', compilation.nodes))
    unbound = {issue.path for issue in bindings}
    issues = []
    for path, failure in census.errors:
        if path in unbound:
            continue
        code, fix = FIXES.get(type(failure), ('value', 'Correct the component as the message describes.'))
        issues.append(Issue(code, path, str(failure), fix, error=type(failure)))
    return issues + bindings


def _issues_of(compilation: Compilation) -> list[Issue]:
    """Return the issues of a compilation planned past the census."""
    system, solver, backend = compilation.system, compilation.solver, compilation.backend
    nodes = cast('list[Node]', compilation.nodes)
    records = cast('Census', compilation.census).records
    issues = _census_issues(compilation)
    issues += _bound_issues(nodes)
    issues += _law_issues(records, declared_times(system))
    issues += _neighborhood_issues(records, nodes)
    issues += _space_issues(system, nodes)
    issues += _order_issues(nodes)
    issues += [issue for check in registered_checks() for issue in check(compilation)]
    if solver != 'auto':
        issues += _solver_issues(solver, records, system)
    issues += _noise_issues(compilation, records)
    if isinstance(backend, str) and backend != 'auto' and backend not in set(BACKENDS):
        message = f'No backend is registered under {backend!r}.'
        fix = f'Choose one of {names_of(BACKENDS)}.'
        issues.append(Issue('backend-unknown', system.label, message, fix, error=BackendError))
    return issues


def missing_declarations() -> list[str]:
    """Return the primitives, solvers, and backends that do not declare their requirements or support.

    Every law class declares its law kind, its time class, and its requirements, every space its kind, every registered
    solver and backend its support. Validation reads only these declarations, so a primitive without one cannot be
    validated.

    Returns:
        missing:
            One sentence per missing declaration.

    Examples:
        >>> from skcomplex.engine.compile import missing_declarations
        >>> missing_declarations()
        []
    """
    missing = []
    for law in law_classes():
        if not isinstance(getattr(law, 'law_kind', None), LawKind):
            missing.append(f'The law {law.__name__} declares no law kind.')
        if not isinstance(getattr(law, 'time_class', None), LawClass):
            missing.append(f'The law {law.__name__} declares no time class.')
        if not isinstance(getattr(law, 'requires', None), Requires):
            missing.append(f'The law {law.__name__} declares no requirements.')
    missing.extend(
        f'The space {space.__name__} declares no space kind.'
        for space in _subclasses(BaseSpace)
        if not inspect.isabstract(space) and not isinstance(getattr(space, 'space_kind', None), SpaceKind)
    )
    solvers = solver_registry()
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


def validate(
    system: System,
    *,
    solver: object = 'auto',
    backend: object = 'auto',
    observables: Sequence[Observable] = (),
) -> 'ValidationReport':
    """Return every problem of a system, of its solver, and of its backend, before compilation.

    Validation compares the requirements every law declares with the states it targets, the space it lives on, and
    the system's time, checks that one solver runs the laws of every partition, and checks the chosen solver and
    backend against their declared support. A state of a fractional order needs an order in `(0, 1)`, continuous time,
    and a solver and a backend that integrate fractional orders. It reports every problem at once, each with its path
    and fix, together with the compilation state planned so far, which compilation continues from. A system without
    other errors is lowered as far as the classes of its laws decide its partitions.

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
            The problems found, with the compilation state.

    Examples:
        >>> from skcomplex.engine.compile import validate
        >>> from skcomplex.models import create
        >>> validate(create('lotka_volterra')).ok
        True
        >>> [issue.code for issue in validate(create('lotka_volterra'), solver='gillespie').issues]
        ['solver-unsupported']
    """
    compilation, issues = _counted(system, solver, backend, observables)
    if compilation is None or compilation.census is None:
        return ValidationReport(issues, compilation)
    issues = _issues_of(compilation)
    if not any(issue.severity == ERROR for issue in issues):
        issues += _plan_issues(compilation)
    return ValidationReport(issues, compilation)


@dataclass
class ValidationReport:
    """Every problem of a system, its solver, and its backend, with the compilation state planned so far.

    Args:
        issues:
            The problems, in the order they were found.

        compilation:
            The compilation state through the plan of the partitions, which compilation continues from, or `None`
            when the system tree cannot be flattened.

    Examples:
        >>> from skcomplex.engine.compile import validate
        >>> from skcomplex.core.system import System
        >>> validate(System()).ok
        True
    """

    issues: list[Issue] = field(default_factory=list)
    compilation: Compilation | None = field(default=None, repr=False, compare=False)

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
