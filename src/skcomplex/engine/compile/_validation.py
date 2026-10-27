"""Validate systems against the requirements of their primitives and the support of solvers and backends."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from sage.all__sagemath_symbolics import CC, Expression
from sage.modules.free_module_element import FreeModuleElement

from ...core.errors import (
    LabelError,
    ParameterError,
    RoleError,
    SkcomplexError,
    SolverError,
    SpaceError,
    SpaceTooLargeError,
    ValidationError,
)
from ...core.laws import JumpLaw, NoiseLaw
from ...core.laws.utils import NeighborReduction, neighborhoods_in, reductions_in
from ...core.spaces import (
    AdjacencyNeighborhood,
    BaseSpace,
    CayleySpace,
    ConeNeighborhood,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
    StaticTime,
)
from ...core.specs import ObservableSpec, WienerProcessSpec
from ...core.system import System
from ..backends import BaseBackend
from ..problems import (
    DomainKind,
    Entry,
    NoiseKind,
    SupportSpec,
    VariableKind,
    declarations_of,
    domain_kind,
    spaces_of,
    symbol,
)
from ..problems._neighbourhoods import cone_heading, position_state
from ._census import Census, CensusRecord
from ._compilation import Compilation
from ._issues import ERROR, WARNING, Issue, ValidationWarning
from ._laws import values_of
from ._noise import process_covariance
from ._observables import (
    _HeldReduction,
    _with_observables,
    held_observables,
    held_reduction,
    with_held_observables,
    with_held_reductions,
)
from ._pipeline import run_passes
from ._planner import _Plan
from ._tree import Node, declared_times, noise_processes
from .passes._core import CHECKS, PASSES, claiming_compiler

COVARIANCE_TOLERANCE = 1e-12
TIME_CLASSES = {
    ContinuousTime: 'ContinuousTime()',
    DiscreteTime: 'DiscreteTime()',
    StaticTime: 'StaticTime()',
}
OWN_NEIGHBORHOOD = (LatticeSpace, NetworkSpace, CayleySpace, MeshSpace)
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


def _subclasses(base: type) -> list[type]:
    """Return every subclass of a class, recursively."""
    found: list[type] = []
    subclasses: list[type] = base.__subclasses__()
    for subclass in subclasses:
        found += [subclass, *_subclasses(subclass)]
    return found


def _domain_kind(entry: Entry, *, vector: bool = False) -> DomainKind:
    """Return the kind of value a state of the layout holds."""
    if vector or entry.is_complex or entry.domain is CC:
        return DomainKind.VECTOR if vector else DomainKind.COMPLEX
    return domain_kind(entry.domain, entry.period)


def _names(classes: tuple[type | None, ...]) -> str:
    """Return the names of classes joined by `or`, with `no space` for `None`."""
    return ' or '.join('no space' if item is None else item.__name__ for item in classes)


def _layer_issues(node: Any, space: BaseSpace | None) -> list[Issue]:  # noqa: ANN401
    """Return the issue of an interaction over neighbours that names a layer its network does not have."""
    layer = getattr(node.component, 'layer', None)
    labels = tuple(getattr(space, 'graphs_', (None,)))
    if layer is None or layer in labels:
        return []
    known = ', '.join(repr(label) for label in labels)
    message = f'The interaction runs on the layer {layer!r}, which the network does not have. Its layers are {known}.'
    fix = f'Name one of the layers {known}, or add the layer with `NetworkSpace(graph, layers=[LayerSpec({layer!r}, ...)])`.'
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
            isinstance(space, LatticeSpace | NetworkSpace)
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
        if space is not None and 'image' in space._get_lacks() and getattr(visual, 'kind', None) == 'image':
            message = (
                f'The image visual of {node.path} needs a regular lattice, but its space is {type(space).__name__}.'
            )
            default = getattr(space, 'default_visual_', None)
            fix = f'Use the {default!r} visual.' if default is not None else 'Use another visual.'
            issues.append(Issue('visual-unsupported', node.path, message, fix))
    if system.space is not None:
        spaces.setdefault(id(system.space), (system.label, system.space))
    for path, space in spaces.values():
        try:
            space.check()
        except SpaceTooLargeError as error:
            fix = 'Ask for a smaller ball, or raise max_sites.'
            issues.append(Issue('space-too-large', path, str(error), fix, error=SpaceTooLargeError))
        except ValidationError as error:
            fix = f'Give the {type(space).__name__} a shape, a boundary, and a geometry that it supports.'
            holder = path.rpartition('/')[0] or system.label
            issues.append(Issue('space-invalid', holder, str(error), fix, error=SpaceError))
    return issues


def _law_issues(records: list[CensusRecord], times: dict[str, type | None]) -> list[Issue]:
    """Return the laws whose space or time their requirements do not accept."""
    issues = []
    for record in records:
        time = times.get(record.path.rpartition('/')[0])
        law = record.law
        requires = law._get_requirements()
        name = f'{type(law).__name__} of {record.path}'
        if requires.space and not isinstance(record.space, requires.space):
            lives = type(record.space).__name__ if record.space is not None else 'no space'
            message = f'The {name} needs a {_names(requires.space)}, but the component lives on {lives}.'
            issues.append(Issue('law-space', record.path, message, 'Give the system a space of a supported class.'))
        if time is not None and requires.time and not issubclass(time, requires.time):
            message = f'The {name} needs {_names(requires.time)}, but its system declares {time.__name__}.'
            classes = ' or '.join(f'`{TIME_CLASSES[item]}`' for item in requires.time)
            fix = f'Declare {classes} for its system, or use laws that run in {time.__name__}.'
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


def _space_fix(spaces: tuple[type, ...]) -> str:
    """Return the fix of a neighbourhood that the space does not offer."""
    if ContinuousSpace in spaces:
        return (
            'Give the group a continuous space, such as `ContinuousSpace(extent=(10.0, 10.0))`, or reduce over the '
            'neighbours of its network or lattice with `over=None`, or over a layer with `AdjacencyNeighborhood(layer)`.'
        )
    return (
        'Give the group a network or lattice space, or reduce over a `RadiusNeighborhood`, a `NearestNeighborhood`, or '
        'a `ConeNeighborhood` in a continuous space, or over a layer of a network with `AdjacencyNeighborhood(layer)`.'
    )


def _layer_reduction_issue(name: str, path: str, over: object, space: object) -> Issue | None:
    """Return the issue of a reduction over a layer that its network does not have, or `None`."""
    if (
        not isinstance(over, AdjacencyNeighborhood)
        or not isinstance(space, NetworkSpace)
        or over.layer in space.graphs_
    ):
        return None
    known = ', '.join(repr(label) for label in space.graphs_)
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
        values = values_of(record.law, include=('rate',) if isinstance(record.law, JumpLaw) else ('value',))
        reductions = {str(item.call): item for value in values for item in _reductions_of(value)}
        for reduction in reductions.values():
            over = reduction.over
            spaces = OWN_NEIGHBORHOOD if over is None else spaces_of(over)
            described = 'the neighbours of its space' if over is None else repr(over)
            if not isinstance(record.space, spaces):
                lives = type(record.space).__name__ if record.space is not None else 'no space'
                message = (
                    f'The {name} reduces over {described}, which needs a {_names(spaces)}, but the component lives on '
                    f'{lives}.'
                )
                issues.append(Issue('law-space', record.path, message, _space_fix(spaces)))
                continue
            layered = _layer_reduction_issue(name, record.path, over, record.space)
            if layered is not None:
                issues.append(layered)
                continue
            node = by_path.get(record.owner)
            if node is None or not isinstance(record.space, ContinuousSpace) or over is None:
                continue
            states = declarations_of(node.component_type, VariableKind.STATE)
            if position_state(states) is None:
                message = (
                    f'The {name} reduces over {over!r}, which measures distances between positions, but '
                    f'{record.owner} declares no vector position state.'
                )
                fix = 'Declare the position with `StateSpec(vector=True, position=True)`.'
                issues.append(Issue('law-domain', record.path, message, fix))
                continue
            if isinstance(over, ConeNeighborhood):
                try:
                    cone_heading(over, states, getattr(record.space, 'n_dim_', 0))
                except ValidationError as error:
                    fix = (
                        "Declare a vector velocity, `StateSpec(vector=True, velocity_of='x')`, or name a heading "
                        "state, such as `ConeNeighborhood(r, angle, heading='theta')`."
                    )
                    issues.append(Issue('law-domain', record.path, f'The {name}: {error}', fix))
    return issues


def _solver_issues(solver: object, census: Census, system: System) -> list[Issue]:
    """Return the requirements of a system that the solver passed does not support."""
    records = census.records
    stepper = type(solver)
    needs = {
        'time': {kind for kind in declared_times(system).values() if kind is not None},
        'laws': {type(record.law) for record in records},
        'space': {None if record.space is None else type(record.space) for record in records},
    }
    issues = []
    for category, terms in needs.items():
        missing = stepper.supports.missing(category, terms)
        if missing:
            message = f'The solver {stepper.__name__} does not support the {category} {", ".join(missing)}.'
            fix = f'Pass a solver that supports the {category} {", ".join(missing)}.'
            issues.append(Issue('solver-unsupported', system.label, message, fix, error=SolverError))
    return issues


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


def _covariance_problem(wiener: WienerProcessSpec) -> str | None:
    """Return why the dimension or the covariance of a `WienerProcessSpec` is invalid, or `None` when both are valid."""
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
    """Return the Wiener processes outside continuous time or with an invalid covariance."""
    times = declared_times(system)
    issues = []
    for path, wiener in noise_processes(nodes).items():
        time = times.get(path.rpartition('/')[0])
        if time is not None and not issubclass(time, ContinuousTime):
            message = f'The Wiener processes {path} exist only in continuous time.'
            issues.append(Issue('wiener-time', path, message, 'Declare continuous time for its system.'))
        problem = _covariance_problem(wiener)
        if problem is not None:
            message = f'The Wiener processes {path} are invalid. {problem}'
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
        f'the local process {item!r} of {path}' if kind == 'local' else f'process {item} of the Wiener processes {path}'
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


def _sharer(
    record: CensusRecord,
    noisy: list[CensusRecord],
    wieners: dict[str, WienerProcessSpec],
) -> CensusRecord | None:
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


def _scope_issues(records: list[CensusRecord], wieners: dict[str, WienerProcessSpec]) -> list[Issue]:
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


def _noise_needs(records: list[CensusRecord], wieners: dict[str, WienerProcessSpec]) -> dict[NoiseKind, CensusRecord]:
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


def _noise_runners(solver: object, backend: object) -> list[tuple[str, type]]:
    """Return the classes of the solver and the backend passed that run the noise of a system."""
    runners: list[tuple[str, type]] = []
    if solver is not None:
        runners.append(('solver', type(solver)))
    if backend is not None:
        runners.append(('backend', type(backend)))
    return runners


def _noise_fix(term: NoiseKind) -> str:
    """Return the fix of a noise structure that the solver or the backend passed does not support."""
    return f'Pass a solver and a backend that support {term} noise.'


def _noise_support_issues(
    records: list[CensusRecord],
    wieners: dict[str, WienerProcessSpec],
    solver: object,
    backend: object,
) -> list[Issue]:
    """Return the noise structures of a system that the chosen solver or backend does not support."""
    needs = _noise_needs(records, wieners)
    if not needs:
        return []
    issues = []
    for role, runner in _noise_runners(solver, backend):
        supported = getattr(runner, 'supports', SupportSpec()).noise
        for term, record in needs.items():
            if term not in supported:
                message = f'The {_law_name(record)} needs {term} noise, which the {role} {runner.__name__} lacks.'
                issues.append(Issue('noise-unsupported', record.path, message, _noise_fix(term), error=SolverError))
    return issues


def _noise_issues(compilation: Compilation, records: list[CensusRecord]) -> list[Issue]:
    """Return the problems of the Wiener processes and the noise laws of a system."""
    nodes = cast('list[Node]', compilation.nodes)
    solver, backend = compilation.solver, compilation.backend
    wieners = noise_processes(nodes)
    issues = _wiener_issues(compilation.system, nodes) + _handle_issues(records)
    if any(issue.code == 'wiener-covariance' for issue in issues):
        return issues
    issues += _convention_issues(records)
    issues += _scope_issues(records, wieners)
    return issues + _noise_support_issues(records, wieners, solver, backend)


def _law_values(record: CensusRecord) -> Iterator[object]:
    """Yield the symbolic values of the law of a census record, with the items of vectors."""
    for value in values_of(record.law):
        if isinstance(value, FreeModuleElement):
            yield from value
        elif isinstance(value, Expression):
            yield value


def _held_reductions(census: Census, time: Expression) -> list[_HeldReduction]:
    """Return the reductions of agents that the laws of a census read, each once, in the order the laws read them."""
    found: dict[tuple[str, ...], _HeldReduction] = {}
    for record in census.records:
        for value in _law_values(record):
            for reduction in reductions_in(value):
                held = held_reduction(reduction, census.agents, time, record.path)
                if held is not None:
                    found.setdefault(held.key, held)
    return list(found.values())


def _counted(
    system: System,
    solver: object,
    backend: object,
    observables: Sequence[ObservableSpec],
) -> tuple[Compilation | None, list[Issue]]:
    """Return the compilation of a system through the census of its laws, or the issues that stop it before.

    The compilation stops after flattening for a system that a whole-system pass compiles. When laws read reductions of
    agents, the compilation starts again from a system tree that holds each of them in an element.
    """
    try:
        observed = _with_observables(system, observables)
    except ValidationError as error:
        return None, [Issue('observable-path', system.label, str(error), 'Label observables by their path.')]
    passes = PASSES
    compilation = Compilation(observed, cast('BaseBackend | None', backend), solver)
    try:
        run_passes(compilation, passes, until='nodes')
    except SkcomplexError as error:
        code, fix = FIXES.get(
            type(error),
            ('structure', 'Pass elements, groups, interactions, and systems as children.'),
        )
        return None, [Issue(code, system.label, str(error), fix, error=type(error))]
    if claiming_compiler(observed) is not None:
        return compilation, []
    held = held_observables(observed, cast('list[Node]', compilation.nodes))
    if held:
        tree, paths = with_held_observables(observed, held)
        compilation = Compilation(tree, cast('BaseBackend | None', backend), solver, held=tuple(paths))
    run_passes(compilation, passes, until='census')
    try:
        reductions = _held_reductions(cast('Census', compilation.census), symbol('t'))
    except ValidationError as error:
        code, fix = FIXES.get(type(error), ('reduction', 'Reduce an expression of the agents of one group.'))
        return None, [Issue(code, system.label, str(error), fix, error=type(error))]
    if not reductions:
        return compilation, []
    tree = with_held_reductions(compilation.system, reductions)
    compilation = Compilation(tree, cast('BaseBackend | None', backend), solver, held=compilation.held)
    return run_passes(compilation, passes, until='census'), []


def _plan_issues(compilation: Compilation) -> list[Issue]:
    """Return the issues of the partitions of a system, lowering its laws as far as their classes decide its form."""
    try:
        run_passes(compilation, PASSES, until='plan')
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
    system, solver = compilation.system, compilation.solver
    nodes = cast('list[Node]', compilation.nodes)
    records = cast('Census', compilation.census).records
    issues = _census_issues(compilation)
    issues += _bound_issues(nodes)
    issues += _law_issues(records, declared_times(system))
    issues += _neighborhood_issues(records, nodes)
    issues += _space_issues(system, nodes)
    issues += [issue for check in CHECKS for issue in check(compilation)]
    if solver is not None:
        issues += _solver_issues(solver, cast('Census', compilation.census), system)
    issues += _noise_issues(compilation, records)
    return issues


def validate(
    system: System,
    *,
    solver: object = None,
    backend: BaseBackend | None = None,
    observables: Sequence[ObservableSpec] = (),
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
            The solver instance, or `None` for the default solver of the system's form.

        backend:
            The backend instance, or `None` for the backend that suits the size of the system.

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
        >>> from skcomplex.engine.solvers import GillespieSolver
        >>> [issue.code for issue in validate(create('lotka_volterra'), solver=GillespieSolver()).issues]
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
            warnings.warn(issue.text, ValidationWarning, stacklevel=3)
        if self.errors:
            lines = '\n'.join(f'- {issue.text}' for issue in self.errors)
            error_msg = f'The system is not valid, with {len(self.errors)} error(s):\n{lines}'
            classes = {issue.error for issue in self.errors}
            raise (classes.pop() if len(classes) == 1 else ValidationError)(error_msg)
