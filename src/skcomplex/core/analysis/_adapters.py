"""Extract every kind of data from trajectories and experiments, and list the analyses that apply to an object."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..base import AnalysisError, Categorical, Domain, Form
from ..compile import Problem, compile_system, observables_of
from ..experiments import ExperimentResult
from ..model import ContinuousSpace, LatticeSpace, NetworkSpace, System
from ..runtime import Trajectory
from ._kinds import ANALYSES, AnalysisEntry, DataKind

type Array = npt.NDArray[np.float64]
DISCRETE_DOMAINS = (Domain.INTEGER, Domain.NONNEGATIVE_INTEGER, Domain.BOOLEAN)
VELOCITY_NAMES = ('vx', 'vy', 'vz')
HEADING_NAMES = ('heading', 'theta', 'angle', 'phase')
POSITION_STATE = 'x'
VELOCITY_STATE = 'v'
SERIES_DIMENSIONS = 1
KIND_ORDER = (
    DataKind.AGENTS,
    DataKind.FIELD,
    DataKind.NETWORK,
    DataKind.PATH,
    DataKind.SERIES,
    DataKind.ENSEMBLE,
)


@dataclass(frozen=True)
class Series:
    """Named series on common times.

    Args:
        times:
            The time of every frame.

        values:
            One row per frame and one column per series.

        labels:
            The name of every column.
    """

    times: Array
    values: Array
    labels: list[str]


@dataclass(frozen=True)
class JumpPath:
    """Path of a discrete state with the time of every jump.

    Args:
        times:
            The first time, then the time of every frame at which the state changed.

        states:
            The state at the first time and after every jump, one row per jump for vector states.

        label:
            The name of the state.
    """

    times: Array
    states: Array
    label: str


@dataclass(frozen=True)
class Snapshots:
    """Agents of one group at every frame, aligned on the agents present in every frame.

    Args:
        times:
            The time of every frame.

        ids:
            The id of every agent, in column order.

        positions:
            The unwrapped positions, frames by agents by dimensions, or `None` without a space with coordinates.

        velocities:
            The velocities, shaped like the positions, or `None` without them.

        headings:
            The heading angles, frames by agents, or `None` without them.

        states:
            Every other state of the agents, frames by agents.

        box:
            The sides of the space, or `None` for an unbounded space.

        periodic:
            Whether opposite sides of the box are joined.
    """

    times: Array
    ids: npt.NDArray[np.int64]
    positions: Array | None
    velocities: Array | None
    headings: Array | None
    states: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    box: Array | None = None
    periodic: bool = False


@dataclass(frozen=True)
class Ensemble:
    """One series of every run of an experiment on common times.

    Args:
        times:
            The common times.

        values:
            One row per run and one column per common time.

        rows:
            The design row of every run.

        seeds:
            The index of the random stream of every run.
    """

    times: Array
    values: Array
    rows: list[dict[str, Any]]
    seeds: npt.NDArray[np.int64]


def _problem_of(source: Trajectory | System | Problem) -> Problem:
    """Return the compiled problem of a trajectory or a system."""
    if isinstance(source, Trajectory):
        return source.problem
    if isinstance(source, System):
        return compile_system(source)
    return source


def _lattice_names(problem: Problem) -> list[str]:
    """Return the states of a problem that live on lattices or grids."""
    return [name for name in problem.states.names if problem.lattice_shape(name) is not None]


def _is_discrete(problem: Problem, name: str) -> bool:
    """Check whether a state takes discrete values."""
    domain = problem.states.entry(name).domain
    return isinstance(domain, Categorical) or domain in DISCRETE_DOMAINS


def _path_names(problem: Problem) -> list[str]:
    """Return the discrete states that are not lattice states, every state of a jump process."""
    lattice = set(_lattice_names(problem))
    names = [name for name in problem.states.names if name not in lattice]
    if problem.form is Form.JUMP:
        return names
    if problem.form is Form.DISCRETE:
        return [name for name in names if _is_discrete(problem, name)]
    return []


def _particle_groups(problem: Problem) -> list[str]:
    """Return the groups whose members carry a vector position `x`."""
    found = []
    for element in problem.elements:
        dotted = element.states.get(POSITION_STATE)
        if element.members is not None and dotted is not None and problem.states.entry(dotted).shape:
            found.append(element.path)
    return found


def _agent_groups(problem: Problem) -> list[str]:
    """Return the agent groups and particle groups of a problem."""
    return [agents.path for agents in problem.agents] + _particle_groups(problem)


def _network_space(problem: Problem) -> NetworkSpace | None:
    """Return the network space of the agents or of a field of a problem, or `None`."""
    if isinstance(problem.agent_space, NetworkSpace):
        return problem.agent_space
    for element in problem.elements:
        space = getattr(element.element, 'space', None)
        if isinstance(space, NetworkSpace):
            return space
    return None


def _graph_observables(trajectory: Trajectory) -> list[str]:
    """Return the observables whose value is a graph."""
    first = trajectory.observables[0] if trajectory.observables else {}
    return [name for name, value in first.items() if isinstance(value, nx.Graph)]


def _scalar_series(trajectory: Trajectory) -> list[str]:
    """Return the scalar observables and reporters, or the scalar states of single elements without them."""
    first = trajectory.observables[0] if trajectory.observables else {}
    observed = [name for name, value in first.items() if np.ndim(value) == 0 and not isinstance(value, nx.Graph)]
    reported = list(trajectory.reporters[0]) if trajectory.reporters else []
    if observed or reported:
        return observed + reported
    lattice = set(_lattice_names(trajectory.problem))
    return [
        entry.name
        for entry in trajectory.problem.states
        if entry.shape == () and entry.members is None and entry.name not in lattice
    ]


def data_kinds(source: Trajectory | System | ExperimentResult | npt.ArrayLike) -> list[DataKind]:
    """Return the kinds of data that a trajectory, a system, an experiment result, or an array holds.

    Args:
        source:
            The trajectory, the system, the experiment result, or the array.

    Returns:
        kinds:
            The data kinds present, in the order agents, field, network, path, series, ensemble.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import data_kinds
        >>> data_kinds(np.zeros((8, 8)))
        [<DataKind.FIELD: 'field'>, <DataKind.SERIES: 'series'>]
    """
    if isinstance(source, ExperimentResult):
        return [DataKind.ENSEMBLE]
    if not isinstance(source, Trajectory | System | Problem):
        dimensions = np.ndim(source)
        return [DataKind.SERIES] if dimensions <= SERIES_DIMENSIONS else [DataKind.FIELD, DataKind.SERIES]
    problem = _problem_of(source)
    present = {
        DataKind.AGENTS: bool(_agent_groups(problem)),
        DataKind.FIELD: bool(_lattice_names(problem)),
        DataKind.NETWORK: _network_space(problem) is not None
        or (isinstance(source, Trajectory) and bool(_graph_observables(source))),
        DataKind.PATH: bool(_path_names(problem)),
        DataKind.SERIES: _has_series(source, problem),
    }
    return [kind for kind in KIND_ORDER if present.get(kind, False)]


def _has_series(source: Trajectory | System | Problem, problem: Problem) -> bool:
    """Check whether a source records scalar series: observables, reporters, or scalar states."""
    if isinstance(source, Trajectory):
        return bool(_scalar_series(source))
    if problem.system is not None and any(True for _ in observables_of(problem.system)):
        return True
    lattice = set(_lattice_names(problem))
    return any(entry.shape == () and entry.members is None and entry.name not in lattice for entry in problem.states)


def _missing(trajectory: Trajectory, wanted: str) -> AnalysisError:
    """Return the error that a trajectory lacks a kind of data, listing the kinds it has."""
    kinds = ', '.join(kind.value for kind in data_kinds(trajectory)) or 'none'
    error_msg = f'The trajectory has no {wanted}. Its data kinds are: {kinds}.'
    return AnalysisError(error_msg)


def series_of(trajectory: Trajectory, names: Sequence[str] | None = None) -> Series:
    """Return named series of a trajectory: states, parameters, reporters, or observables, one column each.

    Vector and group variables give one column per component, labelled `name[i]`.

    Args:
        trajectory:
            The trajectory.

        names:
            The names, or `None` for every scalar observable and reporter, or every scalar state without them.

    Returns:
        series:
            The times, the columns, and their labels.

    Raises:
        AnalysisError:
            The trajectory has no scalar series and no names are given.

    Examples:
        >>> from skcomplex.core.analysis import series_of
        >>> from skcomplex.models import run
        >>> series = series_of(run('lotka_volterra', settings={'until': 1.0}).trajectory_)
        >>> series.labels, series.values.shape[1]
        (['prey.n', 'predators.n'], 2)
    """
    chosen = list(names) if names is not None else _scalar_series(trajectory)
    if not chosen:
        raise _missing(trajectory, 'scalar series')
    columns, labels = [], []
    for name in chosen:
        values = np.asarray(trajectory[name], dtype=float).reshape(len(trajectory), -1)
        columns.append(values)
        labels.extend([name] if values.shape[1] == 1 else [f'{name}[{index}]' for index in range(values.shape[1])])
    return Series(np.asarray(trajectory.times, dtype=float), np.concatenate(columns, axis=1), labels)


def path_of(trajectory: Trajectory, name: str | None = None) -> JumpPath:
    """Return the jump path of a discrete state, dropping the frames where it did not change.

    Args:
        trajectory:
            The trajectory of a jump process, or of a map with discrete states.

        name:
            The dotted name of the state, or `None` for the first discrete state.

    Returns:
        path:
            The jump times and the state after every jump.

    Raises:
        AnalysisError:
            The trajectory has no discrete state.

    Examples:
        >>> from skcomplex.core.analysis import path_of
        >>> from skcomplex.models import run
        >>> path = path_of(run('poisson_process', settings={'until': 20.0, 'random_state': 0}).trajectory_)
        >>> bool((path.states[1:] != path.states[:-1]).any(axis=1).all())
        True
    """
    names = _path_names(trajectory.problem)
    chosen = names[0] if name is None and names else name
    if chosen is None:
        raise _missing(trajectory, 'discrete state')
    values = np.asarray(trajectory[chosen], dtype=float).reshape(len(trajectory), -1)
    changed = np.concatenate([[True], np.any(values[1:] != values[:-1], axis=1)])
    states = values[changed]
    return JumpPath(
        np.asarray(trajectory.times, dtype=float)[changed],
        states[:, 0] if values.shape[1] == 1 else states,
        chosen,
    )


def fields_of(trajectory: Trajectory, name: str | None = None) -> Array:
    """Return the frames of a lattice or grid state in grid shape.

    Args:
        trajectory:
            The trajectory.

        name:
            The dotted name of the state, or `None` for the first lattice state.

    Returns:
        frames:
            One array of the grid shape per frame.

    Raises:
        AnalysisError:
            The trajectory has no lattice or grid state.

    Examples:
        >>> from skcomplex.core.analysis import fields_of
        >>> from skcomplex.models import run
        >>> fields_of(run('ising', size=16, settings={'until': 3}).trajectory_).shape
        (4, 16, 16)
    """
    names = _lattice_names(trajectory.problem)
    chosen = names[0] if name is None and names else name
    if chosen is None:
        raise _missing(trajectory, 'lattice or grid state')
    return np.asarray(trajectory.lattice(chosen), dtype=float)


def _unwrap(positions: Array, box: Array | None, *, periodic: bool) -> Array:
    """Return positions unwrapped across periodic sides by accumulating minimum-image steps."""
    if box is None or not periodic:
        return positions
    steps = np.diff(positions, axis=0)
    steps -= box * np.round(steps / box)
    return np.concatenate([positions[:1], positions[:1] + np.cumsum(steps, axis=0)])


def _space_box(space: object) -> tuple[Array | None, bool]:
    """Return the sides of a continuous or grid space and whether it is periodic."""
    if isinstance(space, ContinuousSpace):
        return np.asarray([space.width, space.height], dtype=float), bool(space.torus)
    if isinstance(space, LatticeSpace):
        return np.asarray([space.width, space.height], dtype=float), bool(space.torus)
    return None, False


def _agent_snapshots(trajectory: Trajectory, group: str) -> Snapshots:
    """Return the snapshots of an agent group, aligned on the agents present in every frame."""
    frames = [trajectory.agents(group, frame) for frame in range(len(trajectory))]
    common = set(frames[0].ids.tolist()).intersection(*(set(frame.ids.tolist()) for frame in frames[1:]))
    ids = np.asarray(sorted(common), dtype=np.int64)
    columns = [
        np.searchsorted(frame.ids, ids) if np.all(np.diff(frame.ids) > 0) else _positions_of(frame.ids, ids)
        for frame in frames
    ]
    names = list(frames[0].values)
    states = {
        name: np.asarray([frame.values[name][column] for frame, column in zip(frames, columns, strict=True)])
        for name in names
    }
    box, periodic = _space_box(trajectory.problem.agent_space)
    positions = None
    located = not isinstance(trajectory.problem.agent_space, NetworkSpace)
    if located and frames[0].locations is not None and np.asarray(frames[0].locations).dtype != object:
        raw = np.asarray(
            [np.asarray(frame.locations, dtype=float)[column] for frame, column in zip(frames, columns, strict=True)],
        )
        positions = _unwrap(raw.reshape(len(frames), ids.size, -1), box, periodic=periodic)
    velocity_names = [name for name in VELOCITY_NAMES if name in states]
    velocities = (
        np.stack([states.pop(name) for name in velocity_names], axis=-1).astype(float) if velocity_names else None
    )
    heading = next((name for name in HEADING_NAMES if name in states), None)
    headings = np.asarray(states.pop(heading), dtype=float) if heading is not None else None
    return Snapshots(
        np.asarray(trajectory.times, dtype=float),
        ids,
        positions,
        velocities,
        headings,
        states,
        box,
        periodic,
    )


def _positions_of(ids: npt.NDArray[np.int64], wanted: npt.NDArray[np.int64]) -> npt.NDArray[np.intp]:
    """Return the index of every wanted id among unsorted ids."""
    lookup = {int(item): index for index, item in enumerate(ids.tolist())}
    return np.asarray([lookup[int(item)] for item in wanted], dtype=np.intp)


def _particle_box(trajectory: Trajectory, group: str) -> Array | None:
    """Return the sides of the periodic box a particle group lives in, or `None` without one."""
    space = getattr(trajectory.problem.system, 'space', None)
    if isinstance(space, ContinuousSpace) and space.boundary == 'periodic':
        return np.asarray(space.extent, dtype=float)
    return None


def _particle_snapshots(trajectory: Trajectory, group: str) -> Snapshots:
    """Return the snapshots of a particle group from its vector states."""
    element = next(item for item in trajectory.problem.elements if item.path == group)
    box = _particle_box(trajectory, group)
    raw = np.asarray(trajectory[element.states[POSITION_STATE]], dtype=float)
    positions = _unwrap(raw, box, periodic=box is not None)
    velocity = element.states.get(VELOCITY_STATE)
    velocities = np.asarray(trajectory[velocity], dtype=float) if velocity is not None else None
    others = {
        name: np.asarray(trajectory[dotted]).reshape(len(trajectory), -1)
        for name, dotted in element.states.items()
        if name not in {POSITION_STATE, VELOCITY_STATE}
    }
    ids = np.arange(positions.shape[1], dtype=np.int64)
    return Snapshots(
        np.asarray(trajectory.times, dtype=float),
        ids,
        positions,
        velocities,
        None,
        others,
        box,
        box is not None,
    )


def snapshots_of(trajectory: Trajectory, group: str | None = None) -> Snapshots:
    """Return the agents of a group at every frame with unwrapped positions, velocities, headings, and states.

    Agent groups give their Mesa positions, velocities from the states `vx`, `vy`, and `vz`, and headings from a
    state named `heading`, `theta`, `angle`, or `phase`. Particle groups give their vector states `x` and `v`.

    Args:
        trajectory:
            The trajectory.

        group:
            The path of the agent or particle group, or `None` for the first one.

    Returns:
        snapshots:
            The aligned snapshots.

    Raises:
        AnalysisError:
            The trajectory has no agent or particle group of that path.

    Examples:
        >>> from skcomplex.core.analysis import snapshots_of
        >>> from skcomplex.models import run
        >>> trajectory = run('run_and_tumble', n_agents=5, settings={'until': 3}).trajectory_
        >>> snapshots_of(trajectory).positions.shape
        (4, 5, 2)
    """
    agent_paths = [agents.path for agents in trajectory.problem.agents]
    particle_paths = _particle_groups(trajectory.problem)
    chosen = group if group is not None else next(iter(agent_paths + particle_paths), None)
    if chosen in particle_paths:
        return _particle_snapshots(trajectory, chosen)
    if chosen is None or chosen not in agent_paths:
        raise _missing(trajectory, f'agent group {chosen!r}' if chosen else 'agent group')
    return _agent_snapshots(trajectory, chosen)


def networks_of(trajectory: Trajectory) -> list[nx.Graph]:
    """Return the network of every frame: graphs recorded as observables, or the network space of the problem.

    Args:
        trajectory:
            The trajectory.

    Returns:
        graphs:
            One NetworkX graph per frame.

    Raises:
        AnalysisError:
            The trajectory has no network.

    Examples:
        >>> from skcomplex.core.analysis import networks_of
        >>> from skcomplex.models import run
        >>> graphs = networks_of(run('network_diffusion', settings={'until': 0.2}).trajectory_)
        >>> len(graphs), graphs[0].number_of_nodes()
        (3, 34)
    """
    recorded = _graph_observables(trajectory)
    if recorded:
        return [frame[recorded[0]] for frame in trajectory.observables]
    space = _network_space(trajectory.problem)
    if space is None:
        raise _missing(trajectory, 'network')
    return [space.graph] * len(trajectory)


def ensemble_of(result: ExperimentResult, name: str) -> Ensemble:
    """Return one kept series of every run of an experiment, interpolated on the times of the shortest run.

    Args:
        result:
            The experiment result.

        name:
            The path of the kept series.

    Returns:
        ensemble:
            The common times, one row per run, the design rows, and the seeds.

    Examples:
        >>> from skcomplex.core.analysis import ensemble_of
        >>> from skcomplex.core.experiments import Experiment
        >>> result = Experiment('logistic_growth', replicates=3, until=5, outputs=['population.n']).run()
        >>> ensemble_of(result, 'population.n').values.shape[0]
        3
    """
    times = min((np.asarray(run.times, dtype=float) for run in result.runs), key=len)
    values = np.asarray(
        [
            np.interp(times, np.asarray(run.times, dtype=float), np.asarray(run.series[name], dtype=float))
            for run in result.runs
        ],
    )
    rows = [dict(run.row) for run in result.runs]
    seeds = np.asarray([run.seed for run in result.runs], dtype=np.int64)
    return Ensemble(times, values, rows, seeds)


@dataclass(frozen=True)
class Applicable:
    """Analysis that applies to an object, with the data it reads.

    Args:
        name:
            The name of the analysis function.

        kind:
            The kind of data it consumes.

        adapter:
            The name of the function that extracts that data from a trajectory or an experiment result.

        reading:
            How the analysis reads the object.
    """

    name: str
    kind: DataKind
    adapter: str
    reading: str


def _array_reading(kind: DataKind, dimensions: int) -> str:
    """Return how an analysis of a kind reads a plain array of some dimensions."""
    if kind is DataKind.FIELD:
        return 'the array as one grid'
    if dimensions > SERIES_DIMENSIONS:
        return 'every column of the array as a series'
    return 'the array as one series'


def analyses_for(source: Trajectory | System | ExperimentResult | npt.ArrayLike) -> list[Applicable]:
    """Return the registered analyses that apply to a trajectory, a system, an experiment result, or an array.

    Args:
        source:
            The object to analyse.

    Returns:
        analyses:
            Every applicable analysis with its data kind, its adapter, and how it reads the object, by data kind.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import analyses_for
        >>> found = {item.name: item.reading for item in analyses_for(np.zeros((8, 8)))}
        >>> found['structure_factor'], found['autocorrelation']
        ('the array as one grid', 'every column of the array as a series')
    """
    kinds = data_kinds(source)
    entries: list[AnalysisEntry] = [ANALYSES.resolve(key) for key in ANALYSES]
    is_array = not isinstance(source, Trajectory | System | ExperimentResult | Problem)
    dimensions = int(np.ndim(cast('npt.ArrayLike', source))) if is_array else 0
    return [
        Applicable(
            entry.name,
            entry.kind,
            entry.kind.adapter,
            _array_reading(entry.kind, dimensions) if is_array else entry.kind.reading,
        )
        for kind in kinds
        for entry in entries
        if entry.kind is kind
    ]
