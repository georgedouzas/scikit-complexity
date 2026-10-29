"""Define the base of every analysis, which stores its settings, reads its source, and keeps its results."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from functools import cache
from importlib import import_module
from types import ModuleType
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..base import BaseConfig
from ..core.spaces import ContinuousSpace, LatticeSpace, NetworkSpace
from ..core.system import System
from ..engine.compile import compile_system
from ..engine.problems import BaseProblemSpec, ProblemSpec
from ..engine.runtime import Trajectory
from ..errors import AnalysisError, ValidationError
from ..specs import JumpPathSpec, NetworkSnapshotSpec, SamplesSpec, SeriesSpec, SnapshotsSpec
from ..types import Array, DataKind, DomainKind, FormKind
from .types import Run, Runs, Source

VELOCITY_NAMES = ('vx', 'vy', 'vz')
HEADING_NAMES = ('heading', 'theta', 'angle', 'phase')
POSITION_STATE = 'x'
VELOCITY_STATE = 'v'
SERIES_DIMENSIONS = 1
SNAPSHOT_DIMENSIONS = 2
RUNS_DIMENSIONS = 3
STEP_TOLERANCE = 1e-6
KIND_ORDER = (DataKind.AGENTS, DataKind.FIELD, DataKind.NETWORK, DataKind.PATH, DataKind.SERIES)


class BaseAnalysis(BaseConfig[Source, object], ABC):
    """Base of every analysis: it stores its settings, checks them, prepares on a source, and runs.

    A child stores its settings in `__init__` and implements four hooks that the verbs call:

    - `_check()` validates the settings.
    - `_get_data_kind()` returns the kind of data the analysis reads, such as `'series'` or `'field'`.
    - `_prepare(source)` reads the data from the source with the readers of the base and sets the attributes that end
      in `_`.
    - `_run()` computes the results and sets them as attributes that end in `_`.

    The readers `_read_series`, `_read_path`, `_read_fields`, `_read_snapshots`, and `_read_networks` take a trajectory,
    and pass an array or a graph through.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import BaseAnalysis
        >>> class MeanAnalysis(BaseAnalysis):
        ...     def __init__(self, variables=None):
        ...         self.variables = variables
        ...
        ...     def _get_data_kind(self):
        ...         return 'series'
        ...
        ...     def _prepare(self, source):
        ...         self.series_ = self._read_series(source, self.variables)
        ...
        ...     def _run(self, inputs):
        ...         self.mean_ = self.series_.values.mean(axis=0)
        >>> MeanAnalysis().check().prepare(np.array([1.0, 2.0, 3.0])).run().mean_.tolist()
        [2.0]
    """

    def _check(self) -> None:
        """Validate the settings of the analysis, raising a named error for one it cannot take."""

    def _check_choice(self, name: str, value: object, choices: tuple[object, ...]) -> None:
        """Raise an error for a setting that is not one of its choices."""
        if value not in choices:
            error_msg = f'{type(self).__name__} has {name}={value!r}. Use one of {", ".join(map(str, choices))}.'
            raise AnalysisError(error_msg)

    def _check_settings(self, name: str, read: Set[str]) -> None:
        """Raise an error for a setting changed from its default that the choice of the setting `name` does not read."""
        defaults = inspect.signature(type(self).__init__).parameters
        ignored = [
            key
            for key, value in self.get_params(deep=False).items()
            if key != name and key not in read and self._is_changed(value, defaults[key].default)
        ]
        if ignored:
            error_msg = (
                f'{type(self).__name__} of {name} {getattr(self, name)!r} does not read {", ".join(ignored)}. It reads '
                f'{", ".join(sorted(read)) or "no other setting"}.'
            )
            raise AnalysisError(error_msg)

    @staticmethod
    def _is_changed(value: object, default: object) -> bool:
        """Return whether a setting differs from its default, comparing arrays and sequences by their items."""
        if value is None or default is None:
            return value is not default
        try:
            return not np.array_equal(np.asarray(value, dtype=object), np.asarray(default, dtype=object))
        except (TypeError, ValueError):
            return value is not default

    @abstractmethod
    def _get_data_kind(self) -> DataKind:
        """Return the kind of data the analysis reads."""

    @abstractmethod
    def _prepare(self, source: Source) -> None:
        """Read the data of the analysis from a source and set the attributes that end in `_`."""

    @abstractmethod
    def _run(self, inputs: object = None) -> None:
        """Compute the results of the analysis and set them as attributes that end in `_`."""

    @staticmethod
    def _get_problem(source: Trajectory | System | BaseProblemSpec) -> BaseProblemSpec:
        """Return the compiled problem of a trajectory or a system."""
        if isinstance(source, Trajectory):
            return source.problem
        if isinstance(source, System):
            return compile_system(source)
        return source

    @staticmethod
    def _compiled_problem(system: System | BaseProblemSpec, purpose: str) -> ProblemSpec:
        """Return the general compiler's problem of a system, or the problem itself, for work that reads its fields.

        Args:
            system:
                A system, or a problem compiled from one.

            purpose:
                What needs the general compiler's problem, such as `'Linear stability'`, for the error message.

        Returns:
            problem:
                The problem of the general compiler.

        Raises:
            ValidationError:
                A compiler of whole systems compiles the system to a problem of its own, which the work cannot read.
        """
        problem = system if isinstance(system, BaseProblemSpec) else compile_system(system)
        if isinstance(problem, ProblemSpec):
            return problem
        error_msg = (
            f'{purpose} reads the problem of the general compiler, but the system compiles to a '
            f'{type(problem).__name__} of the {problem.form} form.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _get_lattice_names(problem: BaseProblemSpec) -> list[str]:
        """Return the states of a problem that live on lattices or grids."""
        return [name for name in problem.states.names if problem.lattice_shape(name) is not None]

    @staticmethod
    def _get_path_names(problem: BaseProblemSpec) -> list[str]:
        """Return the discrete states that are not lattice states, or every state of a jump process."""
        lattice = set(BaseAnalysis._get_lattice_names(problem))
        names = [name for name in problem.states.names if name not in lattice]
        if problem.form == FormKind.JUMP:
            return names
        if problem.form == FormKind.DISCRETE:
            discrete = {DomainKind.CATEGORICAL, DomainKind.BOOLEAN, DomainKind.INTEGER}
            return [name for name in names if problem.states.entry(name).kind in discrete]
        return []

    @staticmethod
    def _get_particle_groups(problem: BaseProblemSpec) -> list[str]:
        """Return the groups whose members carry a vector position `x`."""
        found = []
        for element in problem.elements:
            dotted = element.states.get(POSITION_STATE)
            if element.members is not None and dotted is not None and problem.states.entry(dotted).shape:
                found.append(element.path)
        return found

    @staticmethod
    def _get_network_space(problem: BaseProblemSpec) -> NetworkSpace | None:
        """Return the network space of the agents or of a field of a problem, or `None`."""
        if isinstance(problem.agent_space, NetworkSpace):
            return problem.agent_space
        for element in problem.elements:
            space = getattr(element.element, 'space', None)
            if isinstance(space, NetworkSpace):
                return space
        return None

    @staticmethod
    def _get_graph_observables(trajectory: Trajectory) -> list[str]:
        """Return the observables whose value is a graph."""
        first = trajectory.observables[0] if trajectory.observables else {}
        return [name for name, value in first.items() if isinstance(value, nx.Graph)]

    @staticmethod
    def _get_scalar_series(trajectory: Trajectory) -> list[str]:
        """Return the scalar observables and reporters, or the scalar states of single elements without them."""
        first = trajectory.observables[0] if trajectory.observables else {}
        observed = [
            name for name, value in first.items() if not isinstance(value, nx.Graph) and np.ndim(np.asarray(value)) == 0
        ]
        reported = list(trajectory.reporters[0]) if trajectory.reporters else []
        if observed or reported:
            return observed + reported
        lattice = set(BaseAnalysis._get_lattice_names(trajectory.problem))
        return [
            entry.name
            for entry in trajectory.problem.states.entries
            if entry.shape == () and entry.members is None and entry.name not in lattice
        ]

    @staticmethod
    def _has_series(source: Trajectory | System | BaseProblemSpec, problem: BaseProblemSpec) -> bool:
        """Check whether a source records scalar series: observables, reporters, or scalar states."""
        if isinstance(source, Trajectory):
            return bool(BaseAnalysis._get_scalar_series(source))
        if problem.observables:
            return True
        lattice = set(BaseAnalysis._get_lattice_names(problem))
        return any(
            entry.shape == () and entry.members is None and entry.name not in lattice
            for entry in problem.states.entries
        )

    @staticmethod
    def _get_data_kinds(source: Source | BaseProblemSpec) -> list[DataKind]:
        """Return the kinds of data a trajectory, a system, a graph, or an array holds.

        An array of one dimension is a series, and an array of more dimensions is a field and a set of series.

        Args:
            source:
                The trajectory, the system, the graph, or the array.

        Returns:
            kinds:
                The data kinds present, in the order agents, field, network, path, series.
        """
        if isinstance(source, nx.Graph | NetworkSnapshotSpec):
            return [DataKind.NETWORK]
        if not isinstance(source, Trajectory | System | BaseProblemSpec):
            dimensions = np.ndim(source)
            return [DataKind.SERIES] if dimensions <= SERIES_DIMENSIONS else [DataKind.FIELD, DataKind.SERIES]
        problem = BaseAnalysis._get_problem(source)
        agent_groups = [agents.path for agents in problem.agents] + BaseAnalysis._get_particle_groups(problem)
        recorded = isinstance(source, Trajectory) and bool(BaseAnalysis._get_graph_observables(source))
        present = {
            DataKind.AGENTS: bool(agent_groups),
            DataKind.FIELD: bool(BaseAnalysis._get_lattice_names(problem)),
            DataKind.NETWORK: BaseAnalysis._get_network_space(problem) is not None or recorded,
            DataKind.PATH: bool(BaseAnalysis._get_path_names(problem)),
            DataKind.SERIES: BaseAnalysis._has_series(source, problem),
        }
        return [kind for kind in KIND_ORDER if present[kind]]

    def _get_missing_error(self, trajectory: Trajectory, wanted: str) -> AnalysisError:
        """Return the error that a trajectory lacks a kind of data, listing the kinds it has."""
        kinds = ', '.join(kind.value for kind in self._get_data_kinds(trajectory)) or 'none'
        error_msg = f'The trajectory has no {wanted}. Its data kinds are: {kinds}.'
        return AnalysisError(error_msg)

    @staticmethod
    def _unwrap(positions: Array, box: Array | None, *, periodic: bool) -> Array:
        """Return positions unwrapped across periodic sides by accumulating minimum-image steps."""
        if box is None or not periodic:
            return positions
        steps = np.diff(positions, axis=0)
        steps -= box * np.round(steps / box)
        return np.concatenate([positions[:1], positions[:1] + np.cumsum(steps, axis=0)])

    @staticmethod
    def _get_space_box(space: object) -> tuple[Array | None, bool]:
        """Return the sides of a continuous or grid space and whether it is periodic."""
        if isinstance(space, ContinuousSpace):
            return np.asarray(space.extent_[:2], dtype=float), space.periodic_
        if isinstance(space, LatticeSpace):
            return np.asarray(space.shape_[:2], dtype=float), space.periodic_
        return None, False

    @staticmethod
    def _get_columns(ids: npt.NDArray[np.int64], wanted: npt.NDArray[np.int64]) -> npt.NDArray[np.intp]:
        """Return the index of every wanted id among the ids of a frame."""
        if np.all(np.diff(ids) > 0):
            return np.searchsorted(ids, wanted)
        lookup = {int(item): index for index, item in enumerate(ids.tolist())}
        return np.asarray([lookup[int(item)] for item in wanted], dtype=np.intp)

    @staticmethod
    def _read_agent_snapshots(trajectory: Trajectory, group: str) -> SnapshotsSpec:
        """Return the snapshots of an agent group, aligned on the agents present in every frame."""
        frames = [trajectory.agents(group, frame) for frame in range(len(trajectory.times))]
        common = set(frames[0].ids.tolist()).intersection(*(set(frame.ids.tolist()) for frame in frames[1:]))
        ids = np.asarray(sorted(common), dtype=np.int64)
        columns = [BaseAnalysis._get_columns(frame.ids, ids) for frame in frames]
        states = {
            name: np.asarray([frame.values[name][column] for frame, column in zip(frames, columns, strict=True)])
            for name in frames[0].values
        }
        box, periodic = BaseAnalysis._get_space_box(trajectory.problem.agent_space)
        positions = None
        located = not isinstance(trajectory.problem.agent_space, NetworkSpace)
        if located and frames[0].locations is not None and np.asarray(frames[0].locations).dtype != object:
            raw = np.asarray(
                [
                    np.asarray(frame.locations, dtype=float)[column]
                    for frame, column in zip(frames, columns, strict=True)
                ],
            )
            positions = BaseAnalysis._unwrap(raw.reshape(len(frames), ids.size, -1), box, periodic=periodic)
        velocity_names = [name for name in VELOCITY_NAMES if name in states]
        velocities = (
            np.stack([states.pop(name) for name in velocity_names], axis=-1).astype(float) if velocity_names else None
        )
        heading = next((name for name in HEADING_NAMES if name in states), None)
        headings = np.asarray(states.pop(heading), dtype=float) if heading is not None else None
        times = np.asarray(trajectory.times, dtype=float)
        return SnapshotsSpec(times, ids, positions, velocities, headings, states, box, periodic)

    @staticmethod
    def _read_particle_snapshots(trajectory: Trajectory, group: str) -> SnapshotsSpec:
        """Return the snapshots of a particle group from its vector states and headings, unwrapped in a periodic box."""
        element = next(item for item in trajectory.problem.elements if item.path == group)
        system = trajectory.problem.system
        space = system.space if system is not None else None
        periodic = isinstance(space, ContinuousSpace) and space.periodic_
        box = np.asarray(space.extent_, dtype=np.float64) if isinstance(space, ContinuousSpace) and periodic else None
        raw = np.asarray(trajectory.values(element.states[POSITION_STATE]), dtype=float)
        positions = BaseAnalysis._unwrap(raw, box, periodic=periodic)
        velocity = element.states.get(VELOCITY_STATE)
        velocities = np.asarray(trajectory.values(velocity), dtype=float) if velocity is not None else None
        others = {
            name: np.asarray(trajectory.values(dotted)).reshape(len(trajectory.times), -1)
            for name, dotted in element.states.items()
            if name not in {POSITION_STATE, VELOCITY_STATE}
        }
        heading = next((name for name in HEADING_NAMES if name in others), None)
        headings = np.asarray(others.pop(heading), dtype=float) if heading is not None else None
        ids = np.arange(positions.shape[1], dtype=np.int64)
        times = np.asarray(trajectory.times, dtype=float)
        return SnapshotsSpec(times, ids, positions, velocities, headings, others, box, periodic)

    def _read_series(self, source: Source, names: Sequence[str] | None = None) -> SeriesSpec:
        """Return named series of a trajectory, or the columns of an array, one series per column.

        A trajectory gives its states, parameters, reporters, or observables, and a vector or group variable gives one
        column per component, labelled `name[i]`.

        Args:
            source:
                A trajectory, a series, or an array with one row per time and one column per series.

            names:
                The variables of a trajectory, or `None` for every scalar observable and reporter, or every scalar
                state without them.

        Returns:
            series:
                The times, the columns, and their labels.

        Raises:
            AnalysisError:
                The trajectory has no scalar series and no names are given.
        """
        if not isinstance(source, Trajectory):
            array = np.asarray(source)
            values = array.reshape(-1, 1) if array.ndim <= 1 else array.reshape(array.shape[0], -1)
            labels = list(names) if names is not None else [str(index) for index in range(values.shape[1])]
            return SeriesSpec(np.arange(values.shape[0], dtype=float), values, labels)
        chosen = list(names) if names is not None else self._get_scalar_series(source)
        if not chosen:
            raise self._get_missing_error(source, 'scalar series')
        columns, labels = [], []
        for name in chosen:
            values = np.asarray(source.values(name), dtype=float).reshape(len(source.times), -1)
            columns.append(values)
            labels.extend([name] if values.shape[1] == 1 else [f'{name}[{index}]' for index in range(values.shape[1])])
        return SeriesSpec(np.asarray(source.times, dtype=float), np.concatenate(columns, axis=1), labels)

    def _read_path(self, source: Source, name: str | None = None) -> JumpPathSpec:
        """Return the jump path of a discrete state of a trajectory, or of a series of states.

        The frames where the state did not change are dropped.

        Args:
            source:
                The trajectory of a jump process or of a map with discrete states, or a series of states.

            name:
                The dotted name of the state, or `None` for the first discrete state.

        Returns:
            path:
                The jump times and the state after every jump.

        Raises:
            AnalysisError:
                The trajectory has no discrete state, or the array is not one series.
        """
        if not isinstance(source, Trajectory):
            values = np.asarray(source)
            if values.ndim != 1:
                error_msg = f'A jump path is one series of states, but the array has shape {values.shape}.'
                raise AnalysisError(error_msg)
            changed = np.concatenate([[True], values[1:] != values[:-1]])
            return JumpPathSpec(np.flatnonzero(changed).astype(float), values[changed], name or 'state')
        names = self._get_path_names(source.problem)
        chosen = names[0] if name is None and names else name
        if chosen is None:
            raise self._get_missing_error(source, 'discrete state')
        values = np.asarray(source.values(chosen), dtype=float).reshape(len(source.times), -1)
        changed = np.concatenate([[True], np.any(values[1:] != values[:-1], axis=1)])
        states = values[changed]
        times = np.asarray(source.times, dtype=float)[changed]
        return JumpPathSpec(times, states[:, 0] if values.shape[1] == 1 else states, chosen)

    def _read_fields(self, source: Source, name: str | None = None) -> Array:
        """Return the frames of a lattice or grid state of a trajectory, or an array of frames as it is.

        Args:
            source:
                The trajectory, or a grid or frames of grids.

            name:
                The dotted name of the state, or `None` for the first lattice state.

        Returns:
            frames:
                One array of the grid shape per frame, or the array.

        Raises:
            AnalysisError:
                The trajectory has no lattice or grid state.
        """
        if not isinstance(source, Trajectory):
            return np.asarray(source, dtype=float)
        names = self._get_lattice_names(source.problem)
        chosen = names[0] if name is None and names else name
        if chosen is None:
            raise self._get_missing_error(source, 'lattice or grid state')
        return np.asarray(source.lattice(chosen), dtype=float)

    def _read_snapshots(self, source: Source, group: str | None = None) -> SnapshotsSpec:
        """Return the agents of a group of a trajectory at every frame.

        Agent groups give their positions unwrapped across periodic sides, velocities from the states `vx`, `vy`, and
        `vz`, and headings from a state named `heading`, `theta`, `angle`, or `phase`. Particle groups give their vector
        states `x` and `v`.

        Args:
            source:
                The trajectory.

            group:
                The path of the agent or particle group, or `None` for the first one.

        Returns:
            snapshots:
                The aligned snapshots.

        Raises:
            AnalysisError:
                The source is not a trajectory, or it has no agent or particle group of that path.
        """
        if not isinstance(source, Trajectory):
            error_msg = f'Agent snapshots are read from a trajectory, not from {type(source).__name__}.'
            raise AnalysisError(error_msg)
        agent_paths = [agents.path for agents in source.problem.agents]
        particle_paths = self._get_particle_groups(source.problem)
        chosen = group if group is not None else next(iter(agent_paths + particle_paths), None)
        if chosen in particle_paths:
            return self._read_particle_snapshots(source, chosen)
        if chosen is None or chosen not in agent_paths:
            raise self._get_missing_error(source, f'agent group {chosen!r}' if chosen else 'agent group')
        return self._read_agent_snapshots(source, chosen)

    def _read_samples_of(self, source: Source, names: Sequence[str] | None, group: str | None) -> SeriesSpec:
        """Return samples as columns: the series of a source, or the speeds of agents with one column per frame.

        A trajectory gives the speeds of the agents of a group when `group` is set, or when no names are given and it
        has agents but no scalar series. Any other source gives its series.

        Args:
            source:
                A trajectory, a series, or an array with one column per sample.

            names:
                The variables of a trajectory, or `None` for every scalar series.

            group:
                The path of the agent or particle group, or `None` for the first one when the trajectory has no series.

        Returns:
            series:
                The samples, one per column, and their labels.

        Raises:
            AnalysisError:
                The agents of the group have no velocities.
        """
        if not isinstance(source, Trajectory):
            return self._read_series(source, names)
        reads_agents = group is not None or (
            names is None and not self._get_scalar_series(source) and DataKind.AGENTS in self._get_data_kinds(source)
        )
        if not reads_agents:
            return self._read_series(source, names)
        speeds = np.linalg.norm(self._read_velocities(source, group), axis=-1)
        labels = [f'speed at {time:g}' for time in source.times]
        return SeriesSpec(np.arange(speeds.shape[1], dtype=float), speeds.T, labels)

    def _read_velocities(self, source: Source, group: str | None = None) -> Array:
        """Return the velocities of the agents of a group, or an array of one snapshot or of frames.

        Args:
            source:
                The trajectory, or the velocities of one snapshot, one row per agent, or of one snapshot per frame along
                the first axis.

            group:
                The path of the agent or particle group of a trajectory, or `None` for the first one.

        Returns:
            velocities:
                The velocities, frames by agents by dimensions.

        Raises:
            AnalysisError:
                The agents of the group have no velocities.
        """
        if not isinstance(source, Trajectory):
            array = np.asarray(source, dtype=float)
            return array[None] if array.ndim == SNAPSHOT_DIMENSIONS else array
        velocities = self._read_snapshots(source, group).velocities
        if velocities is None:
            error_msg = f'The agents of {group or "the first group"} have no velocities. Use moving agents.'
            raise AnalysisError(error_msg)
        return velocities

    def _read_networks(self, source: Source) -> list[nx.Graph]:
        """Return the network of every frame of a trajectory, or a graph or a network snapshot as the only frame.

        A trajectory gives the graphs it records as observables, or the network space of its problem at every frame.

        Args:
            source:
                The trajectory, the graph, or the network snapshot.

        Returns:
            graphs:
                One NetworkX graph per frame.

        Raises:
            AnalysisError:
                The source is none of these, or the trajectory has no network.
        """
        if isinstance(source, nx.Graph):
            return [source]
        if isinstance(source, NetworkSnapshotSpec):
            return [source.graph()]
        if not isinstance(source, Trajectory):
            error_msg = f'Networks are read from a trajectory or a graph, not from {type(source).__name__}.'
            raise AnalysisError(error_msg)
        recorded = self._get_graph_observables(source)
        if recorded:
            return [frame[recorded[0]] for frame in source.observables]
        space = self._get_network_space(source.problem)
        if space is None:
            raise self._get_missing_error(source, 'network')
        return [space.graph] * len(source.times)

    def prepare(self, source: Source = None) -> Self:
        """Read the data of the checked analysis from a source.

        Args:
            source:
                The trajectory, array, graph, or system the analysis measures.

        Returns:
            analysis:
                The analysis itself.

        Raises:
            NotCheckedError:
                The analysis was not checked with `check`.
        """
        super().prepare(source)
        self._prepared = True
        return self

    def run(self, inputs: object = None, final: bool = False) -> Self:
        """Compute the results of the prepared analysis from the data it read.

        Args:
            inputs:
                Nothing: an analysis computes from the source it was prepared on.

            final:
                Whether this is the last call.

        Returns:
            analysis:
                The analysis itself, with its results in the attributes that end in `_`.

        Raises:
            AnalysisError:
                The analysis was not prepared with a source.
        """
        if not getattr(self, '_prepared', False):
            error_msg = f'{type(self).__name__} must be prepared with a source before it runs. Call `prepare(source)`.'
            raise AnalysisError(error_msg)
        super().run(inputs, final)
        return self

    @staticmethod
    @cache
    def _import(name: str) -> ModuleType:
        """Return a module, importing it on the first call.

        Args:
            name:
                The dotted name of the module.

        Returns:
            module:
                The imported module.

        Examples:
            >>> from skcomplex.analysis import BaseAnalysis
            >>> BaseAnalysis._import('math').sqrt(4.0)
            2.0
        """
        return import_module(name)

    @staticmethod
    def _read_system(source: object) -> System | BaseProblemSpec:
        """Return the system or the problem of a source: the problem of a trajectory, or a system or problem as it is.

        Args:
            source:
                A system, a problem compiled from one, or a trajectory.

        Returns:
            system:
                The system, or the problem.

        Raises:
            AnalysisError:
                The source is not a system, a problem, or a trajectory.
        """
        if isinstance(source, Trajectory):
            return source.problem
        if isinstance(source, System | BaseProblemSpec):
            return source
        error_msg = (
            f'The analysis reads a system, a problem, or a trajectory, not {type(source).__name__}. Pass one of them.'
        )
        raise AnalysisError(error_msg)

    @staticmethod
    def _read_real(values: npt.ArrayLike, argument: str) -> npt.NDArray[np.float64]:
        """Return values as a float array after rejecting complex values, whose imaginary part a cast would drop.

        Args:
            values:
                The values an analysis function receives.

            argument:
                The name of the argument that holds the values, used in the error message.

        Returns:
            array:
                The values as floats.

        Raises:
            ValidationError:
                The values are complex.
        """
        array = np.asarray(values)
        if np.iscomplexobj(array):
            error_msg = (
                f'The {argument} are complex, and the analysis takes real values. Pass one real view of them: the real '
                'part `values.real`, the imaginary part `values.imag`, the modulus `np.abs(values)`, or the argument '
                '`np.angle(values)`.'
            )
            raise ValidationError(error_msg)
        return np.asarray(array, dtype=float)

    @staticmethod
    def _runs_of(source: Runs) -> list[Run]:
        """Return the runs of a source: a list of trajectories or arrays, the first axis of a 3-D array, or itself.

        Args:
            source:
                A trajectory, an array with one row per sample, or several runs as a list or a 3-D array.

        Returns:
            runs:
                The runs.

        Raises:
            AnalysisError:
                The source is a system, a graph, or a network snapshot, which have no samples.
        """
        if isinstance(source, Trajectory):
            return [source]
        if isinstance(source, System | NetworkSnapshotSpec | nx.Graph):
            error_msg = f'Runs are read from trajectories or arrays, not from {type(source).__name__}.'
            raise AnalysisError(error_msg)
        if isinstance(source, list):
            if all(isinstance(item, Trajectory | np.ndarray) for item in source):
                return list(source)
            if any(isinstance(item, Trajectory) for item in source):
                error_msg = 'A list of runs holds trajectories or arrays, not nested lists beside trajectories.'
                raise AnalysisError(error_msg)
            array = np.asarray([np.asarray(item) for item in source if not isinstance(item, Trajectory)])
        else:
            array = np.asarray(source)
        return list(np.asarray(array, dtype=np.float64)) if array.ndim == RUNS_DIMENSIONS else [array]

    @staticmethod
    def _step_of(series: SeriesSpec) -> float:
        """Return the time between the samples of a series, after checking that it is uniform."""
        steps = np.diff(np.asarray(series.times, dtype=float))
        if steps.size == 0 or not np.allclose(steps, steps[0], rtol=STEP_TOLERANCE, atol=0.0):
            error_msg = (
                f'The series of {series.labels} are not sampled at a uniform step with at least two samples. Record '
                'them at a fixed interval, such as with the `dt` of `Simulation`.'
            )
            raise AnalysisError(error_msg)
        return float(steps[0])

    def _read_samples(
        self,
        source: Runs,
        variables: Sequence[str] | None,
        dt: float,
    ) -> SamplesSpec:
        """Return the uniformly sampled series of the runs of a source, with common variables and step.

        Args:
            source:
                A trajectory, an array with one row per sample, or several runs as a list or a 3-D array.

            variables:
                The variables of trajectories, or `None` for every scalar state or observable.

            dt:
                The time between the rows of an array.

        Returns:
            samples:
                The values of every series, the names of the variables, and the step.

        Raises:
            AnalysisError:
                No run is given, or the runs differ in variables or step, or are not uniformly sampled.
        """
        found = []
        for run in self._runs_of(source):
            series = self._read_series(run, variables)
            found.append(
                series if isinstance(run, Trajectory) else SeriesSpec(series.times * dt, series.values, series.labels),
            )
        if not found:
            error_msg = 'No series is given. Pass a trajectory, an array, or a list of them.'
            raise AnalysisError(error_msg)
        steps = [self._step_of(series) for series in found]
        labels = found[0].labels
        if any(series.labels != labels for series in found) or not np.allclose(steps, steps[0], rtol=STEP_TOLERANCE):
            error_msg = (
                f'The series have the variables {[series.labels for series in found]} and the steps {steps}. Give '
                'series of the same variables sampled at the same step.'
            )
            raise AnalysisError(error_msg)
        return SamplesSpec([np.asarray(series.values, dtype=float) for series in found], list(labels), steps[0])
