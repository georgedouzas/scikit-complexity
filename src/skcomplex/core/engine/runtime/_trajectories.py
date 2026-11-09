"""Record the frames of a simulation as a trajectory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ...specs import AgentSnapshotSpec, BaseRecordSpec, DrawLogSpec, NetworkSnapshotSpec
from ...types import AnyArray
from ..problems import BaseProblemSpec


class Trajectory:
    """Recorded states and parameters of a simulation over time.

    Args:
        problem:
            The compiled problem, which gives the names of the vector components.

        times:
            The time of each recorded frame.

        states:
            One state vector per row.

        parameters:
            One parameter vector per row.

        agents:
            One mapping from agent group path to agents per frame, empty for problems without agents.

        reporters:
            One mapping from dotted reporter name to value per frame, empty for problems without reporters.

        observables:
            One mapping from observable path to value per frame, empty for systems without observables.

        events:
            The events that fired during the run, in order, each with its time and the states before and after its
            reset, the cascades of lattices with their sizes, durations, and areas, and the renewals of renewal laws
            with their members.

        structures:
            One mapping from rewritten element path to its structure per frame, empty for other systems.

        draws:
            The named draws and jump firings of a run that records or replays them, or `None`.

        log_probability:
            The total log-probability of the draws and firings of a run that records or replays them, or `None`.

    Examples:
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = CompositeSystem(
        ...     elements=[PopulationElement('a', x=1)],
        ...     interactions=[ExponentialGrowthInteraction('g', target='a', r=1)],
        ... )
        >>> trajectory = simulate(system, until=1, dt=0.5)
        >>> trajectory.times.tolist()
        [0.0, 0.5, 1.0]
        >>> trajectory.values('a.x').round(3).tolist()
        [1.0, 1.649, 2.718]
    """

    def __init__(
        self,
        problem: BaseProblemSpec,
        times: npt.NDArray[np.float64],
        states: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64],
        *,
        agents: Sequence[Mapping[str, AgentSnapshotSpec]] = (),
        reporters: Sequence[Mapping[str, float]] = (),
        observables: Sequence[Mapping[str, object]] = (),
        events: Sequence[BaseRecordSpec] = (),
        structures: Sequence[Mapping[str, object]] = (),
        draws: DrawLogSpec | None = None,
        log_probability: float | None = None,
    ) -> None:
        self.problem = problem
        self.times = times
        self.states = states
        self.parameters = parameters
        self.agents_by_frame = agents
        self.reporters = reporters
        self.observables = observables
        self.events = list(events)
        self.structures = structures
        self.draws = draws
        self.log_probability = log_probability

    def _get_present(self, name: str, values: AnyArray) -> AnyArray:
        """Return the values of a variable with NaN where a group holds no member.

        The NaN fill the rows of a variable group that hold no member at a frame.
        """
        topology = self.problem.topology
        if topology is None or topology.get_presence(name, self.parameters[0]) is None:
            return values
        present = np.array([topology.get_presence(name, row) for row in self.parameters])
        shaped = present.reshape(present.shape + (1,) * (np.ndim(values) - present.ndim))
        return np.where(shaped, np.asarray(values, dtype=float), np.nan)

    def get_values(self, name: str) -> AnyArray:
        """Return the values of a state or a parameter over time by dotted name.

        Args:
            name:
                The dotted name, such as `rabbits.n` or `growth.r`.

        Returns:
            values:
                One value per frame for a scalar, or one row per frame for a vector, NaN in the rows of a variable
                group on a network that hold no member at a frame.

        Raises:
            KeyError:
                The name is neither a state nor a parameter.
        """
        if self.problem.states.has(name):
            entry = self.problem.states.get_entry(name)
            return self._get_present(name, entry.read(np.asarray(self.problem.states.read(self.states, name))))
        if self.problem.parameters.has(name):
            entry = self.problem.parameters.get_entry(name)
            values = entry.read(np.asarray(self.problem.parameters.read(self.parameters, name)))
            return self._get_present(name, values)
        if self.reporters and name in self.reporters[0]:
            found = [frame[name] for frame in self.reporters]
            if all(isinstance(item, numbers.Real) for item in found):
                return np.asarray(found, dtype=float)
            return np.fromiter(found, dtype=object, count=len(found))
        if self.observables and name in self.observables[0]:
            return np.asarray([frame[name] for frame in self.observables])
        reporter_names = list(self.reporters[0]) if self.reporters else []
        observable_names = list(self.observables[0]) if self.observables else []
        names = [*self.problem.states.names, *self.problem.parameters.names, *reporter_names, *observable_names]
        error_msg = f'No state or parameter named {name!r}. Available names are: {", ".join(names)}.'
        raise KeyError(error_msg)

    def get_member(self, index: int, problem: BaseProblemSpec) -> 'Trajectory':
        """Return the trajectory of one member of a replicated system.

        It is a trajectory of the original system.

        Args:
            index:
                The member index, the same in every group of the replicated system.

            problem:
                The compiled original system, whose single elements the replicated groups hold.

        Returns:
            trajectory:
                The times, states, and parameters of the member, laid out as the original problem lays them out.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.experiments._replicate import _replicate
            >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
            >>> from skcomplex.core.engine.runtime import simulate
            >>> system = LogisticMapSystem().prepare().system_
            >>> ensemble = _replicate(system, 3, {'map.x': [0.1, 0.2, 0.3]})
            >>> trajectory = simulate(ensemble, until=2, dt=1)
            >>> float(trajectory.member(1, simulate(system, until=0).problem).values('map.x')[0])
            0.2
        """
        frames = len(self.times)
        states = np.zeros((frames, problem.states.size))
        parameters = np.zeros((frames, problem.parameters.size))
        for layout, source, target, values in (
            (problem.states, self.problem.states, states, self.states),
            (problem.parameters, self.problem.parameters, parameters, self.parameters),
        ):
            for entry in layout.entries:
                origin = source.get_entry(entry.name)
                if origin.size == entry.size:
                    target[:, entry.start : entry.start + entry.size] = np.asarray(values)[
                        :,
                        origin.start : origin.start + origin.size,
                    ]
                    continue
                columns = np.asarray(values)[:, origin.start : origin.start + origin.size].reshape(
                    frames,
                    -1,
                    entry.size,
                )
                target[:, entry.start : entry.start + entry.size] = columns[:, index]
        return Trajectory(problem, self.times, states, parameters)

    @property
    def names(self) -> list[str]:
        """Names that `values` reads.

        They are the names of the states, parameters, reporters, and observables.
        """
        reporters = list(self.reporters[0]) if self.reporters else []
        observables = list(self.observables[0]) if self.observables else []
        return [*self.problem.states.names, *self.problem.parameters.names, *reporters, *observables]

    def get_agent_locations(self, group: str) -> list[AnyArray]:
        """Return the locations of the agents of a group.

        There is one array per frame with one row per agent alive.

        Args:
            group:
                The path of the agent group.

        Returns:
            locations:
                The locations at every frame, empty arrays for frames without locations.

        Examples:
            >>> from skcomplex.core.engine.runtime import simulate
            >>> from skcomplex.systems.biology.collective_behaviour import BoidsSystem
            >>> trajectory = simulate(BoidsSystem().prepare().system_, until=2, random_state=0)
            >>> [locations.shape[1] for locations in trajectory.agent_locations('flock')]
            [2, 2, 2]
        """
        frames = [self.get_agents(group, frame) for frame in range(len(self.times))]
        return [
            np.asarray(snapshot.locations) if snapshot.locations is not None else np.zeros((0, 0))
            for snapshot in frames
        ]

    def get_lattice(self, name: str) -> AnyArray:
        """Return the values of a lattice state over time in lattice shape.

        Args:
            name:
                The dotted name of a state of a lattice group, such as `cells.alive`.

        Returns:
            values:
                One array of the lattice shape per frame, with the components of a vector or a tensor state on a last
                axis.

        Raises:
            KeyError:
                The name is not a state of a lattice group.
        """
        shape = self.problem.get_lattice_shape(name)
        if shape is None:
            error_msg = f'{name!r} is not a state of a lattice group.'
            raise KeyError(error_msg)
        return self.get_values(name).reshape(len(self.times), *shape, *self.problem.states.get_entry(name).member_shape)

    def get_agents(self, group: str, frame: int) -> AgentSnapshotSpec:
        """Return the agents of a group alive at a frame.

        They come with their ids, states, and locations.

        Args:
            group:
                The path of the agent group.

            frame:
                The index of the frame.

        Returns:
            agents:
                The agents of the group at that frame.

        Raises:
            KeyError:
                The trajectory has no agent group of that path.
        """
        groups = self.agents_by_frame[frame] if self.agents_by_frame else {}
        if group not in groups:
            error_msg = f'No agent group {group!r}. Available groups are: {", ".join(groups)}.'
            raise KeyError(error_msg)
        return groups[group]

    def get_network(self, frame: int, nodes: str | None = None, layer: str | None = None) -> NetworkSnapshotSpec:
        """Return the nodes and edges of a changing network at a frame.

        The network is one whose topology changes.

        Args:
            frame:
                The index of the frame.

            nodes:
                The path of a variable group on the nodes of the network, or `None` for the first such network.

            layer:
                The layer, or `None` for the graph of the network.

        Returns:
            snapshot:
                The rows of the nodes present and the ends and rows of the edges present.

        Raises:
            KeyError:
                The system has no network whose topology changes.
        """
        topology = self.problem.topology
        if topology is None:
            error_msg = 'The system has no network whose topology changes. Give its groups `variable=True`.'
            raise KeyError(error_msg)
        return topology.read_network(self.parameters[frame], nodes, layer)

    def get_structure(self, path: str, frame: int) -> object:
        """Return the word or patch of a rewritten element at a frame.

        Args:
            path:
                The path of the rewritten element.

            frame:
                The index of the frame.

        Returns:
            structure:
                The structure recorded at that frame, which later generations do not change.

        Raises:
            KeyError:
                The path is not a rewritten element of the system.

        Examples:
            >>> from skcomplex.core.engine.runtime import simulate
            >>> from skcomplex.systems.mathematics.fractals import CantorSetSystem
            >>> trajectory = simulate(CantorSetSystem(), until=4, random_state=0)
            >>> trajectory.structure('cantor', 2).parts
            4
        """
        structures = self.structures[frame] if self.structures else {}
        if path not in structures:
            names = ', '.join(structures) or 'none'
            error_msg = f'No rewritten element {path!r}. The rewritten elements of the system are: {names}.'
            raise KeyError(error_msg)
        return structures[path]
