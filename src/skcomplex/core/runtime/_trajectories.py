"""Record the frames of a simulation as a trajectory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ..compile import (
    EventRecord,
    ExtremalRecord,
    NetworkSnapshot,
    Problem,
    RelaxationRecord,
    RenewalRecord,
    Structure,
)
from ..solvers import AgentSnapshot


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
            reset, the relaxations of lattices with their sizes, durations, and areas, and the renewals of renewal laws
            with their members.

        structures:
            One mapping from rewritten element path to its structure per frame, empty for other systems.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowth, Population
        >>> from skcomplex.core.runtime import Simulation
        >>> system = System(components=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1)])
        >>> trajectory = Simulation(system, dt=0.5).run(until=1).trajectory_
        >>> trajectory.times.tolist()
        [0.0, 0.5, 1.0]
        >>> trajectory['a.n'].round(3).tolist()
        [1.0, 1.649, 2.718]
    """

    def __init__(
        self,
        problem: Problem,
        times: npt.NDArray[np.float64],
        states: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64],
        *,
        agents: Sequence[Mapping[str, AgentSnapshot]] = (),
        reporters: Sequence[Mapping[str, float]] = (),
        observables: Sequence[Mapping[str, Any]] = (),
        events: Sequence[EventRecord | RelaxationRecord | RenewalRecord | ExtremalRecord] = (),
        structures: Sequence[Mapping[str, Structure]] = (),
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

    def __len__(self) -> int:
        """Return the number of recorded frames."""
        return len(self.times)

    def _present(self, name: str, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Return the values of a variable with NaN in the rows of a variable group that hold no member at a frame."""
        topology = self.problem.topology
        if topology is None or topology.presence(name, self.parameters[0]) is None:
            return values
        present = np.array([topology.presence(name, row) for row in self.parameters])
        shaped = present.reshape(present.shape + (1,) * (np.ndim(values) - present.ndim))
        return np.where(shaped, np.asarray(values, dtype=float), np.nan)

    def __getitem__(self, name: str) -> npt.NDArray[Any]:
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
        if name in self.problem.states:
            entry = self.problem.states.entry(name)
            return self._present(name, entry.domain.decode(np.asarray(self.problem.states.read(self.states, name))))
        if name in self.problem.parameters:
            entry = self.problem.parameters.entry(name)
            values = entry.domain.decode(np.asarray(self.problem.parameters.read(self.parameters, name)))
            return self._present(name, values)
        if self.reporters and name in self.reporters[0]:
            return np.asarray([frame[name] for frame in self.reporters], dtype=float)
        if self.observables and name in self.observables[0]:
            return np.asarray([frame[name] for frame in self.observables])
        reporter_names = list(self.reporters[0]) if self.reporters else []
        observable_names = list(self.observables[0]) if self.observables else []
        names = [*self.problem.states.names, *self.problem.parameters.names, *reporter_names, *observable_names]
        error_msg = f'No state or parameter named {name!r}. Available names are: {", ".join(names)}.'
        raise KeyError(error_msg)

    def member(self, index: int, problem: Problem) -> 'Trajectory':
        """Return the trajectory of one member of a replicated system, as a trajectory of the original system.

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
            >>> from skcomplex.core.compile import compile_system
            >>> from skcomplex.core.system import replicate
            >>> from skcomplex.core.runtime import Simulation
            >>> from skcomplex.models.mathematics.dynamical_systems import logistic_map
            >>> ensemble = replicate(logistic_map(), 3, {'map.x': [0.1, 0.2, 0.3]})
            >>> trajectory = Simulation(ensemble, dt=1).run(until=2).trajectory_
            >>> float(trajectory.member(1, compile_system(logistic_map()))['map.x'][0])
            0.2
        """
        frames = len(self)
        states = np.zeros((frames, problem.states.size))
        parameters = np.zeros((frames, problem.parameters.size))
        for layout, source, target, values in (
            (problem.states, self.problem.states, states, self.states),
            (problem.parameters, self.problem.parameters, parameters, self.parameters),
        ):
            for entry in layout:
                origin = source.entry(entry.name)
                columns = np.asarray(values)[:, origin.start : origin.start + origin.size].reshape(
                    frames,
                    -1,
                    entry.size,
                )
                target[:, entry.start : entry.start + entry.size] = columns[:, index]
        return Trajectory(problem, self.times, states, parameters)

    @property
    def names(self) -> list[str]:
        """Names of the states, parameters, reporters, and observables that indexing the trajectory reads."""
        reporters = list(self.reporters[0]) if self.reporters else []
        observables = list(self.observables[0]) if self.observables else []
        return [*self.problem.states.names, *self.problem.parameters.names, *reporters, *observables]

    def agent_locations(self, group: str) -> list[npt.NDArray[Any]]:
        """Return the locations of the agents of a group, one array per frame with one row per agent alive.

        Args:
            group:
                The path of the agent group.

        Returns:
            locations:
                The locations at every frame, empty arrays for frames without locations.

        Examples:
            >>> from skcomplex.core.runtime import Simulation
            >>> from skcomplex.models import create
            >>> trajectory = Simulation(create('boids'), random_state=0).run(until=2).trajectory_
            >>> [locations.shape[1] for locations in trajectory.agent_locations('flock')]
            [2, 2, 2]
        """
        frames = [self.agents(group, frame) for frame in range(len(self))]
        return [
            np.asarray(snapshot.locations) if snapshot.locations is not None else np.zeros((0, 0))
            for snapshot in frames
        ]

    def lattice(self, name: str) -> npt.NDArray[Any]:
        """Return the values of a lattice state over time in lattice shape.

        Args:
            name:
                The dotted name of a state of a lattice group, such as `cells.alive`.

        Returns:
            values:
                One array of the lattice shape per frame.

        Raises:
            KeyError:
                The name is not a state of a lattice group.
        """
        shape = self.problem.lattice_shape(name)
        if shape is None:
            error_msg = f'{name!r} is not a state of a lattice group.'
            raise KeyError(error_msg)
        return self[name].reshape(len(self), *shape)

    def agents(self, group: str, frame: int) -> AgentSnapshot:
        """Return the agents of a group alive at a frame, with their ids, states, and locations.

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

    def network(self, frame: int, nodes: str | None = None, layer: str | None = None) -> NetworkSnapshot:
        """Return the nodes and edges of a network whose topology changes, as they are at a frame.

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

    def structure(self, path: str, frame: int) -> Structure:
        """Return the word, patch, or host graph of a rewritten element at a frame.

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
            >>> from skcomplex.models import run
            >>> trajectory = run('cantor_set', settings={'until': 4}).trajectory_
            >>> len(trajectory.structure('cantor', 2))
            4
        """
        structures = self.structures[frame] if self.structures else {}
        if path not in structures:
            names = ', '.join(structures) or 'none'
            error_msg = f'No rewritten element {path!r}. The rewritten elements of the system are: {names}.'
            raise KeyError(error_msg)
        return structures[path]
