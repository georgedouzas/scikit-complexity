"""Define the frame spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...base import BaseSpec
from ...types import AnyArray, Array
from ._agent_snapshot_spec import AgentSnapshotSpec
from ._base_problem_spec import BaseProblemSpec
from ._base_record_spec import BaseRecordSpec


@dataclass(frozen=True)
class FrameSpec(BaseSpec):
    """A spec of the state of a run at one time, which observers receive.

    It holds the step, the time, the states and parameters, and the agents, reporters, observables, events, and
    structures of one frame, such as the size 10 of a population at time 0. A run hands one to every observer at every
    frame. It records the course of an epidemic, the prices of a market, and the moves of a flock.

    Read more in the user guide, under Observe.

    Args:
        step:
            The index of the frame, starting at 0.

        time:
            The simulated time.

        state:
            A copy of the state vector.

        parameters:
            A copy of the parameter vector.

        problem:
            The compiled problem, which gives the names of the vector components.

        agents:
            A mapping from agent group path to the agents alive at this frame.

        reporters:
            A mapping from dotted reporter name to its value at this frame.

        observables:
            A mapping from observable path to its value at this frame.

        events:
            The events, cascades, and renewals since the previous frame, in order.

        structures:
            A mapping from the path of a rewritten element to its word or patch at this frame.

    See Also:
        BaseObserver: Receives the frames of a run.

    Examples:
        >>> from skcomplex.core.engine.runtime import RecorderObserver
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> recorder = RecorderObserver()
        >>> _ = LogisticGrowthSystem().prepare().simulate(until=1, dt=1, observers=[recorder])
        >>> recorder.frames_[0].time, recorder.frames_[0].state.tolist()
        (0.0, [10.0])
    """

    step: int
    time: float
    state: Array
    parameters: Array
    problem: BaseProblemSpec
    agents: Mapping[str, AgentSnapshotSpec] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)
    observables: Mapping[str, npt.ArrayLike] = field(default_factory=dict)
    events: tuple[BaseRecordSpec, ...] = ()
    structures: Mapping[str, object] = field(default_factory=dict)

    def get_value(self, name: str) -> npt.ArrayLike:
        """Return the value of a state or a parameter by dotted name.

        Args:
            name:
                The dotted name, such as `rabbits.n` or `growth.r`.

        Returns:
            value:
                A number for a scalar and an array for a vector.
        """
        if self.problem.states.has(name):
            return self.problem.states.read(self.state, name)
        if name in self.reporters:
            return self.reporters[name]
        if name in self.observables:
            return self.observables[name]
        return self.problem.parameters.read(self.parameters, name)

    def get_lattice(self, name: str) -> AnyArray:
        """Return the values of a lattice state in lattice shape.

        Args:
            name:
                The dotted name of a state of a lattice group, such as `cells.alive`.

        Returns:
            values:
                The array of the lattice shape, with the components of a vector or a tensor state on a last axis.

        Raises:
            KeyError:
                The name is not a state of a lattice group.
        """
        shape = self.problem.get_lattice_shape(name)
        if shape is None:
            error_msg = f'{name!r} is not a state of a lattice group.'
            raise KeyError(error_msg)
        member = self.problem.states.get_entry(name).member_shape
        return np.reshape(self.get_value(name), (*shape, *member))
