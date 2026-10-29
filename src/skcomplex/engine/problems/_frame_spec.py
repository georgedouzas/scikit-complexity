"""Define the frame spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...specs._agent_snapshot_spec import AgentSnapshotSpec
from ...specs._base_record_spec import BaseRecordSpec
from ...specs._base_spec import BaseSpec
from ...types import AnyArray, Array
from ._base_problem_spec import BaseProblemSpec


@dataclass(frozen=True)
class FrameSpec(BaseSpec):
    """State of a simulation at one point in time.

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
            The events, relaxations, and renewals since the previous frame, in order.

        structures:
            A mapping from the path of a rewritten element to its word or patch at this frame.
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

    def value(self, name: str) -> npt.ArrayLike:
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

    def lattice(self, name: str) -> AnyArray:
        """Return the values of a lattice state in lattice shape.

        Args:
            name:
                The dotted name of a state of a lattice group, such as `cells.alive`.

        Returns:
            values:
                The array of the lattice shape.

        Raises:
            KeyError:
                The name is not a state of a lattice group.
        """
        shape = self.problem.lattice_shape(name)
        if shape is None:
            error_msg = f'{name!r} is not a state of a lattice group.'
            raise KeyError(error_msg)
        return np.reshape(self.value(name), shape)
