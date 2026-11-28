"""Record the bursts of activity of a run that connect in space and time as events."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse
from scipy.sparse import csgraph

from ...base import BaseConfig
from ...types import Array, Indices
from ._cascade_kernel import _CascadeRecordSpec
from ._run_state_spec import RunStateSpec


@dataclass(frozen=True)
class _OpenEvent:
    """An event whose members were active at the last step, with its first step, profile, and active members."""

    start: int
    time: float
    profile: list[float]
    members: list[Indices]


class _ActivityRecorder(BaseConfig[None, RunStateSpec]):
    """A recorder of the events of activity of the members of an update law.

    Every run reads the activity of every member at one step, joins the active members that are nearest neighbours of
    the space, and joins them to the open events of the previous step through the same members and their nearest
    neighbours. An open event that no active member continues closes as a cascade record, whose time is its first step.
    `prepare()` forgets the open events, `values_` holds the events that the last run closed, and `steps_` counts the
    steps read.

    Args:
        label:
            The label of the events, the label of the update law.

        function:
            The compiled activity `function(t, *values)`, which returns a list that holds the activity of every member.

        reads:
            Whether every value the activity reads is a state, and its index of every member in the state or the
            parameter vector, in the order of the arguments of the function.

        size:
            The number of members.

        neighbors:
            The nearest neighbours of every member, one row per member padded with the number of members, or `None`
            for members of no space, which connect only to themselves.
    """

    values_: list[_CascadeRecordSpec]
    steps_: int

    def __init__(
        self,
        label: str,
        function: Callable[..., list[object]],
        reads: tuple[tuple[bool, Indices], ...],
        size: int,
        neighbors: Indices | None,
    ) -> None:
        self.label = label
        self.function = function
        self.reads = reads
        self.size = size
        self.neighbors = neighbors

    def _prepare(self, context: None) -> None:
        """Forget the open events and the steps read before."""
        self.values_ = []
        self._open: list[_OpenEvent] = []
        self._events: Indices | None = None
        self.steps_ = 0

    def _read(self, state: RunStateSpec) -> Array:
        """Return the activity of every member at a state."""
        values = self.function(state.t, *(state.x[rows] if is_state else state.p[rows] for is_state, rows in self.reads))
        return np.broadcast_to(np.asarray(values[0], dtype=float), (self.size,))

    def _get_edges(self, active: Indices, size: int) -> tuple[Indices, Indices]:
        """Return the pairs of graph nodes the active members join, to each other and to open events.

        The nodes are the active members in order, then the open events.
        """
        position = np.full(size + 1, -1, dtype=np.intp)
        position[active] = np.arange(active.size)
        around = active[:, None] if self.neighbors is None else np.column_stack([active, self.neighbors[active]])
        rows = np.repeat(np.arange(active.size), around.shape[1]).reshape(around.shape)
        nearby = position[around]
        joined = nearby >= 0
        sources, targets = [rows[joined]], [nearby[joined]]
        if self._events is not None:
            events = np.append(self._events, -1)[around]
            continued = events >= 0
            sources.append(rows[continued])
            targets.append(active.size + events[continued])
        return np.concatenate(sources), np.concatenate(targets)

    def _get_closed(self, event: _OpenEvent) -> _CascadeRecordSpec:
        """Return the record of an event that no member continues."""
        profile = tuple(int(value) if float(value).is_integer() else float(value) for value in event.profile)
        area = int(np.unique(np.concatenate(event.members)).size)
        return _CascadeRecordSpec(self.label, event.time, sum(profile), len(profile), area, profile)

    @staticmethod
    def _merge(events: list[_OpenEvent], step: int, time: float) -> _OpenEvent:
        """Return one event of the open events a burst of activity joins, starting at the earliest of them."""
        if not events:
            return _OpenEvent(step, time, [], [])
        if len(events) == 1:
            return events[0]
        first = min(events, key=lambda event: event.start)
        profile = [0.0] * (step - first.start)
        for event in events:
            for index, value in enumerate(event.profile):
                profile[event.start - first.start + index] += value
        return _OpenEvent(first.start, first.time, profile, [rows for event in events for rows in event.members])

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Read the activity of one step and set the events it closes.

        Args:
            inputs:
                The state of the step.
        """
        if inputs is None:
            return
        values = self._read(inputs)
        active = np.flatnonzero(values > 0)
        n_active, n_open = active.size, len(self._open)
        self.values_ = []
        self.steps_ += 1
        if not n_active and not n_open:
            return
        sources, targets = self._get_edges(active, self.size) if n_active else (np.zeros(0, np.intp),) * 2
        graph = scipy.sparse.coo_array((np.ones(sources.size), (sources, targets)), shape=(n_active + n_open,) * 2)
        _, components = csgraph.connected_components(graph, directed=False)
        order = np.argsort(components[:n_active], kind='stable')
        bursts, starts = np.unique(components[:n_active][order], return_index=True)
        joined: dict[int, list[_OpenEvent]] = {int(burst): [] for burst in bursts}
        for index, event in enumerate(self._open):
            component = int(components[n_active + index])
            if component in joined:
                joined[component].append(event)
            else:
                self.values_.append(self._get_closed(event))
        self._open = []
        self._events = np.full(self.size, -1, dtype=np.intp)
        groups = np.split(active[order], starts[1:]) if n_active else []
        for burst, members in zip(bursts, groups, strict=True):
            event = self._merge(joined[int(burst)], self.steps_ - 1, inputs.t)
            event.profile.append(float(values[members].sum()))
            event.members.append(members)
            self._events[members] = len(self._open)
            self._open.append(event)
