"""Measure the collective order of agents: polar, milling, nematic, and phase order."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from typing import Self

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
KINDS = ('polar', 'milling', 'nematic', 'phase')
SETTINGS = {
    'polar': {'vectors'},
    'milling': set(),
    'nematic': set(),
    'phase': set(),
}
"""Settings every kind reads, besides `kind` and `group`."""


def _unit_vectors(headings: npt.ArrayLike, *, vectors: bool) -> Array:
    """Return unit vectors from angles, or normalized velocity vectors, with agents on the second to last axis."""
    array = np.asarray(headings, dtype=float)
    if not vectors:
        return np.stack([np.cos(array), np.sin(array)], axis=-1)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.where(norms > 0, norms, 1.0)


def _polar_order(headings: npt.ArrayLike, *, vectors: bool = False) -> Array:
    """Return the length of the mean heading of angles or velocity vectors, over the agents of every frame."""
    units = _unit_vectors(headings, vectors=vectors)
    return np.linalg.norm(units.mean(axis=-2), axis=-1)


def _nematic_order(headings: npt.ArrayLike) -> Array:
    """Return `|<exp(2 i theta)>|` of angles over the agents of every frame."""
    return np.abs(np.mean(np.exp(2j * np.asarray(headings, dtype=float)), axis=-1))


def _phase_order(phases: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Return Kuramoto's complex order parameter `<exp(i theta)>` over the agents of every frame."""
    return np.asarray(np.mean(np.exp(1j * np.asarray(phases, dtype=float)), axis=-1), dtype=np.complex128)


def _milling_order(positions: npt.ArrayLike, velocities: npt.ArrayLike) -> Array:
    """Return the mean normalized angular momentum of agents in a plane about their centre, for every frame."""
    places = np.asarray(positions, dtype=float)
    moving = _unit_vectors(velocities, vectors=True)
    relative = _unit_vectors(places - places.mean(axis=-2, keepdims=True), vectors=True)
    momentum = relative[..., 0] * moving[..., 1] - relative[..., 1] * moving[..., 0]
    return np.abs(momentum.mean(axis=-1))


class OrderAnalysis(BaseAnalysis):
    """Collective order of the agents of a group at every frame, of one kind.

    The kinds read headings, velocities, or positions:

    - `'polar'`: the length of the mean heading, 1 when every agent moves the same way and about `1 / sqrt(n)` for
      random headings. It reads the velocities of a trajectory, or its headings without velocities.
    - `'milling'`: the mean normalized angular momentum of the agents about their centre in a plane, 1 when every agent
      turns the same way and near 0 for a polarized or disordered group.
    - `'nematic'`: `|<exp(2 i theta)>|` of the headings, which ignores the sign of a heading.
    - `'phase'`: Kuramoto's complex order parameter `<exp(i theta)>` of phases or headings, whose modulus is the
      synchrony and whose argument is the mean phase.

    An array holds one snapshot, or one snapshot per frame along its first axis. It holds the heading angles, the
    velocity vectors for `'polar'` with `vectors=True`, or for `'milling'` the positions followed by the velocities of
    every agent along the last axis.

    Args:
        kind:
            The order to compute, one of the kinds above.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        vectors:
            Whether an array holds velocity vectors rather than heading angles.

    Attributes:
        headings_:
            The heading angles, frames by agents, or `None` without them.

        velocities_:
            The velocities, frames by agents by dimensions, or `None` without them.

        positions_:
            The positions, frames by agents by dimensions, or `None` without them.

        order_:
            One order per frame, complex for `'phase'`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import OrderAnalysis
        >>> OrderAnalysis().check().prepare([0.0, 0.0, 0.0]).run().order_.tolist()
        [1.0]
        >>> OrderAnalysis('nematic').check().prepare([0.0, np.pi]).run().order_.round(6).tolist()
        [1.0]
        >>> angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        >>> ring = np.column_stack([np.cos(angles), np.sin(angles)])
        >>> mill = np.hstack([ring, np.column_stack([-ring[:, 1], ring[:, 0]])])
        >>> OrderAnalysis('milling').check().prepare(mill).run().order_.round(6).tolist()
        [1.0]
    """

    def __init__(self, kind: str = 'polar', group: str | None = None, *, vectors: bool = False) -> None:
        self.kind = kind
        self.group = group
        self.vectors = vectors

    def _check_ignored(self) -> None:
        """Raise an error for a setting changed from its default that the kind does not read."""
        defaults = {
            name: parameter.default for name, parameter in inspect.signature(type(self).__init__).parameters.items()
        }
        read = SETTINGS[self.kind] | {'kind', 'group'}
        ignored = [name for name, value in self.get_params().items() if name not in read and value != defaults[name]]
        if ignored:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} does not read {", ".join(ignored)}. It reads '
                f'{", ".join(sorted(SETTINGS[self.kind])) or "no other setting"}.'
            )
            raise AnalysisError(error_msg)

    def _check(self) -> Self:
        """Validate the kind and that the kind reads every changed setting."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        self._check_ignored()
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the agents kind."""
        return DataKind.AGENTS

    def _prepare_array(self, source: Source) -> Self:
        """Read the headings, velocities, or positions and velocities of one snapshot or of every frame."""
        array = np.asarray(source, dtype=float)
        self.headings_ = self.velocities_ = self.positions_ = None
        if self.kind == 'milling':
            frames = array[None] if array.ndim == 2 else array  # noqa: PLR2004
            half = frames.shape[-1] // 2
            self.positions_, self.velocities_ = frames[..., :half], frames[..., half:]
        elif self.vectors:
            self.velocities_ = array[None] if array.ndim == 2 else array  # noqa: PLR2004
        else:
            self.headings_ = array[None] if array.ndim == 1 else array
        return self

    def _prepare(self, source: Source) -> Self:
        """Read the agents of a trajectory, or the snapshots of an array."""
        if not isinstance(source, Trajectory):
            return self._prepare_array(source)
        snapshots = self._read_snapshots(source, self.group)
        self.headings_ = snapshots.headings
        self.velocities_ = snapshots.velocities
        self.positions_ = snapshots.positions
        return self

    def _run(self) -> Self:
        """Compute the order of every frame."""
        headings, velocities, positions = self.headings_, self.velocities_, self.positions_
        if headings is None and velocities is not None:
            headings = np.arctan2(velocities[..., 1], velocities[..., 0])
        needed = (positions, velocities) if self.kind == 'milling' else (headings,)
        if any(item is None for item in needed):
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} needs the headings, velocities, or positions of the '
                'agents that the source lacks. Use a source with them.'
            )
            raise AnalysisError(error_msg)
        if self.kind == 'milling':
            self.order_ = _milling_order(positions, velocities)
        elif self.kind == 'polar':
            self.order_ = _polar_order(velocities, vectors=True) if velocities is not None else _polar_order(headings)
        elif self.kind == 'nematic':
            self.order_ = _nematic_order(headings)
        else:
            self.order_ = _phase_order(headings)
        return self
