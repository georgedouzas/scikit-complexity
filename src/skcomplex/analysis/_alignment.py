"""Measure the alignment of angles or vectors: mean vector, mean rotation, axial order, and mean phase."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

KINDS = ('mean-vector', 'mean-rotation', 'axial', 'mean-phase')
SETTINGS = {
    'mean-vector': {'variables', 'group', 'vectors'},
    'mean-rotation': {'group'},
    'axial': {'variables', 'group'},
    'mean-phase': {'variables', 'group'},
}
"""Settings every kind reads, besides `kind`."""
READ_KINDS = (DataKind.AGENTS, DataKind.FIELD, DataKind.SERIES)
VECTOR_DIMENSIONS = 2


class AlignmentAnalysis(BaseAnalysis):
    """Alignment of the members of a source at every frame, of one kind.

    The members are the agents of a group, the sites of a field, or the columns of a set of series, and each holds an
    angle or a vector. A trajectory gives the agents of a group with their headings, velocities, and positions, else
    every frame of a field of angles, or of vectors with `vectors=True`, else the series of `variables`, one column per
    member and one row per sample, such as the phases of oscillators.

    - `'mean-vector'`: the length of the mean unit vector, 1 when every member points the same way and about
      `1 / sqrt(n)` for random directions. It reads vectors when the source has them, and angles otherwise.
    - `'mean-rotation'`: the mean normalized angular momentum of agents about their centre in a plane, 1 when every
      agent turns the same way and near 0 for a polarized or disordered group. It reads positions and velocities.
    - `'axial'`: `|<exp(2 i theta)>|` of the angles, which ignores the sign of a direction.
    - `'mean-phase'`: the modulus `r` and the argument `psi` of the mean `r exp(i psi) = <exp(i theta)>` of the
      angles, the synchrony and the mean phase.

    Angles are derived from the first two components of vectors when a source has no angles. An array of angles holds
    one snapshot when it has one dimension, and one frame per row of its first axis otherwise, every other axis
    indexing members. An array of vectors, with `vectors=True`, holds the components along its last axis, and one
    snapshot when it has two dimensions. An array for `'mean-rotation'` holds the positions followed by the
    velocities of every agent along its last axis.

    Args:
        kind:
            The alignment to compute, one of the kinds above.

        variables:
            The field or the series of a trajectory to read, or `None` for its agents, or else its first field or its
            scalar series.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        vectors:
            Whether the members hold vectors rather than angles.

    Attributes:
        angles_:
            The angles, frames by members, or `None` without them.

        vectors_:
            The vectors or velocities, frames by members by components, or `None` without them.

        positions_:
            The positions of agents, frames by agents by dimensions, or `None` without them.

        order_:
            The alignment of every frame, between 0 and 1.

        phase_:
            The mean phase of every frame, for `'mean-phase'`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import AlignmentAnalysis
        >>> AlignmentAnalysis().check().prepare([0.0, 0.0, 0.0]).run().order_.tolist()
        [1.0]
        >>> AlignmentAnalysis('axial').check().prepare([0.0, np.pi]).run().order_.round(6).tolist()
        [1.0]
        >>> angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        >>> ring = np.column_stack([np.cos(angles), np.sin(angles)])
        >>> mill = np.hstack([ring, np.column_stack([-ring[:, 1], ring[:, 0]])])
        >>> AlignmentAnalysis('mean-rotation').check().prepare(mill).run().order_.round(6).tolist()
        [1.0]
        >>> phases = AlignmentAnalysis('mean-phase').check().prepare([[0.5, 0.5], [0.0, np.pi]]).run()
        >>> phases.order_.round(6).tolist(), phases.phase_.round(6)[:1].tolist()
        ([1.0, 0.0], [0.5])
    """

    def __init__(
        self,
        kind: str = 'mean-vector',
        variables: Sequence[str] | None = None,
        *,
        group: str | None = None,
        vectors: bool = False,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.group = group
        self.vectors = vectors

    def _check(self) -> Self:
        """Validate the kind and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the agents kind, the first of the kinds the analysis reads."""
        return DataKind.AGENTS

    def _prepare_array(self, source: Source) -> Self:
        """Read the angles, the vectors, or the positions and velocities of one snapshot or of every frame."""
        array = np.asarray(source, dtype=float)
        if self.kind == 'mean-rotation':
            frames = array[None] if array.ndim == VECTOR_DIMENSIONS else array
            half = frames.shape[-1] // 2
            self.positions_, self.vectors_ = frames[..., :half], frames[..., half:]
        elif self.vectors:
            frames = array[None] if array.ndim == VECTOR_DIMENSIONS else array
            self.vectors_ = frames.reshape(frames.shape[0], -1, frames.shape[-1])
        else:
            frames = array[None] if array.ndim == 1 else array
            self.angles_ = frames.reshape(frames.shape[0], -1)
        return self

    def _prepare(self, source: Source) -> Self:
        """Read the agents of a trajectory, or else its field or its series, or the snapshots of an array."""
        self.angles_ = self.vectors_ = self.positions_ = None
        if not isinstance(source, Trajectory):
            return self._prepare_array(source)
        kinds = self._get_data_kinds(source)
        name = self.variables[0] if self.variables else None
        if DataKind.AGENTS in kinds and (self.group is not None or self.variables is None):
            snapshots = self._read_snapshots(source, self.group)
            self.angles_ = snapshots.headings
            self.vectors_ = snapshots.velocities
            self.positions_ = snapshots.positions
        elif DataKind.FIELD in kinds and (name is None or name in self._get_lattice_names(source.problem)):
            self._prepare_array(self._read_fields(source, name))
        elif DataKind.SERIES in kinds:
            self.angles_ = np.asarray(self._read_series(source, self.variables).values, dtype=float)
        else:
            held = ', '.join(kind.value for kind in kinds) or 'none'
            error_msg = (
                f'{type(self).__name__} reads {", ".join(kind.value for kind in READ_KINDS)}, but the trajectory holds '
                f'{held}. Use a source with agents, a field of angles or vectors, or series of angles.'
            )
            raise AnalysisError(error_msg)
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the alignment of every frame."""
        angles, vectors, positions = self.angles_, self.vectors_, self.positions_
        if angles is None and vectors is not None:
            angles = np.arctan2(vectors[..., 1], vectors[..., 0])
        needed = (positions, vectors) if self.kind == 'mean-rotation' else (angles,)
        if any(item is None for item in needed):
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} needs the angles, vectors, or positions that the source '
                'lacks. Use a source with them.'
            )
            raise AnalysisError(error_msg)
        if self.kind == 'mean-rotation':
            self.order_ = self._mean_rotation(positions, vectors)
        elif self.kind == 'mean-vector':
            self.order_ = (
                self._mean_vector_length(vectors, vectors=True)
                if vectors is not None
                else self._mean_vector_length(angles)
            )
        elif self.kind == 'axial':
            self.order_ = self._axial_order(angles)
        else:
            mean = self._mean_phase(angles)
            self.order_ = np.abs(mean)
            self.phase_ = np.angle(mean)
        return self

    @staticmethod
    def _unit_vectors(values: npt.ArrayLike, *, vectors: bool) -> Array:
        """Return unit vectors from angles, or normalized vectors, with members on the second to last axis."""
        array = np.asarray(values, dtype=float)
        if not vectors:
            return np.stack([np.cos(array), np.sin(array)], axis=-1)
        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        return array / np.where(norms > 0, norms, 1.0)

    @staticmethod
    def _mean_vector_length(values: npt.ArrayLike, *, vectors: bool = False) -> Array:
        """Return the length of the mean unit vector of angles or vectors, over the members of every frame."""
        units = AlignmentAnalysis._unit_vectors(values, vectors=vectors)
        return np.linalg.norm(units.mean(axis=-2), axis=-1)

    @staticmethod
    def _axial_order(angles: npt.ArrayLike) -> Array:
        """Return `|<exp(2 i theta)>|` of angles over the members of every frame."""
        return np.abs(np.mean(np.exp(2j * np.asarray(angles, dtype=float)), axis=-1))

    @staticmethod
    def _mean_phase(angles: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Return the mean `<exp(i theta)>` of angles over the members of every frame."""
        return np.asarray(np.mean(np.exp(1j * np.asarray(angles, dtype=float)), axis=-1), dtype=np.complex128)

    @staticmethod
    def _mean_rotation(positions: npt.ArrayLike, velocities: npt.ArrayLike) -> Array:
        """Return the mean normalized angular momentum of agents in a plane about their centre, for every frame."""
        places = np.asarray(positions, dtype=float)
        moving = AlignmentAnalysis._unit_vectors(velocities, vectors=True)
        relative = AlignmentAnalysis._unit_vectors(places - places.mean(axis=-2, keepdims=True), vectors=True)
        momentum = relative[..., 0] * moving[..., 1] - relative[..., 1] * moving[..., 0]
        return np.abs(momentum.mean(axis=-1))
