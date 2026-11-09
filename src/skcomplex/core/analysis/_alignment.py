"""Measure the alignment of angles or vectors.

The measures are the mean vector, the mean rotation, the axial order, and the mean phase.

`SETTINGS` holds the settings every kind reads, besides `kind`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import AnalysisTags, TensorSpec
from ..types import Array, DataKind
from ._base import (
    _check_choice,
    _check_settings,
    _get_data_kinds,
    _get_lattice_names,
    _get_phase_states,
    _read_fields,
    _read_series,
    _read_snapshots,
)
from .types import Source

KINDS = ('mean-vector', 'mean-rotation', 'axial', 'mean-phase', 'principal-axis')
SETTINGS = {
    'mean-vector': {'variables', 'group', 'vectors'},
    'mean-rotation': {'group'},
    'axial': {'variables', 'group'},
    'mean-phase': {'variables', 'group'},
    'principal-axis': {'variables'},
}
READ_KINDS = (DataKind.AGENTS, DataKind.FIELD, DataKind.SERIES)
VECTOR_DIMENSIONS = 2
SYMMETRIC_TENSORS = {
    2: TensorSpec('traceless', 2),
    3: TensorSpec('symmetric', 2),
    5: TensorSpec('traceless', 3),
    6: TensorSpec('symmetric', 3),
}
TAGS = AnalysisTags(DataKind.AGENTS)


@dataclass(frozen=True)
class AlignmentResult:
    """Alignment of the members of a source at every frame.

    Args:
        angles:
            The angles, frames by members, or `None` without them.

        vectors:
            The vectors or velocities, frames by members by components, or `None` without them.

        positions:
            The positions of agents, frames by agents by dimensions, or `None` without them.

        order:
            The alignment of every frame, between 0 and 1, or the scalar order of every site of every frame.

        tensors:
            The independent components of the tensors, with one frame per row of the first axis, for `'principal-axis'`.

        axes:
            The principal axis of every site of every frame, with the components on the last axis, for
            `'principal-axis'`.

        phase:
            The mean phase of every frame, for `'mean-phase'`.
    """

    angles: Array | None
    vectors: Array | None
    positions: Array | None
    order: Array
    tensors: Array | None = None
    axes: Array | None = None
    phase: Array | None = None


def alignment(
    source: Source,
    *,
    kind: str = 'mean-vector',
    variables: Sequence[str] | None = None,
    group: str | None = None,
    vectors: bool = False,
) -> AlignmentResult:
    """Return the alignment of the members of a source at every frame, of one kind.

    The members are the agents of a group, the sites of a field, or the columns of a set of series, and each holds an
    angle or a vector. A trajectory gives the agents of a group with their directions or phases, velocities, and
    positions, else every frame of a field of angles, or of vectors with `vectors=True`, else the series of `variables`
    or the states with the role `'phase'`, one column per member and one row per sample.

    - `'mean-vector'`: the length of the mean unit vector, 1 when every member points the same way and about
      `1 / sqrt(n)` for random directions. It reads vectors when the source has them, and angles otherwise.
    - `'mean-rotation'`: the mean normalized angular momentum of agents about their centre in a plane, 1 when every
      agent turns the same way and near 0 for a polarized or disordered group. It reads positions and velocities.
    - `'axial'`: `|<exp(2 i theta)>|` of the angles, which ignores the sign of a direction.
    - `'mean-phase'`: the modulus `r` and the argument `psi` of the mean `r exp(i psi) = <exp(i theta)>` of the
      angles, the synchrony and the mean phase.
    - `'principal-axis'`: the eigenvector of the largest eigenvalue of a symmetric tensor at every site, `axes`, and
      the scalar order of the tensor at every site, `order`. It reads a field of a symmetric tensor state, or an array
      of its independent components on the last axis, whose number gives the symmetry and the dimension: 2 and 5 for a
      symmetric traceless tensor in 2 and 3 dimensions, and 3 and 6 for a symmetric tensor.

    Angles are derived from the first two components of vectors when a source has no angles, and a vector of one
    component points at the angle 0 or `pi` by its sign. An array of angles holds
    one snapshot when it has one dimension, and one frame per row of its first axis otherwise, every other axis
    indexing members. An array of vectors, with `vectors=True`, holds the components along its last axis, and one
    snapshot when it has two dimensions. An array for `'mean-rotation'` holds the positions followed by the
    velocities of every agent along its last axis.

    Args:
        source:
            The trajectory, or an array of angles, vectors, positions and velocities, or tensor components.

        kind:
            The alignment to compute, one of the kinds above.

        variables:
            The field or the series of a trajectory to read, or `None` for its agents, or else its first field, its
            states with the role `'phase'`, or its scalar series.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        vectors:
            Whether the members hold vectors rather than angles.

    Returns:
        result:
            The members read and their alignment at every frame.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, the trajectory holds no kind of data the
            function reads, or the source lacks what the kind needs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import alignment
        >>> alignment([0.0, 0.0, 0.0]).order.tolist()
        [1.0]
        >>> alignment([0.0, np.pi], kind='axial').order.round(6).tolist()
        [1.0]
        >>> angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        >>> ring = np.column_stack([np.cos(angles), np.sin(angles)])
        >>> mill = np.hstack([ring, np.column_stack([-ring[:, 1], ring[:, 0]])])
        >>> alignment(mill, kind='mean-rotation').order.round(6).tolist()
        [1.0]
        >>> phases = alignment([[0.5, 0.5], [0.0, np.pi]], kind='mean-phase')
        >>> phases.order.round(6).tolist(), phases.phase.round(6)[:1].tolist()
        ([1.0, 0.0], [0.5])
        >>> alignment([[0.5, 0.0], [0.0, 0.5]], kind='principal-axis').order.tolist()
        [[1.0, 1.0]]
    """
    _check_choice('alignment', 'kind', kind, KINDS)
    _check_settings(alignment, 'kind', SETTINGS[kind], kind=kind, variables=variables, group=group, vectors=vectors)
    if isinstance(source, Trajectory):
        angles, velocities, positions, tensors = _read_trajectory(source, kind, variables, group, vectors=vectors)
    else:
        angles, velocities, positions, tensors = _read_array(source, kind, vectors=vectors)
    if kind == 'principal-axis':
        axes, order = _principal_axis(tensors)
        return AlignmentResult(angles, velocities, positions, order, tensors=tensors, axes=axes)
    derived = angles
    if derived is None and velocities is not None:
        across = velocities[..., 1] if velocities.shape[-1] > 1 else np.zeros_like(velocities[..., 0])
        derived = np.arctan2(across, velocities[..., 0])
    error_msg = (
        f'alignment of kind {kind!r} needs the angles, vectors, or positions that the source lacks. Use a source with '
        'them.'
    )
    if kind == 'mean-rotation':
        if positions is None or velocities is None:
            raise AnalysisError(error_msg)
        return AlignmentResult(angles, velocities, positions, _mean_rotation(positions, velocities))
    if derived is None:
        raise AnalysisError(error_msg)
    if kind == 'mean-vector':
        order = (
            _mean_vector_length(velocities, vectors=True) if velocities is not None else _mean_vector_length(derived)
        )
        return AlignmentResult(angles, velocities, positions, order)
    if kind == 'axial':
        return AlignmentResult(angles, velocities, positions, _axial_order(derived))
    mean = _mean_phase(derived)
    return AlignmentResult(angles, velocities, positions, np.abs(mean), phase=np.angle(mean))


def _principal_axis(tensors: Array | None) -> tuple[Array, Array]:
    """Return the principal axis and the scalar order of symmetric tensors.

    They are given for every site of every frame.
    """
    if tensors is None:
        error_msg = 'alignment of kind principal-axis reads a field or an array of tensor components.'
        raise AnalysisError(error_msg)
    columns = tensors.shape[-1]
    if columns not in SYMMETRIC_TENSORS:
        error_msg = f'A symmetric tensor has 2, 3, 5, or 6 independent components, but the source has {columns}.'
        raise AnalysisError(error_msg)
    return SYMMETRIC_TENSORS[columns].get_principal_of(tensors)


def _read_array(
    source: Source,
    kind: str,
    *,
    vectors: bool,
) -> tuple[Array | None, Array | None, Array | None, Array | None]:
    """Return the angles, vectors, positions, and tensors of an array.

    The array holds one snapshot or every frame.
    """
    array = np.asarray(source, dtype=float)
    if kind == 'principal-axis':
        return None, None, None, (array[None] if array.ndim == VECTOR_DIMENSIONS else array)
    if kind == 'mean-rotation':
        frames = array[None] if array.ndim == VECTOR_DIMENSIONS else array
        half = frames.shape[-1] // 2
        return None, frames[..., half:], frames[..., :half], None
    if vectors:
        frames = array[None] if array.ndim == VECTOR_DIMENSIONS else array
        return None, frames.reshape(frames.shape[0], -1, frames.shape[-1]), None, None
    frames = array[None] if array.ndim == 1 else array
    return frames.reshape(frames.shape[0], -1), None, None, None


def _read_trajectory(
    source: Trajectory,
    kind: str,
    variables: Sequence[str] | None,
    group: str | None,
    *,
    vectors: bool,
) -> tuple[Array | None, Array | None, Array | None, Array | None]:
    """Return the agents of a trajectory, or else its field or its series.

    They come as angles, vectors, positions, and tensors.
    """
    kinds = _get_data_kinds(source)
    name = variables[0] if variables else None
    if DataKind.AGENTS in kinds and (group is not None or variables is None):
        snapshots = _read_snapshots(source, group)
        angles = snapshots.directions if snapshots.directions is not None else snapshots.phases
        return angles, snapshots.velocities, snapshots.positions, None
    if DataKind.FIELD in kinds and (name is None or name in _get_lattice_names(source.problem)):
        return _read_array(_read_fields(source, name), kind, vectors=vectors)
    if DataKind.SERIES in kinds:
        names = variables or _get_phase_states(source.problem) or None
        return np.asarray(_read_series(source, names).values, dtype=float), None, None, None
    held = ', '.join(item.value for item in kinds) or 'none'
    error_msg = (
        f'alignment reads {", ".join(item.value for item in READ_KINDS)}, but the trajectory holds {held}. Use a '
        'source with agents, a field of angles or vectors, or series of angles.'
    )
    raise AnalysisError(error_msg)


def _unit_vectors(values: npt.ArrayLike, *, vectors: bool) -> Array:
    """Return unit vectors from angles, or normalized vectors.

    The members lie on the second to last axis.
    """
    array = np.asarray(values, dtype=float)
    if not vectors:
        return np.stack([np.cos(array), np.sin(array)], axis=-1)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.where(norms > 0, norms, 1.0)


def _mean_vector_length(values: npt.ArrayLike, *, vectors: bool = False) -> Array:
    """Return the length of the mean unit vector of angles or vectors.

    The mean is over the members of every frame.
    """
    units = _unit_vectors(values, vectors=vectors)
    return np.linalg.norm(units.mean(axis=-2), axis=-1)


def _axial_order(angles: npt.ArrayLike) -> Array:
    """Return `|<exp(2 i theta)>|` of angles over the members of every frame."""
    return np.abs(np.mean(np.exp(2j * np.asarray(angles, dtype=float)), axis=-1))


def _mean_phase(angles: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Return the mean `<exp(i theta)>` of angles over the members of every frame."""
    return np.asarray(np.mean(np.exp(1j * np.asarray(angles, dtype=float)), axis=-1), dtype=np.complex128)


def _mean_rotation(positions: npt.ArrayLike, velocities: npt.ArrayLike) -> Array:
    """Return the mean normalized angular momentum of agents about their centre.

    The agents move in a plane. It is given for every frame.
    """
    places = np.asarray(positions, dtype=float)
    moving = _unit_vectors(velocities, vectors=True)
    relative = _unit_vectors(places - places.mean(axis=-2, keepdims=True), vectors=True)
    momentum = relative[..., 0] * moving[..., 1] - relative[..., 1] * moving[..., 0]
    return np.abs(momentum.mean(axis=-1))
