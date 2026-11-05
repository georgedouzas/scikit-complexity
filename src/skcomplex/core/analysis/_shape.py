"""Measure the geometry of chains of points: radius of gyration, end-to-end distance, and deviation from a reference."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

SHAPE_KINDS = ('gyration', 'end-to-end', 'rmsd')
FRAME_DIMENSIONS = 3


class ShapeAnalysis(BaseAnalysis):
    """Geometry of a chain of points per frame, from the positions of the members of a group in member order.

    Args:
        kind:
            `'gyration'` for the radius of gyration, `'end-to-end'` for the distance between the first and the last
            point, or `'rmsd'` for the root mean square deviation from `reference` after optimal superposition.

        group:
            The path of an agent or particle group of a trajectory, or the dotted name of a lattice state whose last
            axis holds the coordinates of a point, or `None` for the first agent or particle group.

        reference:
            The positions the deviation compares with, points by dimensions, for the kind `'rmsd'`.

    Attributes:
        positions_:
            The positions, frames by points by dimensions.

        values_:
            The value per frame.

    Raises:
        AnalysisError:
            The kind is unknown, the kind `'rmsd'` has no reference, or the positions are not frames by points by
            dimensions.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import ShapeAnalysis
        >>> chain = np.array([[[0.0, 0.0], [3.0, 4.0]]])
        >>> ShapeAnalysis('end-to-end').prepare(chain).run().values_.tolist()
        [5.0]
    """

    def __init__(
        self,
        kind: str = 'gyration',
        group: str | None = None,
        reference: npt.ArrayLike | None = None,
    ) -> None:
        self.kind = kind
        self.group = group
        self.reference = reference

    def _get_data_kind(self) -> DataKind:
        """Return the agents kind."""
        return DataKind.AGENTS

    def _check(self) -> None:
        """Raise an error for an unknown kind, or a deviation without a reference."""
        self._check_choice('kind', self.kind, SHAPE_KINDS)
        if self.kind == 'rmsd' and self.reference is None:
            error_msg = "ShapeAnalysis(kind='rmsd') compares with a reference. Pass `reference`."
            raise AnalysisError(error_msg)

    def _read_positions(self, source: Trajectory) -> npt.ArrayLike:
        """Return the positions of a lattice state or of a group of a trajectory."""
        if self.group is not None and source.problem.states.has(self.group):
            return source.lattice(self.group)
        positions = self._read_snapshots(source, self.group).positions
        if positions is None:
            error_msg = f'The members of {self.group or "the first group"} have no positions.'
            raise AnalysisError(error_msg)
        return positions

    def _prepare(self, source: Source) -> None:
        """Read the positions of a trajectory, or an array of frames by points by dimensions."""
        positions = self._read_positions(source) if isinstance(source, Trajectory) else source
        self.positions_: Array = np.asarray(positions, dtype=float)
        if self.positions_.ndim != FRAME_DIMENSIONS:
            error_msg = f'ShapeAnalysis reads frames by points by dimensions, not the shape {self.positions_.shape}.'
            raise AnalysisError(error_msg)

    def _deviation(self, frame: Array, reference: Array) -> float:
        """Return the root mean square deviation of centred points from a centred reference after superposition."""
        rotation, _ = Rotation.align_vectors(reference, frame)
        return float(np.sqrt(np.mean(np.sum((rotation.apply(frame) - reference) ** 2, axis=1))))

    def _run(self, inputs: object = None) -> None:
        """Set the value of every frame."""
        centred = self.positions_ - self.positions_.mean(axis=1, keepdims=True)
        if self.kind == 'gyration':
            self.values_ = np.sqrt(np.mean(np.sum(centred**2, axis=2), axis=1))
        elif self.kind == 'end-to-end':
            self.values_ = np.linalg.norm(self.positions_[:, -1] - self.positions_[:, 0], axis=1)
        else:
            reference = np.asarray(self.reference, dtype=float)
            reference = reference - reference.mean(axis=0)
            self.values_ = np.array([self._deviation(frame, reference) for frame in centred])
