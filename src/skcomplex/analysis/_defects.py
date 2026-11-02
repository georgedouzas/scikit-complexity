"""Find the point defects of a two-dimensional field of axes or vectors by the winding of their angles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

FIELD_RANK = 2
CHARGE_TOLERANCE = 1e-6


class DefectAnalysis(BaseAnalysis):
    """Point defects of a two-dimensional field of angles, with their topological charges, at every frame.

    The winding around a plaquette of four neighbouring sites is the sum of the differences of the angles along its
    edges, each wrapped to one period around 0, over `2 pi`. A plaquette with a winding that is not 0 holds a defect of
    that charge. The charges of a periodic lattice sum to 0.

    An array of two dimensions holds one snapshot, and an array of three dimensions one frame per row of its first axis.

    Args:
        period:
            The period of the angles, `np.pi` for axes without head and tail and `2 * np.pi` for vectors.

        variables:
            The field of a trajectory to read, or `None` for its first field.

        periodic:
            Whether the lattice joins its opposite edges, or has plaquettes only between its sites.

    Attributes:
        angles_:
            The angles, with one frame per row of the first axis.

        positions_:
            The centres of the plaquettes with a defect at every frame, each an array of defects by dimensions.

        charges_:
            The charges of the defects at every frame, each an array in the order of `positions_`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import DefectAnalysis
        >>> rows, columns = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing='ij')
        >>> vortex = np.arctan2(rows - 3.5, columns - 3.5)
        >>> defects = DefectAnalysis(2 * np.pi, periodic=False).check().prepare(vortex).run()
        >>> defects.positions_[0].tolist(), defects.charges_[0].tolist()
        ([[3.5, 3.5]], [1.0])
    """

    positions_: list[Array]
    charges_: list[Array]

    def __init__(
        self,
        period: float = np.pi,
        variables: Sequence[str] | None = None,
        *,
        periodic: bool = True,
    ) -> None:
        self.period = period
        self.variables = variables
        self.periodic = periodic

    def _check(self) -> None:
        """Raise an error for a period that is not positive."""
        if not self.period > 0:
            error_msg = f'The period of the angles is {self.period!r}. Give a positive number.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the field kind, the only kind the analysis reads."""
        return DataKind.FIELD

    def _prepare(self, source: Source) -> None:
        """Read the frames of a field of angles of a trajectory or an array."""
        frames = self._read_fields(source, self.variables[0] if self.variables else None)
        if frames.ndim not in {FIELD_RANK, FIELD_RANK + 1}:
            error_msg = f'{type(self).__name__} reads a two-dimensional field, but got the shape {frames.shape}.'
            raise AnalysisError(error_msg)
        self.angles_ = frames[None] if frames.ndim == FIELD_RANK else frames

    def _wrapped(self, differences: Array) -> Array:
        """Return differences of angles in one period around 0."""
        half = self.period / 2
        return np.mod(differences + half, self.period) - half

    def _windings(self, angles: Array) -> Array:
        """Return the winding around every plaquette of a frame, in units of a full turn."""
        below = np.roll(angles, -1, axis=0)
        right = np.roll(angles, -1, axis=1)
        diagonal = np.roll(below, -1, axis=1)
        steps = [right - angles, diagonal - right, below - diagonal, angles - below]
        windings = sum(self._wrapped(step) for step in steps) / (2 * np.pi)
        return np.asarray(windings) if self.periodic else np.asarray(windings)[:-1, :-1]

    def _run(self, inputs: object = None) -> None:
        """Set the positions and the charges of the defects of every frame."""
        self.positions_, self.charges_ = [], []
        for angles in self.angles_:
            windings = self._windings(angles)
            found = np.argwhere(np.abs(windings) > CHARGE_TOLERANCE)
            self.positions_.append(np.asarray(found + 0.5, dtype=float))
            self.charges_.append(np.round(windings[tuple(found.T)], 6))
