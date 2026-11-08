"""Define simplex spaces.

They are phase spaces whose class axes are the free frequencies of a simplex.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from scipy.stats import irwinhall

from ...errors import ValidationError
from ._phase import PhaseSpace


class SimplexSpace(PhaseSpace):
    """Phase space whose class axes are the free frequencies of a simplex.

    The frequencies sum to at most one. The class axes run over `[0, 1]` with the same number of cells `n`, and the last
    frequency is one minus the sum of the others, so two class axes hold the 2-simplex of three frequencies. The cells
    whose corner farthest from the origin lies inside the simplex are whole, the cells cut by the slanted face are
    partial, and the cells beyond it are empty. Every cell has the fraction of its volume inside the simplex, and every
    face between two cells the fraction of its area inside. Finite-volume fluxes pass only through the open part of
    every face, and reductions over the class axes weight every cell by its volume.

    Args:
        position:
            The lattice of the positions.

        axes:
            The class axes, each on `[0, 1]` with the same number of cells and not periodic.

        limit:
            The largest number of sites of the space.

        label:
            The label of the space in its system.

    Attributes:
        volumes_:
            The fraction of the volume of every cell inside the simplex, with one entry along every position axis.

        apertures_:
            The fraction of the area inside the simplex of every face along every axis of the product, one more face
            than cells along a class axis, and one entry along the position axes.

    Raises:
        PhaseSpaceTooLargeError:
            The product has more sites than the limit.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace, SimplexSpace
        >>> from skcomplex.core.specs import AxisSpec
        >>> axes = [AxisSpec('x', 10, 0.0, 1.0), AxisSpec('y', 10, 0.0, 1.0)]
        >>> space = SimplexSpace(LatticeSpace((1,)), axes).prepare()
        >>> space.shape_, float(space.volumes_.sum())
        ((1, 10, 10), 50.0)
        >>> sorted(set(space.volumes_.ravel().tolist()))
        [0.0, 0.5, 1.0]
    """

    def _prepare(self, context: None = None) -> None:
        """Set the sites of the phase space and the cell volumes and face apertures.

        It first raises an error for class axes that are not intervals on `[0, 1]` of equal cells. They must have the
        same number of cells.
        """
        super()._prepare(context)
        sizes = {int(axis.size) for axis in self.axes}
        unit = all(axis.low == 0.0 and axis.high == 1.0 and not axis.periodic for axis in self.axes)
        if len(sizes) != 1 or not unit:
            error_msg = (
                'The class axes of a simplex space are intervals on [0, 1] with the same number of cells, such as '
                f"`AxisSpec('x', 30, 0.0, 1.0)`, but they are {list(self.axes)!r}."
            )
            raise ValidationError(error_msg)
        dimensions = len(self.axes)
        cells = int(self.axes[0].size)
        positions = (1,) * len(self.position_axes_)
        indices = np.indices((cells,) * dimensions).sum(axis=0)
        self.volumes_ = np.reshape(self._fraction(cells - indices, dimensions), (*positions, *indices.shape))
        apertures = [np.ones((1,) * self.n_dim_) for _ in self.position_axes_]
        for axis in range(dimensions):
            faces = [cells] * dimensions
            faces[axis] = cells + 1
            grid = np.indices(faces)
            others = grid.sum(axis=0) - grid[axis]
            fraction = self._fraction(cells - grid[axis] - others, dimensions - 1)
            apertures.append(np.reshape(fraction, (*positions, *fraction.shape)))
        self.apertures_ = tuple(apertures)

    @staticmethod
    def _fraction(room: npt.NDArray[np.int_], dimensions: int) -> npt.NDArray[np.float64]:
        """Return the fraction of a unit cube whose coordinates sum to at most the room.

        The cube has a dimension, and the room is that of every cell.
        """
        if dimensions == 0:
            return (room > 0).astype(float)
        return np.asarray(irwinhall(dimensions).cdf(room), dtype=float)
