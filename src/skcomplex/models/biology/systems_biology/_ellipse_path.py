"""Project dihedral angles onto the ellipse path of the Ramachandran map of the early-stage folding intermediate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from scipy.optimize import minimize_scalar

from ....base import BaseConfig
from ....errors import ValidationError
from ....types import Array

LONG_DIAGONAL = np.radians(83.0)
SHORT_DIAGONAL = np.radians(58.0)
GRID_POINTS = 3600
TURN = 2 * np.pi
PAIR_DIMENSIONS = 2


class EllipsePath(BaseConfig[None, Array]):
    """Ellipse path on the Ramachandran map, with the projection of dihedral angles by the shortest periodic distance.

    The path is `phi = A cos(t) - B sin(t)`, `psi = -A cos(t) - B sin(t)`, in the form of Eq. A.2 of Brylinski et al.
    (2005, J. Biomed. Biotechnol. 2005:65), which has `t = 0` in the lower right of the map and grows clockwise. The
    paper names `A` and `B` the long and the short diagonal without values. The defaults, 83° and 58°, are a least
    squares fit to the path drawn in its Figure 4(e), so they are readings of a figure and not published numbers, and
    they have not been compared with Brylinski et al. (2004, Bioinformatics 20:199).

    Args:
        long_diagonal:
            The value `A` in radians.

        short_diagonal:
            The value `B` in radians.

        cells:
            The number of equal cells of the parameter `t`.

    Attributes:
        parameters_:
            The parameter `t` of the projection of every pair of the last run, in `[0, 2 pi)`.

        angles_:
            The projected `phi` and `psi` of the last run in radians, pairs by 2.

        cells_:
            The cell of the path of every pair of the last run.

    Raises:
        ValidationError:
            A diagonal or the number of cells is not positive, or the angles are not pairs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.biology.systems_biology import EllipsePath
        >>> path = EllipsePath().check().prepare()
        >>> path.read_points(np.array([0.0])).shape
        (1, 2)
        >>> path.run(np.array([[1.0, -1.0]])).cells_.tolist()
        [0]
    """

    def __init__(
        self,
        long_diagonal: float = LONG_DIAGONAL,
        short_diagonal: float = SHORT_DIAGONAL,
        cells: int = 72,
    ) -> None:
        self.long_diagonal = long_diagonal
        self.short_diagonal = short_diagonal
        self.cells = cells

    def _check(self) -> None:
        """Raise an error for a diagonal or a number of cells that is not positive."""
        if self.long_diagonal <= 0 or self.short_diagonal <= 0 or self.cells < 1:
            error_msg = (
                'The diagonals and the cells of an ellipse path must be positive, but they are '
                f'{self.long_diagonal}, {self.short_diagonal}, and {self.cells}.'
            )
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Set the grid of the path that starts the search of a projection."""
        self.grid_ = np.linspace(0.0, TURN, GRID_POINTS, endpoint=False)
        self.grid_points_ = self.read_points(self.grid_)

    def read_points(self, parameters: Array) -> Array:
        """Return the points of the path at some values of its parameter.

        Args:
            parameters:
                The values of `t`.

        Returns:
            points:
                The `phi` and `psi` in radians, values by 2.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.models.biology.systems_biology import EllipsePath
            >>> phi, psi = EllipsePath(1.0, 0.5).read_points(np.array([0.0]))[0]
            >>> float(phi), float(psi)
            (1.0, -1.0)
        """
        along, across = self.long_diagonal * np.cos(parameters), self.short_diagonal * np.sin(parameters)
        return np.column_stack([along - across, -along - across])

    def _read_distance(self, parameter: float, pair: Array) -> float:
        """Return the shortest periodic distance between a pair of angles and a point of the path."""
        point = self.read_points(np.array([parameter]))[0]
        difference = (pair - point + np.pi) % TURN - np.pi
        return float(np.hypot(*difference))

    def _project(self, pair: Array) -> float:
        """Return the parameter of the point of the path closest to a pair, refining the best points of the grid."""
        differences = (pair - self.grid_points_ + np.pi) % TURN - np.pi
        distances = np.hypot(differences[:, 0], differences[:, 1])
        best = float(self.grid_[np.argmin(distances)])
        step = TURN / GRID_POINTS
        refined = minimize_scalar(
            self._read_distance,
            bounds=(best - step, best + step),
            args=(pair,),
            method='bounded',
            options={'xatol': 1e-10},
        )
        return float(refined.x if refined.fun <= distances.min() else best) % TURN

    def _run(self, inputs: Array | None) -> None:
        """Set the projection of pairs of `phi` and `psi` in radians."""
        pairs = np.asarray(inputs, dtype=float)
        if pairs.ndim != PAIR_DIMENSIONS or pairs.shape[1] != PAIR_DIMENSIONS:
            error_msg = f'An ellipse path projects pairs of phi and psi, not the shape {pairs.shape}.'
            raise ValidationError(error_msg)
        self.parameters_ = np.array([self._project(pair) for pair in pairs])
        self.angles_ = self.read_points(self.parameters_)
        self.cells_ = np.minimum((self.parameters_ / TURN * self.cells).astype(int), self.cells - 1)
