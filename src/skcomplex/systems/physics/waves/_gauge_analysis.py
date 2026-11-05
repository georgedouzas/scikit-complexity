"""Measure Wilson loops, Polyakov loops, the topological charge, and the Gauss residual of link states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence

import numpy as np

from ....core.analysis import BaseAnalysis
from ....core.components.spaces import CellComplexSpace, LatticeSpace
from ....core.engine.runtime import Trajectory
from ....core.errors import AnalysisError
from ....core.types import Array, DataKind

KINDS = ('wilson', 'polyakov', 'charge', 'gauss')
CHARGE_DIMENSIONS = 2
LOOP_AXES = 2


class GaugeAnalysis(BaseAnalysis):
    """Gauge observables of a link state on the cell complex of a periodic lattice, one value per frame.

    The kind `'wilson'` is the mean cosine of the oriented sum of the link angles around the rectangles of given extents
    in every plane. The kind `'polyakov'` is the mean cosine of the sum of the link angles along the last axis. The kind
    `'charge'` is the sum over the plaquettes of a two-dimensional lattice of the oriented sums wrapped into one period
    around 0, over `2 pi`. The kind `'gauss'` is the largest absolute value of the codifferential of an electric link
    state minus a site charge.

    An array of one dimension holds one snapshot, and an array of two dimensions one frame per row.

    Args:
        kind:
            `'wilson'`, `'polyakov'`, `'charge'`, or `'gauss'`.

        space:
            The cell complex of the links.

        variables:
            The dotted name of the state of a trajectory, or `None` for a source that is an array.

        extents:
            The sides of the rectangles of the kind `'wilson'`.

        charges:
            The charge of every site for the kind `'gauss'`, or `None` for none.

    Attributes:
        values_:
            The link states, with one row per frame.

        value_:
            The observable of every frame.

    Raises:
        AnalysisError:
            The kind is unknown, the space is not the cell complex of a periodic lattice, or the charge is asked of a
            lattice that does not have two dimensions.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
        >>> from skcomplex.systems.physics.waves import GaugeAnalysis
        >>> space = CellComplexSpace(LatticeSpace((4, 4))).prepare()
        >>> GaugeAnalysis('wilson', space).prepare(np.zeros(32)).run().value_.tolist()
        [1.0]
    """

    values_: Array
    value_: Array

    def __init__(
        self,
        kind: str,
        space: CellComplexSpace,
        variables: str | None = None,
        extents: tuple[int, int] = (1, 1),
        charges: Sequence[float] | None = None,
    ) -> None:
        self.kind = kind
        self.space = space
        self.variables = variables
        self.extents = extents
        self.charges = charges

    def _check(self) -> None:
        """Raise an error for an unknown kind, or a space that is not the cell complex of a periodic lattice."""
        self._check_choice('kind', self.kind, KINDS)
        place = getattr(self.space, 'space', None)
        if not isinstance(place, LatticeSpace) or place.boundary != 'periodic':
            error_msg = f'GaugeAnalysis reads the cell complex of a periodic lattice, not {self.space!r}.'
            raise AnalysisError(error_msg)
        if self.kind == 'charge' and len(np.ravel(place.shape)) != CHARGE_DIMENSIONS:
            error_msg = f'The topological charge is defined on a two-dimensional lattice, not on {place!r}.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the kind of data the analysis reads, the values of the links of a complex."""
        return DataKind.FIELD

    def _prepare(self, source: object) -> None:
        """Read the link states of a trajectory or an array, with one row per frame."""
        self.space.prepare()
        if isinstance(source, Trajectory):
            self.values_ = np.asarray(source.values(str(self.variables)), dtype=float)
            return
        self.values_ = np.atleast_2d(np.asarray(source, dtype=float))

    def _rectangles_of(self, first: int, second: int, sums: Array) -> Array:
        """Return the oriented sums around the rectangles of a plane, with the frame first and the sites next."""
        shape = self.space.space.shape_
        faces = np.flatnonzero(np.all(self.space.face_planes_ == (first, second), axis=1))
        corners = np.asarray([self.space.hyperedges_[face][0] for face in faces], dtype=np.intp)
        flat = np.zeros((sums.shape[0], int(np.prod(shape))))
        flat[:, corners] = sums[:, faces]
        grid = flat.reshape(sums.shape[0], *shape)
        loops = np.zeros_like(grid)
        for steps in itertools.product(range(self.extents[0]), range(self.extents[1])):
            loops += np.roll(grid, (-steps[0], -steps[1]), axis=(first + 1, second + 1))
        return loops

    def _run_wilson(self) -> Array:
        """Return the mean cosine of the rectangles of every plane."""
        sums = (self.space.incidences_[1] @ self.values_.T).T
        planes = np.unique(self.space.face_planes_, axis=0).reshape(-1, LOOP_AXES)
        loops = [self._rectangles_of(int(first), int(second), sums) for first, second in planes]
        axes = tuple(range(1, self.space.space.n_dim_ + 1))
        return np.mean([np.cos(loop).mean(axis=axes) for loop in loops], axis=0)

    def _run_polyakov(self) -> Array:
        """Return the mean cosine of the sums of the link angles along the last axis."""
        place = self.space.space
        last = place.n_dim_ - 1
        links = np.flatnonzero(self.space.link_axes_ == last)
        flat = np.zeros((self.values_.shape[0], place.size_))
        flat[:, self.space.edges_[None][0][links]] = self.values_[:, links]
        grid = flat.reshape(self.values_.shape[0], *place.shape_)
        return np.cos(grid.sum(axis=last + 1)).mean(axis=tuple(range(1, place.n_dim_)))

    def _run_charge(self) -> Array:
        """Return the sum of the wrapped oriented sums of the plaquettes, over `2 pi`."""
        sums = (self.space.incidences_[1] @ self.values_.T).T
        return np.angle(np.exp(1j * sums)).sum(axis=1) / (2 * np.pi)

    def _run_gauss(self) -> Array:
        """Return the largest absolute difference of the codifferential of the links and the charge of the sites."""
        divergence = -(self.space.incidences_[0].T @ self.values_.T).T / self.space.stars_[0]
        charges = np.zeros(divergence.shape[1]) if self.charges is None else np.asarray(self.charges, dtype=float)
        return np.abs(divergence - charges).max(axis=1)

    def _run(self, inputs: object = None) -> None:
        """Compute the observable of the kind of the analysis at every frame."""
        readers = {
            'wilson': self._run_wilson,
            'polyakov': self._run_polyakov,
            'charge': self._run_charge,
            'gauss': self._run_gauss,
        }
        self.value_ = readers[self.kind]()
