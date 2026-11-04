"""Decompose the flows on the edges of a cell complex, count its holes, and find the curved faces of an angle field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ..core.spaces import CellComplexSpace
from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

KINDS = ('hodge', 'betti', 'curvature')
DEFAULT_PERIOD = 2 * np.pi
SOLVER_TOLERANCE = 1e-14
FLAT_TOLERANCE = 1e-9


class CellComplexAnalysis(BaseAnalysis):
    """Hodge decomposition of an edge state, Betti numbers of a complex, or curved faces of an angle state.

    It reads a state of a group on a cell complex from a trajectory, or an array with the complex passed as `space`. An
    array of one dimension holds one snapshot, and an array of two dimensions one frame per row. Every result has one
    row per frame.

    The kind `'hodge'` writes an edge state as the sum of the exterior derivative of a site potential, the transposed
    incidence of a face potential, and a remainder that both incidences map to 0. The three parts are orthogonal. The
    kind `'betti'` reads the complex alone. The kind `'curvature'` compares the oriented sum of an angle state around
    every face with the multiples of the period of the state, or of `2 pi` for an array.

    Args:
        kind:
            `'hodge'`, `'betti'`, or `'curvature'`.

        space:
            The cell complex, or `None` for the complex of the group of `variables` on a trajectory.

        variables:
            The dotted name of the state of a trajectory, or `None` for a source that is an array.

    Attributes:
        betti_:
            The Betti numbers of every degree, for the kind `'betti'`.

        gradient_:
            The part of every frame that is the exterior derivative of a site potential, for the kind `'hodge'`.

        curl_:
            The part of every frame that the transposed incidence of a face potential gives, for the kind `'hodge'`.

        harmonic_:
            The remainder of every frame, for the kind `'hodge'`.

        potential_:
            The site potential of every frame, of zero mean, for the kind `'hodge'`.

        share_:
            The share of the faces of every frame that are curved, for the kind `'curvature'`.

        energy_:
            The sum over the faces of every frame of one minus the cosine of the oriented sum, for the kind
            `'curvature'`.

    Raises:
        AnalysisError:
            The kind is unknown, the source names no state of a cell complex, or the values do not fit the complex.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import CellComplexAnalysis
        >>> from skcomplex.core.spaces import CellComplexSpace, LatticeSpace
        >>> space = CellComplexSpace(LatticeSpace((4, 4))).check().prepare()
        >>> CellComplexAnalysis('betti', space=space).check().prepare().run().betti_
        (1, 2, 1)
        >>> uniform = np.r_[np.ones(16), np.zeros(16)]
        >>> hodge = CellComplexAnalysis('hodge', space=space).check().prepare(uniform).run()
        >>> bool(np.allclose(hodge.harmonic_[0], uniform))
        True
    """

    betti_: tuple[int, ...]
    gradient_: Array
    curl_: Array
    harmonic_: Array
    potential_: Array
    share_: Array
    energy_: Array

    def __init__(
        self,
        kind: str,
        space: CellComplexSpace | None = None,
        variables: str | None = None,
    ) -> None:
        self.kind = kind
        self.space = space
        self.variables = variables

    def _check(self) -> None:
        """Raise an error for an unknown kind, or a space that is not a cell complex."""
        self._check_choice('kind', self.kind, KINDS)
        if self.space is not None and not isinstance(self.space, CellComplexSpace):
            error_msg = f'CellComplexAnalysis reads a CellComplexSpace, not {self.space!r}.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the kind of data the analysis reads, the values of the cells of a complex."""
        return DataKind.FIELD

    def _space_of(self, trajectory: Trajectory) -> CellComplexSpace | None:
        """Return the complex of the group of the state a trajectory holds."""
        owner = str(self.variables).rpartition('.')[0]
        for element in trajectory.problem.elements:
            space = getattr(element.element, 'space', None)
            if element.path.replace('/', '.') == owner and isinstance(space, CellComplexSpace):
                return space
        return None

    def _read_values(self, source: Source) -> Array:
        """Return the values of a source with one row per frame, setting the complex and the period they read."""
        self.period_ = DEFAULT_PERIOD
        if isinstance(source, Trajectory):
            if self.variables is None:
                error_msg = 'CellComplexAnalysis reads a trajectory by the dotted name of a state. Give `variables`.'
                raise AnalysisError(error_msg)
            self.complex_ = self.space if self.space is not None else self._space_of(source)
            period = source.problem.states.entry(self.variables).period
            self.period_ = DEFAULT_PERIOD if period is None else float(period)
            return np.asarray(source.values(self.variables), dtype=float)
        return np.atleast_2d(np.asarray(source, dtype=float))

    def _prepare(self, source: Source) -> None:
        """Read the values of the source, and the complex they live on."""
        self.complex_ = self.space
        self.values_ = np.empty((0, 0)) if source is None else self._read_values(source)
        if self.complex_ is None:
            error_msg = 'CellComplexAnalysis reads no complex. Give `space`, or the state of a group on a cell complex.'
            raise AnalysisError(error_msg)
        self.complex_.check().prepare()
        if self.kind != 'betti' and self.values_.shape[-1] != self.complex_.incidences_[0].shape[0]:
            error_msg = (
                f'CellComplexAnalysis reads {self.values_.shape[-1]} values, but the complex has '
                f'{self.complex_.incidences_[0].shape[0]} edges. Read a state of a group on its edges.'
            )
            raise AnalysisError(error_msg)

    def _solve(self, matrix: scipy.sparse.csr_array, values: Array) -> Array:
        """Return the least-squares solution of a sparse system, of the smallest norm."""
        return scipy.sparse.linalg.lsqr(matrix, values, atol=SOLVER_TOLERANCE, btol=SOLVER_TOLERANCE, conlim=0.0)[0]

    def _run_hodge(self) -> None:
        """Set the gradient, the curl, the harmonic part, and the site potential of every frame."""
        edges, faces = self.complex_.incidences_[:2]
        potentials = np.stack([self._solve(edges, flow) for flow in self.values_])
        self.potential_ = potentials - potentials.mean(axis=1, keepdims=True)
        self.gradient_ = (edges @ potentials.T).T
        circulations = np.stack([self._solve(faces.T, flow) for flow in self.values_]) if faces.shape[0] else None
        self.curl_ = np.zeros_like(self.values_) if circulations is None else (faces.T @ circulations.T).T
        self.harmonic_ = self.values_ - self.gradient_ - self.curl_

    def _run_betti(self) -> None:
        """Set the Betti numbers from the ranks of the incidence matrices."""
        matrices = self.complex_.incidences_
        ranks = [0, *(int(np.linalg.matrix_rank(matrix.toarray())) for matrix in matrices), 0]
        sizes = [matrices[0].shape[1], *(matrix.shape[0] for matrix in matrices)]
        self.betti_ = tuple(sizes[k] - ranks[k + 1] - ranks[k] for k in range(len(sizes)))

    def _run_curvature(self) -> None:
        """Set the share of the curved faces of every frame, and the energy of their oriented sums."""
        faces = self.complex_.incidences_[1]
        angles = (faces @ self.values_.T).T * (2 * np.pi / self.period_)
        wrapped = np.angle(np.exp(1j * angles))
        self.share_ = (np.abs(wrapped) > FLAT_TOLERANCE).mean(axis=1)
        self.energy_ = (1.0 - np.cos(angles)).sum(axis=1)

    def _run(self, inputs: object) -> None:
        """Compute the results of the kind of the analysis."""
        {'hodge': self._run_hodge, 'betti': self._run_betti, 'curvature': self._run_curvature}[self.kind]()
