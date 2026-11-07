"""Decompose the flows on the edges of a cell complex, count its holes, and find the curved faces of an angle field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ..components.spaces import CellComplexSpace
from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_choice
from .types import Source

KINDS = ('hodge', 'betti', 'curvature')
DEFAULT_PERIOD = 2 * np.pi
SOLVER_TOLERANCE = 1e-14
FLAT_TOLERANCE = 1e-9

TAGS = AnalysisTags(DataKind.FIELD)


@dataclass(frozen=True)
class CellComplexResult:
    """Hodge decomposition of an edge state, Betti numbers of a complex, or curved faces of an angle state.

    Args:
        complex:
            The cell complex the analysis reads.

        values:
            The values of the source, with one row per frame.

        period:
            The period of the values.

        betti:
            The Betti numbers of every degree, for the kind `'betti'`.

        gradient:
            The part of every frame that is the exterior derivative of a site potential, for the kind `'hodge'`.

        curl:
            The part of every frame that the transposed incidence of a face potential gives, for the kind `'hodge'`.

        harmonic:
            The remainder of every frame, for the kind `'hodge'`.

        potential:
            The site potential of every frame, of zero mean, for the kind `'hodge'`.

        share:
            The share of the faces of every frame that are curved, for the kind `'curvature'`.

        energy:
            The sum over the faces of every frame of one minus the cosine of the oriented sum, for the kind
            `'curvature'`.
    """

    complex: CellComplexSpace
    values: Array
    period: float
    betti: tuple[int, ...] | None = None
    gradient: Array | None = None
    curl: Array | None = None
    harmonic: Array | None = None
    potential: Array | None = None
    share: Array | None = None
    energy: Array | None = None


def cell_complex(
    source: Source,
    *,
    kind: str,
    space: CellComplexSpace | None = None,
    variables: str | None = None,
) -> CellComplexResult:
    """Return the Hodge decomposition of an edge state, the Betti numbers of a complex, or the curved faces of angles.

    It reads a state of a group on a cell complex from a trajectory, or an array with the complex passed as `space`. An
    array of one dimension holds one snapshot, and an array of two dimensions one frame per row. Every result has one
    row per frame.

    The kind `'hodge'` writes an edge state as the sum of the exterior derivative of a site potential, the transposed
    incidence of a face potential, and a remainder that both incidences map to 0. The three parts are orthogonal. The
    kind `'betti'` reads the complex alone. The kind `'curvature'` compares the oriented sum of an angle state around
    every face with the multiples of the period of the state, or of `2 pi` for an array.

    Args:
        source:
            The trajectory, an array of values on the edges, or `None` for the kind `'betti'` with a `space`.

        kind:
            `'hodge'`, `'betti'`, or `'curvature'`.

        space:
            The cell complex, or `None` for the complex of the group of `variables` on a trajectory.

        variables:
            The dotted name of the state of a trajectory, or `None` for a source that is an array.

    Returns:
        result:
            The complex, the values, and the results of the kind.

    Raises:
        AnalysisError:
            The kind is unknown, the source names no state of a cell complex, or the values do not fit the complex.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import cell_complex
        >>> from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
        >>> space = CellComplexSpace(LatticeSpace((4, 4))).prepare()
        >>> cell_complex(None, kind='betti', space=space).betti
        (1, 2, 1)
        >>> uniform = np.r_[np.ones(16), np.zeros(16)]
        >>> hodge = cell_complex(uniform, kind='hodge', space=space)
        >>> bool(np.allclose(hodge.harmonic[0], uniform))
        True
    """
    _check_choice('cell_complex', 'kind', kind, KINDS)
    if space is not None and not isinstance(space, CellComplexSpace):
        error_msg = f'cell_complex reads a CellComplexSpace, not {space!r}.'
        raise AnalysisError(error_msg)
    values, period = (np.empty((0, 0)), DEFAULT_PERIOD) if source is None else _read_values(source, variables)
    found = _space_of(source, variables) if space is None and isinstance(source, Trajectory) else space
    if found is None:
        error_msg = 'cell_complex reads no complex. Give `space`, or the state of a group on a cell complex.'
        raise AnalysisError(error_msg)
    complex_space = found.prepare()
    if kind != 'betti' and values.shape[-1] != complex_space.incidences_[0].shape[0]:
        error_msg = (
            f'cell_complex reads {values.shape[-1]} values, but the complex has '
            f'{complex_space.incidences_[0].shape[0]} edges. Read a state of a group on its edges.'
        )
        raise AnalysisError(error_msg)
    if kind == 'hodge':
        return _hodge(complex_space, values, period)
    if kind == 'betti':
        return _betti(complex_space, values, period)
    return _curvature(complex_space, values, period)


def _space_of(trajectory: Trajectory, variables: str | None) -> CellComplexSpace | None:
    """Return the complex of the group of the state a trajectory holds."""
    owner = str(variables).rpartition('.')[0]
    for element in trajectory.problem.elements:
        space = getattr(element.element, 'space', None)
        if element.path.replace('/', '.') == owner and isinstance(space, CellComplexSpace):
            return space
    return None


def _read_values(source: Source, variables: str | None) -> tuple[Array, float]:
    """Return the values of a source with one row per frame, and the period they read."""
    if isinstance(source, Trajectory):
        if variables is None:
            error_msg = 'cell_complex reads a trajectory by the dotted name of a state. Give `variables`.'
            raise AnalysisError(error_msg)
        period = source.problem.states.entry(variables).period
        return np.asarray(source.values(variables), dtype=float), DEFAULT_PERIOD if period is None else float(period)
    return np.atleast_2d(np.asarray(source, dtype=float)), DEFAULT_PERIOD


def _solve(matrix: scipy.sparse.csr_array, values: Array) -> Array:
    """Return the least-squares solution of a sparse system, of the smallest norm."""
    return scipy.sparse.linalg.lsqr(matrix, values, atol=SOLVER_TOLERANCE, btol=SOLVER_TOLERANCE, conlim=0.0)[0]


def _hodge(complex_space: CellComplexSpace, values: Array, period: float) -> CellComplexResult:
    """Return the gradient, the curl, the harmonic part, and the site potential of every frame."""
    edges, faces = complex_space.incidences_[:2]
    potentials = np.stack([_solve(edges, flow) for flow in values])
    gradient = (edges @ potentials.T).T
    circulations = np.stack([_solve(faces.T, flow) for flow in values]) if faces.shape[0] else None
    curl = np.zeros_like(values) if circulations is None else (faces.T @ circulations.T).T
    return CellComplexResult(
        complex_space,
        values,
        period,
        gradient=gradient,
        curl=curl,
        harmonic=values - gradient - curl,
        potential=potentials - potentials.mean(axis=1, keepdims=True),
    )


def _betti(complex_space: CellComplexSpace, values: Array, period: float) -> CellComplexResult:
    """Return the Betti numbers from the ranks of the incidence matrices."""
    matrices = complex_space.incidences_
    ranks = [0, *(int(np.linalg.matrix_rank(matrix.toarray())) for matrix in matrices), 0]
    sizes = [matrices[0].shape[1], *(matrix.shape[0] for matrix in matrices)]
    betti = tuple(sizes[k] - ranks[k + 1] - ranks[k] for k in range(len(sizes)))
    return CellComplexResult(complex_space, values, period, betti=betti)


def _curvature(complex_space: CellComplexSpace, values: Array, period: float) -> CellComplexResult:
    """Return the share of the curved faces of every frame, and the energy of their oriented sums."""
    faces = complex_space.incidences_[1]
    angles = (faces @ values.T).T * (2 * np.pi / period)
    wrapped = np.angle(np.exp(1j * angles))
    return CellComplexResult(
        complex_space,
        values,
        period,
        share=(np.abs(wrapped) > FLAT_TOLERANCE).mean(axis=1),
        energy=(1.0 - np.cos(angles)).sum(axis=1),
    )
