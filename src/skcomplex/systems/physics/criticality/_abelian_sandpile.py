"""Measure the toppling matrix, the lattice Green function, and the mean avalanche size of the abelian sandpile."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from ....core.errors import AnalysisError
from ....core.types import Array
from ._criticality import SandpileSystem

KINDS = ('toppling', 'green', 'mean-size')
MAX_GREEN_SITES = 10_000
THRESHOLD = 4


@dataclass(frozen=True)
class AbelianSandpileResult:
    """Toppling matrix, lattice Green function, or mean avalanche size of an abelian sandpile.

    Args:
        kind:
            The kind of the result, `'toppling'`, `'green'`, or `'mean-size'`.

        shape:
            The shape of the lattice of the sandpile.

        toppling:
            The sparse toppling matrix `Delta` in the row-major order of the sites, for `'toppling'`.

        green:
            The inverse of the toppling matrix, whose entry `(i, j)` is the mean number of topplings at `j` after one
            grain falls on `i`, for `'green'`.

        mean_size:
            The mean number of topplings of an avalanche per grain, `(1 / N) sum_ij G_ij`, for `'mean-size'`.
    """

    kind: str
    shape: tuple[int, ...]
    toppling: scipy.sparse.csr_array | None = None
    green: Array | None = None
    mean_size: float | None = None


def abelian_sandpile(system: object, *, kind: str = 'toppling') -> AbelianSandpileResult:
    """Return the toppling matrix, the Green function, or Dhar's mean avalanche size of a sandpile.

    The Bak-Tang-Wiesenfeld sandpile is abelian: the order in which sites topple does not change the final heights, so
    its toppling matrix `Delta`, the threshold on the diagonal and minus one for every pair of neighbouring sites, holds
    its statistics. A grain that falls on site `i` makes `G_ij = (Delta^-1)_ij` topplings at site `j` on average, so
    Dhar's mean avalanche size is `(1 / N) sum_ij G_ij`, which grows as `L^2` with the side `L` of the lattice. The
    open edges make the matrix invertible, since an edge site loses the grains its missing neighbours would receive.
    It gives the avalanches of a sandpile, the relaxation of a stressed fault, and the spread of overloads.

    Args:
        system:
            The `SandpileSystem`, prepared or not, with fixed neighbours.

        kind:
            What to return. `'toppling'` gives the sparse toppling matrix. `'green'` gives its dense inverse, the
            lattice Green function, for lattices of at most `MAX_GREEN_SITES` sites. `'mean-size'` gives the mean
            avalanche size from one sparse linear solve, without the inverse.

    Returns:
        result:
            The toppling matrix, the Green function, or the mean avalanche size.

    Raises:
        AnalysisError:
            The kind is unknown, the system is no `SandpileSystem`, it has random neighbours, or the lattice is too
            large for the dense Green function.

    See Also:
        crackling_noise: Measures the scaling of the avalanches of a run.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SandpileSystem, abelian_sandpile
        >>> abelian_sandpile(SandpileSystem((3, 3))).toppling.diagonal().tolist()
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        >>> round(abelian_sandpile(SandpileSystem((1, 1)), kind='mean-size').mean_size, 2)
        0.25
    """
    if kind not in KINDS:
        error_msg = f'abelian_sandpile has kind={kind!r}. Use one of {", ".join(KINDS)}.'
        raise AnalysisError(error_msg)
    if not isinstance(system, SandpileSystem):
        error_msg = (
            f'abelian_sandpile reads a SandpileSystem, whose toppling matrix is fixed, but it was given '
            f'{type(system).__name__}.'
        )
        raise AnalysisError(error_msg)
    if system.neighbors == 'random':
        error_msg = (
            'abelian_sandpile reads a sandpile with fixed neighbours. Random neighbours change at every toppling, so '
            'the pile has no toppling matrix.'
        )
        raise AnalysisError(error_msg)
    shape = tuple(int(size) for size in system.shape)
    sites = int(np.prod(shape))
    nodes = [tuple(reversed(index)) for index in np.ndindex(*shape)]
    adjacency = nx.to_scipy_sparse_array(nx.grid_graph(dim=shape), nodelist=nodes, format='csr')
    toppling = scipy.sparse.csr_array(THRESHOLD * scipy.sparse.eye_array(sites, format='csr') - adjacency)
    if kind == 'toppling':
        return AbelianSandpileResult(kind, shape, toppling=toppling)
    if kind == 'green':
        if sites > MAX_GREEN_SITES:
            error_msg = (
                f'The lattice has {sites} sites, more than the {MAX_GREEN_SITES} whose dense Green function fits in '
                "memory. Use kind='mean-size', which solves the toppling matrix without its inverse."
            )
            raise AnalysisError(error_msg)
        return AbelianSandpileResult(kind, shape, green=scipy.linalg.inv(toppling.toarray()))
    solution = scipy.sparse.linalg.spsolve(scipy.sparse.csc_array(toppling), np.ones(sites))
    return AbelianSandpileResult(kind, shape, mean_size=float(np.mean(solution)))
