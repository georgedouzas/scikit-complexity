"""Define the matrix group spec.

It is a compact group of unitary matrices whose elements a state holds by components.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.stats

from ..errors import ValidationError
from ..types import Array, ComplexArray
from ._base_spec import BaseSpec

MATRIX_GROUPS = ('SU', 'U')
PARTS = 2


@dataclass(frozen=True)
class MatrixGroupSpec(BaseSpec):
    """Compact group of unitary matrices of a degree.

    It is `SU(n)`, of unit determinant, or `U(n)`. A state of the group holds every matrix by its `2 n^2` real
    components, the real and the imaginary part of every entry in row order. The path-ordered product of the matrices
    along the oriented edges of a path is its holonomy, the product of the matrix of every edge traversed along its
    orientation and of the inverse, the conjugate transpose, of every edge traversed against it, from the first edge of
    the path to the last.

    Args:
        name:
            `'SU'` for the special unitary group, or `'U'` for the unitary group.

        degree:
            The number of rows and columns of the matrices.

    Raises:
        ValidationError:
            The name is unknown, or the degree is not a positive integer.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import MatrixGroupSpec
        >>> group = MatrixGroupSpec('SU', 2)
        >>> group.size, group.identity.tolist()
        (8, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        >>> links = group.uniform_of(np.random.default_rng(0), 3)
        >>> loop = group.holonomy_of(links, [0, 1, 1, 0], [1, 1, -1, -1])
        >>> bool(np.allclose(loop, np.eye(2)))
        True
    """

    name: str = 'SU'
    degree: int = 2

    def _check(self) -> None:
        """Raise an error for an unknown group or a degree that is not a positive integer.

        Raises:
            ValidationError:
                The name is unknown, or the degree is not a positive integer.
        """
        if self.name not in MATRIX_GROUPS:
            error_msg = f'Unknown matrix group {self.name!r}. Use one of {", ".join(MATRIX_GROUPS)}.'
            raise ValidationError(error_msg)
        if not isinstance(self.degree, int) or self.degree < 1:
            error_msg = f'The degree of a matrix group is a positive integer, but got {self.degree!r}.'
            raise ValidationError(error_msg)

    @functools.cached_property
    def size(self) -> int:
        """Number of real components of a matrix of the group."""
        return PARTS * self.degree**2

    @functools.cached_property
    def identity(self) -> Array:
        """Components of the identity matrix."""
        return self.get_components_of(np.eye(self.degree, dtype=complex))

    def get_matrices_of(self, components: npt.ArrayLike) -> ComplexArray:
        """Return the matrices of real components on the last axis.

        Args:
            components:
                The components, `2 n^2` on the last axis.

        Returns:
            matrices:
                The complex matrices on the last two axes.

        Raises:
            ValidationError:
                The last axis does not hold `2 n^2` components.

        Examples:
            >>> from skcomplex.core.specs import MatrixGroupSpec
            >>> MatrixGroupSpec('U', 1).matrices_of([0.0, 1.0]).tolist()
            [[1j]]
        """
        array = np.asarray(components, dtype=float)
        if array.shape[-1:] != (self.size,):
            error_msg = (
                f'A matrix of {self.name}({self.degree}) has {self.size} real components, but the values have the '
                f'shape {array.shape}.'
            )
            raise ValidationError(error_msg)
        parts = array.reshape(*array.shape[:-1], self.degree, self.degree, PARTS)
        return parts[..., 0] + 1j * parts[..., 1]

    def get_components_of(self, matrices: npt.ArrayLike) -> Array:
        """Return the real components of matrices on the last two axes.

        Args:
            matrices:
                The complex matrices on the last two axes.

        Returns:
            components:
                The components, `2 n^2` on the last axis.

        Examples:
            >>> from skcomplex.core.specs import MatrixGroupSpec
            >>> MatrixGroupSpec('U', 1).components_of([[1j]]).tolist()
            [0.0, 1.0]
        """
        array = np.asarray(matrices, dtype=complex)
        return np.stack([array.real, array.imag], axis=-1).reshape(*array.shape[:-2], self.size)

    def get_projection_of(self, matrices: npt.ArrayLike) -> ComplexArray:
        """Return the nearest matrices of the group.

        They undo the rounding of repeated products. The unitary factor of the polar decomposition is the nearest
        unitary matrix, and for `SU(n)` it is divided by the principal `n`-th root of its determinant.

        Args:
            matrices:
                The complex matrices on the last two axes.

        Returns:
            projected:
                The matrices of the group.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.specs import MatrixGroupSpec
            >>> projected = MatrixGroupSpec('SU', 2).projection_of(1.01 * np.eye(2))
            >>> bool(np.allclose(projected, np.eye(2)))
            True
        """
        left, _, right = np.linalg.svd(np.asarray(matrices, dtype=complex))
        unitary = left @ right
        if self.name == 'U':
            return unitary
        phase = np.exp(1j * np.angle(np.linalg.det(unitary)) / self.degree)
        return unitary / phase[..., None, None]

    def get_near_identity_of(self, rng: np.random.Generator, width: float, count: int) -> ComplexArray:
        """Return random matrices of the group near the identity.

        They are `exp(i width H)` for random Hermitian `H`. The entries of `H` are standard normal up to Hermitian
        symmetry, and traceless for `SU(n)`. A matrix and its inverse are equally likely, so a move by these matrices is
        symmetric.

        Args:
            rng:
                The random generator.

            width:
                The spread of the matrices around the identity.

            count:
                The number of matrices.

        Returns:
            matrices:
                The matrices, one per row of the first axis.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.specs import MatrixGroupSpec
            >>> moves = MatrixGroupSpec('SU', 3).near_identity_of(np.random.default_rng(0), 0.1, 4)
            >>> bool(np.allclose(np.linalg.det(moves), 1.0))
            True
        """
        shape = (count, self.degree, self.degree)
        draws = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        hermitian = (draws + np.conj(np.swapaxes(draws, -1, -2))) / 2
        if self.name == 'SU':
            trace = np.trace(hermitian, axis1=-2, axis2=-1).real / self.degree
            hermitian = hermitian - trace[:, None, None] * np.eye(self.degree)
        values, vectors = np.linalg.eigh(hermitian)
        rotations = (vectors * np.exp(1j * width * values)[:, None, :]) @ np.conj(np.swapaxes(vectors, -1, -2))
        return np.asarray(rotations, dtype=complex)

    def get_uniform_of(self, rng: np.random.Generator, count: int) -> ComplexArray:
        """Return matrices drawn from the Haar measure of the group.

        SciPy draws unitary matrices from the Haar measure of `U(n)`, and dividing by the principal `n`-th root of the
        determinant keeps the Haar measure of `SU(n)`.

        Args:
            rng:
                The random generator.

            count:
                The number of matrices.

        Returns:
            matrices:
                The matrices, one per row of the first axis.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.specs import MatrixGroupSpec
            >>> draws = MatrixGroupSpec('SU', 2).uniform_of(np.random.default_rng(0), 5)
            >>> draws.shape, bool(np.allclose(np.linalg.det(draws), 1.0))
            ((5, 2, 2), True)
        """
        if self.degree == 1:
            unitary = np.exp(2j * np.pi * rng.random(count)).reshape(count, 1, 1)
        else:
            drawn = scipy.stats.unitary_group.rvs(self.degree, size=count, random_state=rng)
            unitary = np.asarray(drawn, dtype=complex).reshape(count, self.degree, self.degree)
        if self.name == 'U':
            return unitary
        return self.get_projection_of(unitary)

    def get_holonomy_of(self, matrices: npt.ArrayLike, edges: npt.ArrayLike, signs: npt.ArrayLike) -> ComplexArray:
        """Return the path-ordered products of the matrices of the edges along paths.

        Every path is a row of edges on the last axis with the sign of its traversal, 1 along the orientation of the
        edge, -1 against it, and 0 for padding that contributes the identity.

        Args:
            matrices:
                The matrix of every edge, one per row of the first axis.

            edges:
                The edges of the paths, in path order on the last axis.

            signs:
                The sign of the traversal of every edge, of the shape of `edges`.

        Returns:
            holonomies:
                The product along every path, with the shape of the paths and the matrix on the last two axes.

        Raises:
            ValidationError:
                The edges and the signs do not have the same shape, or a sign is not 1, -1, or 0.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.specs import MatrixGroupSpec
            >>> group = MatrixGroupSpec('U', 1)
            >>> phases = np.exp(1j * np.array([0.5, 0.25])).reshape(2, 1, 1)
            >>> round(float(np.angle(group.holonomy_of(phases, [0, 1], [1, -1]))[0, 0]), 12)
            0.25
        """
        array = np.asarray(matrices, dtype=complex)
        path = np.asarray(edges, dtype=np.intp)
        turn = np.asarray(signs)
        if path.shape != turn.shape or not np.isin(turn, (-1, 0, 1)).all():
            error_msg = (
                f'A path takes one sign of 1, -1, or 0 per edge, but got edges of the shape {path.shape} and signs '
                f'{np.unique(turn).tolist()} of the shape {turn.shape}.'
            )
            raise ValidationError(error_msg)
        identity = np.eye(self.degree, dtype=complex)
        product = np.broadcast_to(identity, (*path.shape[:-1], self.degree, self.degree)).copy()
        for step in range(path.shape[-1]):
            factor = array[np.where(turn[..., step] == 0, 0, path[..., step])]
            factor = np.where((turn[..., step] < 0)[..., None, None], np.conj(np.swapaxes(factor, -1, -2)), factor)
            product = product @ np.where((turn[..., step] == 0)[..., None, None], identity, factor)
        return product
