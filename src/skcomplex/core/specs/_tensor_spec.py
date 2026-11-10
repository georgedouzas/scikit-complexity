"""Define the spec of the independent components of a rank-2 tensor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, matrix
from sage.structure.element import Matrix

from ..errors import ValidationError
from ..types import Array, IntArray
from ._base_spec import BaseSpec

TENSOR_SYMMETRIES = ('full', 'symmetric', 'antisymmetric', 'traceless')
TENSOR_DIMENSIONS = (2, 3)
AXIS_NAMES = 'xyz'


@dataclass(frozen=True)
class TensorSpec(BaseSpec):
    """A spec of the independent components of a rank-2 tensor of one symmetry.

    The tensor lives in a space of two or three dimensions.

    Args:
        symmetry:
            `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`, which means symmetric and traceless.

        n_dim:
            The number of dimensions of the space.

    Raises:
        ValidationError:
            The symmetry is unknown or the space does not have two or three dimensions.

    Examples:
        >>> from skcomplex.core.specs import TensorSpec
        >>> tensor = TensorSpec('traceless', 2)
        >>> tensor.names
        ('xx', 'xy')
        >>> tensor.get_array_of([1.0, 2.0]).tolist()
        [[1.0, 2.0], [2.0, -1.0]]
    """

    symmetry: str
    n_dim: int

    def _check(self) -> None:
        """Raise an error for an unknown symmetry or a space of the wrong dimension.

        The space must have two or three dimensions.
        """
        if self.symmetry not in TENSOR_SYMMETRIES:
            error_msg = f'The tensor symmetry is {self.symmetry!r}. Use one of {", ".join(TENSOR_SYMMETRIES)}.'
            raise ValidationError(error_msg)
        if self.n_dim not in TENSOR_DIMENSIONS:
            error_msg = f'A tensor state needs a space of 2 or 3 dimensions, but its space has {self.n_dim}.'
            raise ValidationError(error_msg)

    def _get_unit(self, row: int, column: int) -> IntArray:
        """Return the tensor whose independent component at an index is 1.

        Its other independent components are 0.
        """
        unit = np.zeros((self.n_dim, self.n_dim), dtype=np.int64)
        unit[row, column] = 1
        if self.symmetry == 'antisymmetric':
            unit[column, row] = -1
        if self.symmetry in {'symmetric', 'traceless'}:
            unit[column, row] = 1
        if self.symmetry == 'traceless' and row == column:
            unit[-1, -1] = -1
        return unit

    @functools.cached_property
    def indices(self) -> tuple[tuple[int, int], ...]:
        """Row and column of every independent component, in row order."""
        pairs = [(row, column) for row in range(self.n_dim) for column in range(self.n_dim)]
        if self.symmetry == 'full':
            return tuple(pairs)
        if self.symmetry == 'antisymmetric':
            return tuple((row, column) for row, column in pairs if row < column)
        upper = [(row, column) for row, column in pairs if row <= column]
        return tuple(upper[:-1] if self.symmetry == 'traceless' else upper)

    @functools.cached_property
    def names(self) -> tuple[str, ...]:
        """Name of every independent component, such as `xx` and `xy`."""
        return tuple(f'{AXIS_NAMES[row]}{AXIS_NAMES[column]}' for row, column in self.indices)

    @functools.cached_property
    def basis(self) -> IntArray:
        """Tensors of the independent components.

        There is one tensor per component on the first axis.
        """
        return np.stack([self._get_unit(row, column) for row, column in self.indices])

    def get_matrix_of(self, components: Iterable[object]) -> Matrix:
        """Return the symbolic matrix of independent components.

        The dependent entries are written in terms of them.

        Args:
            components:
                The expressions of the independent components.

        Returns:
            matrix:
                The matrix over the symbolic ring.

        Examples:
            >>> from sage.all__sagemath_symbolics import var
            >>> from skcomplex.core.specs import TensorSpec
            >>> TensorSpec('traceless', 2).get_matrix_of(var('a, b'))
            [ a  b]
            [ b -a]
        """
        total = matrix(SR, self.n_dim, self.n_dim)
        for component, unit in zip(components, self.basis, strict=True):
            total += SR(component) * matrix(SR, unit.tolist())
        return total

    def get_array_of(self, components: npt.ArrayLike) -> Array:
        """Return the arrays of tensors of independent components on the last axis.

        Args:
            components:
                The independent components on the last axis.

        Returns:
            tensors:
                The tensors on the last two axes.

        Examples:
            >>> from skcomplex.core.specs import TensorSpec
            >>> TensorSpec('antisymmetric', 2).get_array_of([3.0]).tolist()
            [[0.0, 3.0], [-3.0, 0.0]]
        """
        return np.tensordot(np.asarray(components, dtype=float), self.basis, axes=1)

    def get_principal_of(self, components: npt.ArrayLike) -> tuple[Array, Array]:
        """Return the principal axis and the scalar order of symmetric tensors.

        The tensors are given by their independent components. The principal axis is the eigenvector of the largest
        eigenvalue of the traceless part. The scalar order is that eigenvalue times `n_dim / (n_dim - 1)`, so a tensor
        `S (n n - I / n_dim)` has the order `S`.

        Args:
            components:
                The independent components on the last axis.

        Returns:
            axes:
                The axes, unit vectors on the last axis, whose sign has no meaning.

            orders:
                The scalar orders.

        Raises:
            ValidationError:
                The tensor is not symmetric.

        Examples:
            >>> from skcomplex.core.specs import TensorSpec
            >>> axes, orders = TensorSpec('traceless', 2).get_principal_of([0.5, 0.0])
            >>> abs(axes).round(6).tolist(), float(orders)
            ([1.0, 0.0], 1.0)
        """
        if self.symmetry not in {'symmetric', 'traceless'}:
            error_msg = f'A tensor of the symmetry {self.symmetry!r} has no principal axes. Use a symmetric tensor.'
            raise ValidationError(error_msg)
        tensors = self.get_array_of(components)
        means = np.trace(tensors, axis1=-2, axis2=-1) / self.n_dim
        values, vectors = np.linalg.eigh(tensors - means[..., None, None] * np.eye(self.n_dim))
        return vectors[..., -1], values[..., -1] * self.n_dim / (self.n_dim - 1)

    def get_components_of(self, value: object) -> Array:
        """Return the independent components of a numeric matrix.

        Args:
            value:
                A matrix of numbers, such as a nested sequence, a NumPy array, or a Sage matrix.

        Returns:
            components:
                The independent components.

        Raises:
            ValidationError:
                The value is not a matrix of the shape and the symmetry of the tensor.

        Examples:
            >>> from skcomplex.core.specs import TensorSpec
            >>> TensorSpec('symmetric', 2).get_components_of([[1, 2], [2, 3]]).tolist()
            [1.0, 2.0, 3.0]
        """
        array = np.asarray(value.numpy() if isinstance(value, Matrix) else value, dtype=float)
        if array.shape != (self.n_dim, self.n_dim):
            error_msg = f'The value has the shape {array.shape}, but the tensor is {self.n_dim} by {self.n_dim}.'
            raise ValidationError(error_msg)
        components = np.asarray([array[row, column] for row, column in self.indices])
        if not np.allclose(self.get_array_of(components), array):
            error_msg = f'The matrix {array.tolist()} does not have the symmetry {self.symmetry!r}.'
            raise ValidationError(error_msg)
        return components
