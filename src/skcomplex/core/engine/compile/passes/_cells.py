"""Read the degree of the groups on a cell complex and the sparse matrices of its exterior calculus."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import scipy.sparse

from ....components.elements import GroupElement
from ....components.spaces import CellComplexSpace, LatticeSpace
from ....errors import SpaceError
from ....types import Array
from ...problems._declarations_spec import DeclarationsSpec

DEGREES = {'sites': 0, 'edges': 1, 'hyperedges': 2}
PARTS = tuple(DEGREES)
STAR_PARAMETER = 'weight'


class CellTables:
    """Degrees of the groups on a cell complex, and the matrices of the exterior calculus between them."""

    @staticmethod
    def _star(space: CellComplexSpace, degree: int, declarations: DeclarationsSpec) -> Array:
        """Return the diagonal of the Hodge star of a degree, with the weights of the group on a network complex."""
        if isinstance(space.space, LatticeSpace):
            return space.stars_[degree]
        for block in declarations.blocks.values():
            component = block.node.component
            if (
                block.node.space is not space
                or not isinstance(component, GroupElement)
                or component.on != PARTS[degree]
            ):
                continue
            if STAR_PARAMETER in component.element._get_specs():
                member = component.element(component.label).set_params(**(component.values or {}))
                return np.broadcast_to(np.asarray(getattr(member, STAR_PARAMETER), dtype=float), (block.rows,))
        return space.stars_[degree]

    @staticmethod
    def _diagonal(values: Array) -> scipy.sparse.csr_array:
        """Return the sparse diagonal of some values."""
        return scipy.sparse.diags_array(values, format='csr')

    @staticmethod
    def exterior_derivative_of(space: CellComplexSpace, degree: int) -> scipy.sparse.csr_array:
        """Return the matrix of the exterior derivative of a degree.

        Args:
            space:
                The cell complex.

            degree:
                The degree of the argument.

        Returns:
            matrix:
                One row per cell of the next degree and one column per cell of the degree.

        Raises:
            SpaceError:
                The complex has no cells of the next degree.
        """
        if degree >= len(space.incidences_):
            error_msg = (
                f'An exterior derivative of degree {degree} needs cells of degree {degree + 1}, but {space!r} has none.'
            )
            raise SpaceError(error_msg)
        return space.incidences_[degree]

    @staticmethod
    def codifferential_of(
        space: CellComplexSpace,
        degree: int,
        declarations: DeclarationsSpec,
    ) -> scipy.sparse.csr_array:
        """Return the matrix `(-1)^k star^-1 d^T star` of the codifferential of a degree `k`.

        Args:
            space:
                The cell complex.

            degree:
                The degree of the argument.

            declarations:
                The layout, whose groups hold the weights of a network complex.

        Returns:
            matrix:
                One row per cell of the previous degree and one column per cell of the degree.

        Raises:
            SpaceError:
                The degree has no cells of a previous degree.
        """
        if degree == 0:
            error_msg = 'The codifferential of a site state has no degree below the sites to map to.'
            raise SpaceError(error_msg)
        inverse = CellTables._diagonal(1.0 / CellTables._star(space, degree - 1, declarations))
        star = CellTables._diagonal(CellTables._star(space, degree, declarations))
        matrix = (-1.0) ** degree * inverse @ space.incidences_[degree - 1].T @ star
        return scipy.sparse.csr_array(matrix)

    @staticmethod
    def hodge_star_of(space: CellComplexSpace, degree: int, declarations: DeclarationsSpec) -> scipy.sparse.csr_array:
        """Return the diagonal matrix of the Hodge star of a degree.

        Args:
            space:
                The cell complex.

            degree:
                The degree of the argument.

            declarations:
                The layout, whose groups hold the weights of a network complex.

        Returns:
            matrix:
                One row and one column per cell of the degree.
        """
        return CellTables._diagonal(CellTables._star(space, degree, declarations))

    @staticmethod
    def group_at(space: CellComplexSpace, degree: int, declarations: DeclarationsSpec) -> str:
        """Return the path of a group on the cells of a degree, or an empty path without one.

        Args:
            space:
                The cell complex.

            degree:
                The degree of the cells.

            declarations:
                The layout.

        Returns:
            group:
                The path of the first such group.
        """
        for name in sorted(declarations.groups):
            block = declarations.blocks[name]
            if block.node.space is space and getattr(block.node.component, 'on', None) == PARTS[min(degree, 2)]:
                return name
        return ''
