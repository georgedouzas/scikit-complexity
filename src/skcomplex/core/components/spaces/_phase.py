"""Define the space of positions combined with extra axes, such as velocity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from ...errors import PhaseSpaceTooLargeError, ValidationError
from ._axis_spec import AxisSpec
from ._lattice import LatticeSpace

DEFAULT_LIMIT = 2**26


class PhaseSpace(LatticeSpace):
    """A space of positions combined with extra axes, such as velocity.

    Each site pairs one position with one cell of every extra axis, so a density can vary over position and a trait
    together, such as the 64 positions and 128 velocities of
    `PhaseSpace(LatticeSpace(64), [AxisSpec('v', 128, -6.0, 6.0)])`. The position axes come first and keep the walls of
    the lattice, and a field on the space takes the discretization of the lattice. Gas molecules are followed over
    position and velocity, bacteria over position and swimming direction, and cells over position and age.

    Read more in the user guide, under Spaces.

    Args:
        position:
            The lattice of the positions.

        axes:
            The class axes.

        limit:
            The largest number of sites of the space.

        label:
            The name of the space in its system, such as `'grid'`.

    Attributes:
        coordinates_:
            The coordinate of every cell of every class axis, at the cell centres, by the name of the axis.

        position_axes_:
            The indices of the position axes among the axes of the product.

        class_axes_:
            The index of every class axis among the axes of the product, by its name.

        position_size_:
            The number of positions.

        axes_periodic_:
            Whether the edges of every axis of the product are joined.

    Raises:
        PhaseSpaceTooLargeError:
            The product has more sites than the limit.

    See Also:
        LatticeSpace: Holds the positions alone.
        AxisSpec: Cuts the range of one extra axis into cells.

    Examples:
        >>> from skcomplex.core.components.spaces import AxisSpec, LatticeSpace, PhaseSpace
        >>> space = PhaseSpace(LatticeSpace(64), [AxisSpec('v', 128, -6.0, 6.0)]).prepare()
        >>> space.size_, space.shape_
        (8192, (64, 128))
    """

    space_tags: ClassVar[tuple[str, ...]] = ('phase space',)

    def __init__(
        self,
        position: LatticeSpace,
        axes: Sequence[AxisSpec],
        *,
        limit: int = DEFAULT_LIMIT,
        label: str = 'space',
    ) -> None:
        super().__init__(
            position.shape,
            geometry=position.geometry,
            boundary=position.boundary,
            discretization=position.discretization,
            scheme=position.scheme,
            symmetry=position.symmetry,
            fill=position.fill,
            spacing=position.spacing,
            label=label,
        )
        self.position = position
        self.axes = axes
        self.limit = limit

    def _prepare(self, context: None = None) -> None:
        """Set the shape, the size, the spacings, the coordinates, and the site positions.

        It first raises an error for a bad position, no class axes, or axes sharing a name. The position must be a
        square lattice, which preparing it checks. The lattice base checks the shape, the geometry, and the boundary
        the space takes from its position, and its facts are replaced by those of the phase space.

        Args:
            context:
                Nothing, since a space is prepared alone.
        """
        super()._prepare(context)
        if type(self.position) is not LatticeSpace or self.position.geometry != 'square':
            error_msg = f'The position of a phase space is a square lattice, but it is {self.position!r}.'
            raise ValidationError(error_msg)
        if not self.axes or not all(isinstance(axis, AxisSpec) for axis in self.axes):
            error_msg = f'A phase space has one or more class axes, each an `AxisSpec`, but it was given {self.axes!r}.'
            raise ValidationError(error_msg)
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            error_msg = f'The class axes of a phase space have distinct names, but they are {names}.'
            raise ValidationError(error_msg)
        position = self.position.prepare()
        self.position_size_ = position.size_
        self.position_axes_ = tuple(range(position.n_dim_))
        self.class_axes_ = {axis.name: position.n_dim_ + index for index, axis in enumerate(self.axes)}
        self.shape_ = (*position.shape_, *(int(axis.size) for axis in self.axes))
        self.n_dim_ = len(self.shape_)
        self.size_ = int(np.prod(self.shape_, dtype=object))
        if self.size_ > self.limit:
            error_msg = f'The phase space would take {self.size_} sites, above its limit of {self.limit}.'
            raise PhaseSpaceTooLargeError(error_msg)
        widths = [(axis.high - axis.low) / axis.size for axis in self.axes]
        self.spacings_ = (*position.spacings_, *widths)
        self.coordinates_ = {
            axis.name: axis.low + (np.arange(axis.size) + 0.5) * width
            for axis, width in zip(self.axes, widths, strict=True)
        }
        self.positions_ = np.indices(self.shape_).reshape(self.n_dim_, -1).T * np.asarray(self.spacings_)
        self.axes_periodic_ = tuple(self._get_axis_boundary(axis) == 'periodic' for axis in range(self.n_dim_))
        self.periodic_ = all(self.axes_periodic_)
        self.default_visual_ = 'image'

    def _get_axis_boundary(self, axis: int) -> str:
        """Return the edges of an axis.

        They are those of the position lattice or those of the class axis.
        """
        if axis < self.position.n_dim_:
            return self._get_edges()
        return 'periodic' if self.axes[axis - self.position.n_dim_].periodic else 'reflecting'

    def _copy_with_edges(self, boundary: str, fill: float) -> PhaseSpace:
        """Return the sites of the space with other edges of its position axes.

        The class axes keep theirs.
        """
        position = LatticeSpace(self.position.shape_, boundary=boundary, fill=fill, spacing=self.position.spacings_)
        return PhaseSpace(position, self.axes, limit=self.limit).prepare()
