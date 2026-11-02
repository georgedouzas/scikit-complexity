"""Define phase spaces, the product of a position lattice and class axes such as velocity or orientation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...errors import PhaseSpaceTooLargeError, ValidationError
from ...specs import AxisSpec
from ._lattice import LatticeSpace

DEFAULT_LIMIT = 2**26


class PhaseSpace(LatticeSpace):
    """Product of a position lattice and class axes, a lattice whose sites are the points of the product.

    The position axes come first and sites are in row-major order, so the sites at one position are contiguous. A class
    axis has its own cells, of width `(high - low) / size`, and its own edges, without flux for an interval and joined
    for a periodic axis. The position axes keep the boundary of the position lattice.

    Args:
        position:
            The lattice of the positions.

        axes:
            The class axes.

        limit:
            The largest number of sites of the space.

        label:
            The label of the space in its system.

    Attributes:
        coordinates_:
            The coordinate of every cell of every class axis, at the cell centres, by the name of the axis.

        position_axes_:
            The indices of the position axes among the axes of the product.

        class_axes_:
            The index of every class axis among the axes of the product, by its name.

        position_size_:
            The number of positions.

    Raises:
        PhaseSpaceTooLargeError:
            The product has more sites than the limit.

    Examples:
        >>> from skcomplex.core.spaces import LatticeSpace, PhaseSpace
        >>> from skcomplex.specs import AxisSpec
        >>> space = PhaseSpace(LatticeSpace((64,), spacing=0.2), [AxisSpec('v', 128, -6.0, 6.0)]).check().prepare()
        >>> space.size_, space.shape_, space.class_axes_
        (8192, (64, 128), {'v': 1})
        >>> space.coordinates_['v'][:2].round(4).tolist()
        [-5.9531, -5.8594]
    """

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
            fill=position.fill,
            spacing=position.spacing,
            label=label,
        )
        self.position = position
        self.axes = axes
        self.limit = limit

    def _check(self) -> None:
        """Raise an error for a position that is no square lattice, no class axes, or axes that share a name."""
        super()._check()
        self.position.check()
        if type(self.position) is not LatticeSpace or self.position.geometry != 'square':
            error_msg = f'The position of a phase space is a square lattice, but it is {self.position!r}.'
            raise ValidationError(error_msg)
        if not self.axes or not all(isinstance(axis, AxisSpec) for axis in self.axes):
            error_msg = f'A phase space has one or more class axes, each an `AxisSpec`, but it was given {self.axes!r}.'
            raise ValidationError(error_msg)
        names = [axis.check().name for axis in self.axes]
        if len(set(names)) != len(names):
            error_msg = f'The class axes of a phase space have distinct names, but they are {names}.'
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Set the shape, the size, the spacings, the coordinates, and the position of every site."""
        position = self.position.check().prepare()
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
        self.periodic_ = all(self._axis_boundary(axis) == 'periodic' for axis in range(self.n_dim_))
        self.default_visual_ = 'image'

    def _axis_boundary(self, axis: int) -> str:
        """Return the edges of an axis, those of the position lattice or those of the class axis."""
        if axis < self.position.n_dim_:
            return self.boundary
        return 'periodic' if self.axes[axis - self.position.n_dim_].periodic else 'reflecting'

    def _with_edges(self, boundary: str, fill: float) -> PhaseSpace:
        """Return the sites of the space with other edges of its position axes, the class axes keeping theirs."""
        position = LatticeSpace(self.position.shape_, boundary=boundary, fill=fill, spacing=self.position.spacings_)
        return PhaseSpace(position, self.axes, limit=self.limit).check().prepare()
