"""Define the space of positions combined with extra axes, such as velocity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from ...errors import PhaseSpaceTooLargeError, ValidationError
from ...specs import AxisSpec
from ._lattice import LatticeSpace

DEFAULT_LIMIT = 2**26


class PhaseSpace(LatticeSpace):
    """A space of positions combined with extra axes, such as velocity.

    Each site is one position paired with one value on every extra axis, called a class axis. A class axis is a range
    cut into equal cells, such as velocities from -6 to 6, and its edges are closed for an interval or joined for a
    periodic axis. The position axes come first, so the sites of one position come one after the other, and they keep
    the boundary of the position lattice.

    Unlike `LatticeSpace`, whose sites are positions alone, it adds axes of a trait, so a density can vary over position
    and trait together. Choose it when a kinetic equation follows a distribution over position and a trait.

    It appears in many fields. Kinetic theory follows gas molecules over position and velocity, bacteria are counted
    over position and swimming direction, and cell populations over position and age.

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

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
        >>> from skcomplex.core.specs import AxisSpec
        >>> space = PhaseSpace(LatticeSpace((64,), spacing=0.2), [AxisSpec('v', 128, -6.0, 6.0)]).prepare()
        >>> space.size_, space.shape_, space.class_axes_
        (8192, (64, 128), {'v': 1})
        >>> space.coordinates_['v'][:2].round(4).tolist()
        [-5.9531, -5.8594]
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
            return self.boundary
        return 'periodic' if self.axes[axis - self.position.n_dim_].periodic else 'reflecting'

    def _copy_with_edges(self, boundary: str, fill: float) -> PhaseSpace:
        """Return the sites of the space with other edges of its position axes.

        The class axes keep theirs.
        """
        position = LatticeSpace(self.position.shape_, boundary=boundary, fill=fill, spacing=self.position.spacings_)
        return PhaseSpace(position, self.axes, limit=self.limit).prepare()
