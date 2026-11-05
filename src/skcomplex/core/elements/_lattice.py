"""Define lattice groups, groups of elements with one member per site of a lattice or node of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np

from ...specs import VisualSpec
from ..spaces import BaseSpace, NetworkSpace
from ._base import BaseElement
from ._group import GroupElement


class LatticeElement(GroupElement):
    """Group of elements with one member per site of a lattice or node of a network.

    The laws of the element type are written for one cell, and local rules read the cells around it. Preparing turns
    values in the shape of the lattice, or mappings from node to value on a network, into one row per site.

    Args:
        label:
            The label of the group, unique among the parts of its system.

        element:
            The element type of the cells.

        space:
            The lattice or other space of sites, or the network whose nodes are the sites, in the order of
            `graph.nodes`.

        values:
            A mapping from state or parameter name to an array of the lattice shape, an array in node order, a mapping
            from node to value on a network, or a value shared by every cell.

        visual:
            How the group is drawn, or `None` for `default_visual_`.

    Attributes:
        default_visual_:
            The visual of a member, or the default visual of the space when the member sets none, which the group
            takes when it sets none.

        size_:
            The number of sites of the space.

        values_:
            The values with one row per site, or one value shared by every site.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
        >>> from skcomplex.core.elements import Element, LatticeElement
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.specs import StateSpec
        >>> class CellElement(Element):
        ...     def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        ...         super().__init__(label)
        ...         self.alive = alive
        >>> space = LatticeSpace((3, 4))
        >>> cells = LatticeElement('cells', element=CellElement, space=space, values={'alive': np.eye(3, 4)})
        >>> cells.prepare().size_, cells.values_['alive'].shape
        (12, (12,))
    """

    def __init__(
        self,
        label: str,
        *,
        element: type[BaseElement],
        space: BaseSpace,
        values: Mapping[str, object] | None = None,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, element=element, values=values, visual=visual, space=space, on='sites')

    def _flatten_values(self) -> dict[str, object]:
        """Return the values with mappings from node to value and arrays of the lattice shape as one row per site."""
        values = dict(self.values or {})
        if isinstance(self.space, NetworkSpace):
            nodes = list(self.space.graph.nodes)
            values = {
                name: np.asarray([value[node] for node in nodes]) if isinstance(value, Mapping) else value
                for name, value in values.items()
            }
        return {
            name: (
                np.asarray(value).ravel()
                if self.space is not None and np.asarray(value).shape == self.space.shape_
                else value
            )
            for name, value in values.items()
        }

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `size_`, the number of sites, `values_`, with one row per site, and `default_visual_`.

        The default visual is the visual of a member, or the default visual of the space when the member sets none.

        Args:
            context:
                The space the group inherits from its system, which a lattice group does not read.

        Raises:
            SpaceError:
                The space does not have sites.

            ValidationError:
                A per-site value does not have one row per site.
        """
        super()._prepare(context)
        if self.space is not None and self.default_visual_ == VisualSpec():
            self.default_visual_ = VisualSpec(self.space.default_visual_ or 'sites')
