"""Define fields, groups of elements on the sites of a lattice, a network, or a mesh."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ...core.base import (
    ComponentKind,
    VisualSpec,
)
from ...core.components import ElementComponent, GroupComponent
from ...core.space import LatticeSpace, NetworkSpace
from ._laws import FIELD_BOUNDARIES
from ._spaces import MeshSpace


class FieldComponent(GroupComponent):
    """Group of members of one element type at the sites of a lattice or the nodes of a network.

    The states of the members are fields, with one value per site, and the laws of the element type are written for
    one site, with `laplacian`, `gradient`, `divergence`, and `advection` reading the neighbouring sites. Every state
    takes a boundary condition at the edges of a lattice: `'periodic'`, `'zero-flux'`, or `('fixed', value)`, where
    the value can be a function of the node coordinates on a mesh.

    The discretization turns the operators into numbers. `'finite-difference'` uses stencils on lattices and the graph
    Laplacian on networks. `'spectral'` multiplies Fourier modes on periodic lattices, with the two-thirds rule
    removing aliasing. `'finite-volume'` updates cell averages by the differences of the `FluxLaw`s of the states.
    `'finite-element'` assembles stiffness and mass matrices on a `MeshSpace`.

    Args:
        label:
            The label of the field.

        element:
            The element type of the members.

        space:
            The lattice, the network, or the mesh.

        boundary:
            A mapping from state name to its boundary condition, which overrides the edges of the lattice.

        values:
            Arrays of the lattice shape, or with one value per network node in node order, or shared values.

        visual:
            The visual of the field, or `None` for an image of every state on a lattice and a colored mesh on a mesh.

        discretization:
            `'finite-difference'`, `'spectral'`, `'finite-volume'`, or `'finite-element'`.

        symmetry:
            A mapping from state name to its parity along every axis of a periodic lattice, `'odd'`, `'even'`, or
            `None`, which the rates keep exactly. A lattice twice as long as a domain along an axis, with states odd
            or even about its middle, represents walls there, such as the free-slip walls of convection.

    Examples:
        >>> from skcomplex.core.components import ElementComponent
        >>> from skcomplex.formalisms.fields import FieldComponent, laplacian
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.core.space import LatticeSpace
        >>> HeatComponent = ElementComponent.define(
        ...     'HeatComponent',
        ...     states={'u': 0.0},
        ...     parameters={'D': 1.0},
        ...     laws=lambda s: [RateLaw(s.u, s.D * laplacian(s.u))],
        ... )
        >>> FieldComponent('plate', HeatComponent, LatticeSpace((10, 20)), boundary={'u': 'zero-flux'}).size
        200
        >>> FieldComponent('plate', HeatComponent, LatticeSpace((4, 4))).prepare().default_visual_.kind
        'image'
    """

    component_kind: ClassVar[ComponentKind] = ComponentKind.SITE_GROUP

    def __init__(
        self,
        label: str,
        element: type[ElementComponent],
        space: LatticeSpace | NetworkSpace | MeshSpace,
        *,
        boundary: Mapping[str, Any] | None = None,
        values: Mapping[str, Any] | None = None,
        visual: VisualSpec | None = None,
        discretization: str = 'finite-difference',
        symmetry: Mapping[str, Sequence[str | None]] | None = None,
    ) -> None:
        size = space.prepare().size_
        super().__init__(label, element=element, size=size, values=values, visual=visual, space=space, on='sites')
        self.boundary = boundary
        self.discretization = discretization
        self.symmetry = symmetry

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label, element type, and size."""
        return f'Field({self.label!r}, element={self.element.__name__}, size={self.size})'

    def boundary_of(self, state: str) -> tuple[str, Any]:
        """Return the boundary condition of a state and its fixed value.

        Args:
            state:
                The name of the state.

        Returns:
            boundary:
                `'periodic'`, `'zero-flux'`, or `'fixed'`, and the value read outside fixed edges, a number or a
                function of the node coordinates.

        Raises:
            ValueError:
                The boundary condition is not one of the three.
        """
        space = self.space
        default = {'periodic': 'periodic', 'reflecting': 'zero-flux', 'fixed': ('fixed', getattr(space, 'fill', 0.0))}
        edges = 'reflecting' if isinstance(space, MeshSpace) else getattr(space, 'boundary', 'periodic')
        condition = (self.boundary or {}).get(state, default.get(edges, 'periodic'))
        if isinstance(condition, tuple | list):
            kind, value = condition[0], condition[1] if callable(condition[1]) else float(condition[1])
        else:
            kind, value = condition, 0.0
        if kind not in FIELD_BOUNDARIES:
            error_msg = (
                f"Unknown boundary {condition!r} of {self.label}.{state}. Use 'periodic', 'zero-flux', or "
                "('fixed', value)."
            )
            raise ValueError(error_msg)
        return kind, value

    def _prepare(self) -> None:
        """Set `declarations_` and `default_visual_`, the visual of the field when it sets none.

        The default visual is an image on a lattice and a colored mesh on a mesh. On a network the field takes the
        visual of its element type, or the sites of the network when the type declares none.
        """
        super()._prepare()
        declared: VisualSpec = self.element.visual
        if isinstance(self.space, LatticeSpace):
            self.default_visual_ = VisualSpec('image')
        elif isinstance(self.space, MeshSpace):
            self.default_visual_ = VisualSpec('mesh')
        elif isinstance(self.space, NetworkSpace) and declared == VisualSpec():
            self.default_visual_ = VisualSpec('sites')
        else:
            self.default_visual_ = declared
