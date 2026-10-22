"""Define fields, groups of elements on the sites of a lattice, a network, or a mesh."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ..base import (
    ComponentKind,
    VisualSpec,
)
from ..laws._fields import FIELD_BOUNDARIES
from ..space._lattices import LatticeSpace
from ..space._meshes import Mesh
from ..space._spaces import NetworkSpace
from ._components import Element, Group


class Field(Group):
    """Group of members of one element type at the sites of a lattice or the nodes of a network.

    The states of the members are fields, with one value per site, and the laws of the element type are written for
    one site, with `laplacian`, `gradient`, `divergence`, and `advection` reading the neighbouring sites. Every state
    takes a boundary condition at the edges of a lattice: `'periodic'`, `'zero-flux'`, or `('fixed', value)`, where
    the value can be a function of the node coordinates on a mesh.

    The discretization turns the operators into numbers. `'finite-difference'` uses stencils on lattices and the graph
    Laplacian on networks. `'spectral'` multiplies Fourier modes on periodic lattices, with the two-thirds rule
    removing aliasing. `'finite-volume'` updates cell averages by the differences of the `Flux` laws of the states.
    `'finite-element'` assembles stiffness and mass matrices on a `Mesh`.

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
        >>> from skcomplex.core.components import Element, Field
        >>> from skcomplex.core.laws import Rate, laplacian
        >>> from skcomplex.core.space import LatticeSpace
        >>> Heat = Element.define(
        ...     'Heat',
        ...     states={'u': 0.0},
        ...     parameters={'D': 1.0},
        ...     laws=lambda s: [Rate(s.u, s.D * laplacian(s.u))],
        ... )
        >>> Field('plate', Heat, LatticeSpace((10, 20)), boundary={'u': 'zero-flux'}).size
        200
    """

    component_kind: ClassVar[ComponentKind] = ComponentKind.SITE_GROUP

    def __init__(
        self,
        label: str,
        element: type[Element],
        space: LatticeSpace | NetworkSpace | Mesh,
        *,
        boundary: Mapping[str, Any] | None = None,
        values: Mapping[str, Any] | None = None,
        visual: VisualSpec | None = None,
        discretization: str = 'finite-difference',
        symmetry: Mapping[str, Sequence[str | None]] | None = None,
    ) -> None:
        self.label = label
        self.element = element
        self.space = space
        self.boundary = boundary
        self.values = values
        self.visual = visual
        self.discretization = discretization
        self.symmetry = symmetry
        self.size = space.graph.number_of_nodes() if isinstance(space, NetworkSpace) else space.size
        self.variable = False
        self.on = 'sites'

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
        edges = 'reflecting' if isinstance(space, Mesh) else getattr(space, 'boundary', 'periodic')
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
