"""Define fields.

A field is a group of elements on the sites of a lattice, the nodes of a network, or the nodes of a mesh.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from sage.structure.element import Matrix

from ...errors import ValidationError
from ...specs import StateSpec, TensorSpec, VisualSpec
from ..spaces import BaseSpace, LatticeSpace, MeshSpace, NetworkSpace
from ._base import BaseElement
from ._lattice import LatticeElement

FIELD_BOUNDARIES = ('periodic', 'zero-flux', 'fixed')
MATRIX_RANK = 2
DISCRETIZATIONS = ('finite-difference', 'spectral', 'finite-volume', 'finite-element')


class FieldElement(LatticeElement):
    """Group of members of an element type at the sites of a lattice, network, or mesh.

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
            A mapping from state name to its boundary condition, which overrides the edges of the lattice. The value of
            `('fixed', value)` for a tensor state may be a matrix of the symmetry of the state.

        values:
            Arrays of the lattice shape, or with one value per network node in node order, or shared values.

        visual:
            How the field is drawn, or `None` for `default_visual_`.

        discretization:
            `'finite-difference'`, `'spectral'`, `'finite-volume'`, or `'finite-element'`.

        symmetry:
            A mapping from state name to its parity along every axis of a periodic lattice, `'odd'`, `'even'`, or
            `None`, which the rates keep exactly. A lattice twice as long as a domain along an axis, with states odd
            or even about its middle, represents walls there, such as the free-slip walls of convection. A vector or a
            tensor state takes one such sequence for its first component, or one sequence per component.

    Attributes:
        default_visual_:
            An image on a lattice, a colored mesh on a mesh, and on a network the visual of a member, or the sites of
            the network when the member sets none.

        size_:
            The number of sites of the space.

        values_:
            The values with one row per site, or one value shared by every site.

    Examples:
        >>> from typing import Annotated
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.elements import Element, FieldElement
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.components.laws.utils import laplacian
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec
        >>> class HeatElement(Element):
        ...     def __init__(
        ...         self, label, u: Annotated[float, StateSpec()] = 0.0, D: Annotated[float, ParameterSpec()] = 1.0
        ...     ):
        ...         super().__init__(label)
        ...         self.u = u
        ...         self.D = D
        ...
        ...     def _get_laws(self, namespace):
        ...         return [RateLaw(namespace.u, namespace.D * laplacian(namespace.u))]
        >>> space = LatticeSpace((10, 20))
        >>> plate = FieldElement('plate', element=HeatElement, space=space, boundary={'u': 'zero-flux'})
        >>> plate.prepare().size_, plate.default_visual_.kind
        (200, 'image')
    """

    def __init__(
        self,
        label: str,
        *,
        element: type[BaseElement],
        space: LatticeSpace | NetworkSpace | MeshSpace,
        boundary: Mapping[str, object] | None = None,
        values: Mapping[str, object] | None = None,
        visual: VisualSpec | None = None,
        discretization: str = 'finite-difference',
        symmetry: Mapping[str, Sequence[object]] | None = None,
    ) -> None:
        super().__init__(label, element=element, space=space, values=values, visual=visual)
        self.boundary = boundary
        self.discretization = discretization
        self.symmetry = symmetry

    def _tensor_of(self, state: str) -> TensorSpec | None:
        """Return the independent components of a tensor state of the element.

        The result is `None` for any other state.
        """
        spec = self._get_member_specs().get(state)
        if not isinstance(spec, StateSpec) or spec.tensor is None:
            return None
        tensor = TensorSpec(spec.tensor, int(cast(BaseSpace, self.space).prepare().n_dim_))
        tensor._check()
        return tensor

    def _get_boundary(self, state: str, column: int = 0) -> tuple[str, object]:
        """Return the boundary condition of a state and the value read outside fixed edges.

        A state without a boundary condition of its own takes the edges of the space: periodic edges stay periodic,
        reflecting edges and the edges of a mesh have zero flux, and fixed edges read the fill value of the lattice.

        Args:
            state:
                The name of the state.

            column:
                The independent component of a tensor state, whose fixed value a matrix gives.

        Returns:
            boundary:
                `'periodic'`, `'zero-flux'`, or `'fixed'`, and the value read outside fixed edges, a number or a
                function of the node coordinates.
        """
        default = {
            'periodic': 'periodic',
            'reflecting': 'zero-flux',
            'fixed': ('fixed', getattr(self.space, 'fill', 0.0)),
        }
        edges = 'reflecting' if isinstance(self.space, MeshSpace) else getattr(self.space, 'boundary', 'periodic')
        condition = (self.boundary or {}).get(state, default.get(edges, 'periodic'))
        if isinstance(condition, tuple | list):
            tensor = self._tensor_of(state)
            if tensor is not None and (isinstance(condition[1], Matrix) or np.ndim(condition[1]) == MATRIX_RANK):
                return str(condition[0]), float(tensor.components_of(condition[1])[column])
            return str(condition[0]), condition[1] if callable(condition[1]) else float(condition[1])
        return str(condition), 0.0

    def _check(self) -> None:
        """Raise an error for an unknown discretization or boundary condition.

        Raises:
            ValidationError:
                The discretization or a boundary condition is unknown, or the matrix of a tensor state does not have
                its symmetry.
        """
        super()._check()
        if self.discretization not in DISCRETIZATIONS:
            error_msg = f'Unknown discretization {self.discretization!r} of {self.label}. Use one of {DISCRETIZATIONS}.'
            raise ValidationError(error_msg)
        for state in self.boundary or {}:
            try:
                kind, _ = self._get_boundary(state)
            except ValidationError as error:
                error_msg = f'The boundary of {self.label}.{state} is not valid: {error}'
                raise ValidationError(error_msg) from error
            if kind not in FIELD_BOUNDARIES:
                error_msg = (
                    f"Unknown boundary {kind!r} of {self.label}.{state}. Use 'periodic', 'zero-flux', or "
                    "('fixed', value)."
                )
                raise ValidationError(error_msg)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `size_`, `values_`, and `default_visual_`.

        The default visual is an image on a lattice and a colored mesh on a mesh.

        Args:
            context:
                The space the field inherits from its system, unused since a field has a space of its own.
        """
        super()._prepare(context)
        if isinstance(self.space, LatticeSpace):
            self.default_visual_ = VisualSpec('image')
        elif isinstance(self.space, MeshSpace):
            self.default_visual_ = VisualSpec('mesh')
