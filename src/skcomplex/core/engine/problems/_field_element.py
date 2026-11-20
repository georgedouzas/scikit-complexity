"""Define the element that holds a field, with one value per site of a space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from sage.structure.element import Matrix

from ...components import TensorSpec
from ...components._state_spec import _StateSpec
from ...components.elements import BaseElement, VisualSpec
from ...components.spaces import BaseSpace, CellComplexSpace, FiniteVolumeSpec, LatticeSpace, NetworkSpace
from ...errors import ValidationError
from ._group_element import _GroupElement

FIELD_BOUNDARIES = ('periodic', 'zero-flux', 'fixed')
MATRIX_RANK = 2
DISCRETIZATIONS = ('finite-difference', 'spectral', 'finite-volume', 'finite-element')


class _FieldElement(_GroupElement):
    """An element that holds a field, with one value per site of a space.

    A field is a quantity spread over space, such as the temperature of a plate. Its members sit on the sites of a
    lattice, the nodes of a network, or the nodes of a mesh, and the laws of the element class are written for one site.
    The operators `laplacian`, `gradient`, `divergence`, and `advection` of `skcomplex.core.components.laws.utils` read
    the neighbouring sites, and `discretization` says how they become numbers. Each state takes a boundary condition at
    the edges of a lattice.

    Unlike `GroupElement`, it places one member at every site and gives its states spatial operators and boundary
    conditions. Choose it when a model is a partial differential equation, a law for how a quantity changes in both time
    and space, solved on a grid, a network, or a mesh.

    It appears in many fields. Heat flowing through a metal plate, chemicals that react and diffuse in a gel, a
    pollutant carried along a river, and the vorticity of an ocean gyre are fields.

    Args:
        label:
            The name of the field, such as `'plate'`.

        element:
            The element class of one site, with the states of the field and their laws.

        space:
            The lattice, network, or mesh whose sites hold the field, such as `LatticeSpace((10, 20))`.

        boundary:
            The boundary condition of each state by name, which replaces the edges of the lattice. A state without one
            takes the edges of the space.

            - `'periodic'`: the field wraps around, as on a ring or a torus.
            - `'zero-flux'`: nothing flows through the edge, as at an insulated wall.
            - `('fixed', value)`: the edge holds a value, a number or, on a mesh, a function of the node coordinates.
              For a tensor state the value may be a matrix with the symmetry of the state.

        values:
            The initial value of each state or parameter by name, as an array of the lattice shape, one value per node
            of a network in node order, or one value for all sites, such as `{'u': [9.0, 0.0, 0.0]}`.

        discretization:
            How the operators of space become numbers.

            - `'finite-difference'`: stencils on a lattice and the graph Laplacian on a network.
            - `'spectral'`: products of Fourier modes on a periodic lattice, with the two-thirds rule, which drops the
              highest modes so that their products create no false ones.
            - `'finite-volume'`: averages over cells. A rate `-divergence(flux)` of a state is a conservation law,
              whose cells change by the differences of the flux between them under the scheme of `scheme`.
            - `'finite-element'`: stiffness and mass matrices assembled on a `CellComplexSpace` of points and cells.

        scheme:
            The limiter, numerical flux, and wave speed of a finite-volume field, or `None` for the minmod limiter and
            the Rusanov flux. Fields of other discretizations take none.

        symmetry:
            The parity of each state by name along every axis of a periodic lattice, `'odd'`, `'even'`, or `None`, which
            the rates keep exactly. A lattice twice as long as a domain along an axis, with states odd or even about its
            middle, stands for walls there, such as the free-slip walls of convection. A vector or tensor state takes
            one such sequence for its first component, or one sequence per component.

    Attributes:
        default_visual_:
            The `visual` of the member class, or else an image on a lattice, a coloured mesh on a mesh, and the sites
            of a network.

        size_:
            The number of sites of the space.

        values_:
            The values with one row per site, or one value shared by every site.
    """

    def __init__(
        self,
        label: str,
        *,
        element: type[BaseElement],
        space: LatticeSpace | NetworkSpace,
        boundary: Mapping[str, object] | None = None,
        values: Mapping[str, object] | None = None,
        discretization: str = 'finite-difference',
        scheme: FiniteVolumeSpec | None = None,
        symmetry: Mapping[str, Sequence[object]] | None = None,
    ) -> None:
        super().__init__(label, element=element, values=values, space=space, on='sites')
        self.boundary = boundary
        self.discretization = discretization
        self.scheme = scheme
        self.symmetry = symmetry

    def _get_scheme(self) -> FiniteVolumeSpec:
        """Return the scheme of a finite-volume field, the default one without a scheme."""
        return FiniteVolumeSpec() if self.scheme is None else self.scheme

    def _get_tensor_of(self, state: str) -> TensorSpec | None:
        """Return the independent components of a tensor state of the element.

        The result is `None` for any other state.
        """
        spec = self._get_member_specs().get(state)
        if not isinstance(spec, _StateSpec) or spec.tensor is None:
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
        mesh = isinstance(self.space, CellComplexSpace) and self.space._is_mesh()
        edges = 'periodic'
        if mesh:
            edges = 'reflecting'
        elif isinstance(self.space, LatticeSpace):
            edges = self.space._get_edges()
        condition = (self.boundary or {}).get(state, default.get(edges, 'periodic'))
        if isinstance(condition, tuple | list):
            tensor = self._get_tensor_of(state)
            if tensor is not None and (isinstance(condition[1], Matrix) or np.ndim(condition[1]) == MATRIX_RANK):
                return str(condition[0]), float(tensor.get_components_of(condition[1])[column])
            return str(condition[0]), condition[1] if callable(condition[1]) else float(condition[1])
        return str(condition), 0.0

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `size_`, `values_`, and `default_visual_`.

        It first raises an error for an unknown discretization or boundary condition.

        The default visual is the `visual` of the member class, or else an image on a lattice and a colored mesh on a
        mesh.

        Args:
            context:
                The space the field inherits from its system, unused since a field has a space of its own.

        Raises:
            ValidationError:
                The discretization or a boundary condition is unknown, or the matrix of a tensor state does not have
                its symmetry.
        """
        super()._prepare(context)
        if self.discretization not in DISCRETIZATIONS:
            error_msg = f'Unknown discretization {self.discretization!r} of {self.label}. Use one of {DISCRETIZATIONS}.'
            raise ValidationError(error_msg)
        if self.scheme is not None:
            if not isinstance(self.scheme, FiniteVolumeSpec) or self.discretization != 'finite-volume':
                error_msg = (
                    f'The scheme of {self.label} is {self.scheme!r}. Give a `FiniteVolumeSpec` to a field with '
                    "discretization='finite-volume'."
                )
                raise ValidationError(error_msg)
            self.scheme._check()
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
        if self._get_member()._get_visual() != VisualSpec():
            return
        if isinstance(self.space, LatticeSpace):
            self.default_visual_ = VisualSpec('image')
        elif isinstance(self.space, CellComplexSpace) and self.space._is_mesh():
            self.default_visual_ = VisualSpec('mesh')
