"""Define fields over lattices and networks and the differential operators that act on them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Any

from sage.all__sagemath_symbolics import SR
from sage.symbolic.function import BuiltinFunction

from ..core import Expression, SymbolicVector, VisualSpec, components_of, vector
from ._components import Element, Group
from ._lattices import LatticeSpace
from ._spaces import NetworkSpace

FIELD_BOUNDARIES = ('periodic', 'zero-flux', 'fixed')


class _Laplacian(BuiltinFunction):
    """Laplacian of a field expression, `laplacian(u)`."""

    def __init__(self) -> None:
        super().__init__('laplacian', nargs=1, latex_name=r'\nabla^2')


class _Partial(BuiltinFunction):
    """Derivative of a field expression along one dimension, `partial(u, axis)`."""

    def __init__(self) -> None:
        super().__init__('partial', nargs=2)


class _Upwind(BuiltinFunction):
    """Upwind derivative of a field expression along one dimension for a velocity, `upwind(u, axis, velocity)`."""

    def __init__(self) -> None:
        super().__init__('upwind', nargs=3)


laplacian = _Laplacian()
partial = _Partial()
upwind = _Upwind()


def _items(values: Sequence[Any] | SymbolicVector) -> list[Any]:
    """Return the components of a vector given as a sequence or a symbolic vector."""
    return list(values) if isinstance(values, list | tuple) else components_of(values)


def gradient(expression: object, n_dim: int = 2) -> SymbolicVector:
    """Return the gradient of a field expression, central differences along every dimension of a lattice.

    Args:
        expression:
            An expression of the states and parameters of a field.

        n_dim:
            The number of dimensions of the lattice.

    Returns:
        gradient:
            One derivative per dimension.

    Examples:
        >>> from skcomplex.core import symbol
        >>> from skcomplex.model import gradient
        >>> gradient(symbol('u'))
        (partial(u, 0), partial(u, 1))
    """
    return vector(partial(SR(expression), SR(axis)) for axis in range(n_dim))


def divergence(components: Sequence[Any] | SymbolicVector) -> Expression:
    """Return the divergence of a vector of field expressions, the sum of the derivative of each along its dimension.

    Args:
        components:
            One expression per dimension, such as a flux.

    Returns:
        divergence:
            The divergence.

    Examples:
        >>> from skcomplex.core import symbol
        >>> from skcomplex.model import divergence
        >>> u = symbol('u')
        >>> sorted(str(term) for term in divergence([u, 2 * u]).operands())
        ['partial(2*u, 1)', 'partial(u, 0)']
    """
    return sum((partial(SR(item), SR(axis)) for axis, item in enumerate(_items(components))), SR(0))


def advection(expression: object, velocity: Sequence[Any] | SymbolicVector) -> Expression:
    """Return the advection of a field expression by a velocity, its rate of transport, with upwind differences.

    Args:
        expression:
            An expression of the states and parameters of a field.

        velocity:
            One velocity component per dimension, numbers or expressions.

    Returns:
        advection:
            The advection term, to subtract from the rate of the field.

    Examples:
        >>> from skcomplex.core import symbol
        >>> from skcomplex.model import advection
        >>> sorted(str(term) for term in advection(symbol('u'), [1.0, 2.0]).operands())
        ['upwind(u, 0, 1.0)', 'upwind(u, 1, 2.0)']
    """
    return sum(
        (upwind(SR(expression), SR(axis), SR(item)) for axis, item in enumerate(_items(velocity))),
        SR(0),
    )


class Field(Group):
    """Group of members of one element type at the sites of a lattice or the nodes of a network.

    The states of the members are fields, with one value per site, and the laws of the element type are written for
    one site, with `laplacian`, `gradient`, `divergence`, and `advection` reading the neighbouring sites. Every state
    takes a boundary condition at the edges of a lattice: `'periodic'`, `'zero-flux'`, or `('fixed', value)`.

    Args:
        label:
            The label of the field.

        element:
            The element type of the members.

        space:
            The lattice or the network.

        boundary:
            A mapping from state name to its boundary condition, which overrides the edges of the lattice.

        values:
            Arrays of the lattice shape, or with one value per network node in node order, or shared values.

        visual:
            The visual of the field, or `None` for an image of every state on a lattice.

    Examples:
        >>> from skcomplex.model import Element, Field, LatticeSpace, Rate, laplacian
        >>> Heat = Element.define(
        ...     'Heat',
        ...     states={'u': 0.0},
        ...     parameters={'D': 1.0},
        ...     laws=lambda s: [Rate(s.u, s.D * laplacian(s.u))],
        ... )
        >>> Field('plate', Heat, LatticeSpace((10, 20)), boundary={'u': 'zero-flux'}).size
        200
    """

    def __init__(
        self,
        label: str,
        element: type[Element],
        space: LatticeSpace | NetworkSpace,
        *,
        boundary: Mapping[str, Any] | None = None,
        values: Mapping[str, Any] | None = None,
        visual: VisualSpec | None = None,
    ) -> None:
        self.label = label
        self.element = element
        self.space = space
        self.boundary = boundary
        self.values = values
        self.visual = visual
        self.size = space.size if isinstance(space, LatticeSpace) else space.graph.number_of_nodes()

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label, element type, and size."""
        return f'Field({self.label!r}, element={self.element.__name__}, size={self.size})'

    def boundary_of(self, state: str) -> tuple[str, float]:
        """Return the boundary condition of a state and its fixed value.

        Args:
            state:
                The name of the state.

        Returns:
            boundary:
                `'periodic'`, `'zero-flux'`, or `'fixed'`, and the value read outside fixed edges.

        Raises:
            ValueError:
                The boundary condition is not one of the three.
        """
        space = self.space
        default = {'periodic': 'periodic', 'reflecting': 'zero-flux', 'fixed': ('fixed', getattr(space, 'fill', 0.0))}
        condition = (self.boundary or {}).get(state, default.get(getattr(space, 'boundary', 'periodic'), 'periodic'))
        kind, value = (condition[0], float(condition[1])) if isinstance(condition, tuple | list) else (condition, 0.0)
        if kind not in FIELD_BOUNDARIES:
            error_msg = (
                f"Unknown boundary {condition!r} of {self.label}.{state}. Use 'periodic', 'zero-flux', or "
                "('fixed', value)."
            )
            raise ValueError(error_msg)
        return kind, value
