"""Define the differential operators that laws of fields use, and the flux law of conservation forms."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from sage.all__sagemath_symbolics import SR
from sage.symbolic.function import BuiltinFunction

from ..base import (
    DomainKind,
    Expression,
    LawKind,
    Requires,
    SpaceKind,
    SymbolicVector,
    TimeKind,
    components_of,
    vector,
)

FIELD_BOUNDARIES = ('periodic', 'zero-flux', 'fixed')
DISCRETIZATIONS = ('finite-difference', 'spectral', 'finite-volume', 'finite-element')
LIMITERS = ('minmod', 'van-leer', None)
RIEMANN_SOLVERS = ('rusanov', 'godunov')


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


class _InverseLaplacian(BuiltinFunction):
    """Solution of the Poisson equation for a field expression with zero mean, `inverse_laplacian(u)`."""

    def __init__(self) -> None:
        super().__init__('inverse_laplacian', nargs=1)


laplacian = _Laplacian()
partial = _Partial()
upwind = _Upwind()
inverse_laplacian = _InverseLaplacian()
inverse_laplacian.__doc__ = """Return the solution `psi` of `laplacian(psi) = u` with zero mean on a spectral field.

Examples:
    >>> from skcomplex.core.base import symbol
    >>> from skcomplex.core.laws import inverse_laplacian
    >>> inverse_laplacian(-symbol('omega'))
    inverse_laplacian(-omega)
"""


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
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import gradient
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
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import divergence
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
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import advection
        >>> sorted(str(term) for term in advection(symbol('u'), [1.0, 2.0]).operands())
        ['upwind(u, 0, 1.0)', 'upwind(u, 1, 2.0)']
    """
    return sum(
        (upwind(SR(expression), SR(axis), SR(item)) for axis, item in enumerate(_items(velocity))),
        SR(0),
    )


@dataclass(frozen=True)
class Flux:
    """Physical flux of a conserved state of a finite-volume field, `du/dt + divergence(flux) = 0`.

    The flux is a symbolic expression of the field's states for one dimension, or one expression per dimension. The
    states of a field with fluxes form a system of conservation laws, whose wave speeds come from the eigenvalues of
    the flux Jacobians. Rate laws on the same states add sources.

    Args:
        target:
            The symbol of the conserved state.

        flux:
            The flux, an expression or one expression per dimension.

        limiter:
            The slope limiter of the MUSCL reconstruction, `'minmod'`, `'van-leer'`, or `None` for first order.

        riemann:
            The numerical flux, `'rusanov'` or `'godunov'`, the exact solver for one state.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import Flux
        >>> u = symbol('u')
        >>> Flux(u, u**2 / 2, riemann='godunov').limiter
        'minmod'
    """

    law_kind: ClassVar[LawKind] = LawKind.RATE
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        space={SpaceKind.LATTICE, SpaceKind.MESH},
        domains={DomainKind.REAL},
    )

    target: Expression
    flux: object
    limiter: str | None = 'minmod'
    riemann: str = 'rusanov'

    def __post_init__(self) -> None:
        if self.limiter not in LIMITERS:
            error_msg = f"Unknown limiter {self.limiter!r}. Use 'minmod', 'van-leer', or None."
            raise ValueError(error_msg)
        if self.riemann not in RIEMANN_SOLVERS:
            error_msg = f"Unknown numerical flux {self.riemann!r}. Use 'rusanov' or 'godunov'."
            raise ValueError(error_msg)

    @property
    def fluxes(self) -> list[object]:
        """Flux of every dimension."""
        return list(self.flux) if isinstance(self.flux, list | tuple) else components_of(self.flux)
