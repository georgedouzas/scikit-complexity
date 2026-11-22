"""Define the spec of how a finite-volume field moves its values between cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError

LIMITERS = ('minmod', 'van-leer', None)
RIEMANN_SOLVERS = ('rusanov', 'godunov')


@dataclass(frozen=True)
class FiniteVolumeSpec(BaseSpec):
    """A spec of how a finite-volume field moves its values between cells.

    On a lattice with `discretization='finite-volume'`, the law `u' = -divergence(flux)` moves amounts of `u` between
    neighbouring cells, so the total stays the same. The spec says how: `limiter` rebuilds the value inside each cell
    from its neighbours, and `riemann` computes the amount that crosses each wall, which keeps a sudden jump sharp, such
    as `LatticeSpace(100, discretization='finite-volume', scheme=FiniteVolumeSpec(limiter='van-leer'))`. Cars pile up
    into jams on a road, a gas forms shock waves, and the trees of a forest move along their growth curve.

    Read more in the user guide, under Spaces.

    Args:
        limiter:
            The slope limiter of the MUSCL reconstruction, `'minmod'`, `'van-leer'`, or `None` for first order.

        riemann:
            The numerical flux, `'rusanov'` or `'godunov'`, the exact solver for one state.

        speed:
            A function `f(t, values)` of the mapping from the name of every conserved state and parameter of the fluxes
            to its array over the cells, which returns the wave speed of the Rusanov flux, or one function per
            dimension. `None` takes the speeds from the eigenvalues of the flux Jacobians.

    Raises:
        ValidationError:
            The limiter or the numerical flux is unknown, or a speed comes with the Godunov flux, raised when its field
            checks it.

    See Also:
        LatticeSpace: Takes this spec as the `scheme` of a finite-volume lattice.

    Examples:
        >>> from skcomplex.core.components.spaces import FiniteVolumeSpec, LatticeSpace
        >>> scheme = FiniteVolumeSpec(limiter='van-leer', riemann='godunov')
        >>> LatticeSpace(100, discretization='finite-volume', scheme=scheme).prepare().scheme.limiter
        'van-leer'
    """

    limiter: str | None = 'minmod'
    riemann: str = 'rusanov'
    speed: Callable[..., object] | tuple[Callable[..., object], ...] | None = None

    def _check(self) -> None:
        """Raise an error for an unknown limiter or flux, or a speed with the Godunov flux.

        Raises:
            ValidationError:
                The limiter or the numerical flux is unknown, or a speed comes with the Godunov flux.
        """
        if self.limiter not in LIMITERS:
            error_msg = f"Unknown limiter {self.limiter!r}. Use 'minmod', 'van-leer', or None."
            raise ValidationError(error_msg)
        if self.riemann not in RIEMANN_SOLVERS:
            error_msg = f"Unknown numerical flux {self.riemann!r}. Use 'rusanov' or 'godunov'."
            raise ValidationError(error_msg)
        if self.speed is not None and self.riemann != 'rusanov':
            error_msg = (
                f'The {self.riemann} flux takes no `speed`, because it is exact and symbolic only. '
                "Use riemann='rusanov'."
            )
            raise ValidationError(error_msg)
