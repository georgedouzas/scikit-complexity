"""Define flux laws, the fluxes of the conserved states of finite-volume fields."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime, LatticeSpace, MeshSpace
from ..types import Target
from ._base import BaseLaw

LIMITERS = ('minmod', 'van-leer', None)
RIEMANN_SOLVERS = ('rusanov', 'godunov')


class FluxLaw(BaseLaw):
    """Flux of a conserved state of a finite-volume field, `du/dt + divergence(flux) = 0`.

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

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import FluxLaw
        >>> u = var('u')
        >>> FluxLaw(u, u**2 / 2, riemann='godunov').check().prepare().run({'u': 2.0}).value_
        2.0
    """

    def __init__(
        self,
        target: Target,
        flux: object,
        *,
        limiter: str | None = 'minmod',
        riemann: str = 'rusanov',
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.flux = flux
        self.limiter = limiter
        self.riemann = riemann

    def _get_targets(self) -> tuple[object, ...]:
        """Return the conserved state."""
        return (self.target,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time on a lattice or a mesh."""
        return RequirementSpec(time=(ContinuousTime,), space=(LatticeSpace, MeshSpace))

    def _check(self) -> None:
        """Raise an error for an unknown limiter or numerical flux."""
        self._check_array('flux')
        if self.limiter not in LIMITERS:
            error_msg = f"Unknown limiter {self.limiter!r}. Use 'minmod', 'van-leer', or None."
            raise ValidationError(error_msg)
        if self.riemann not in RIEMANN_SOLVERS:
            error_msg = f"Unknown numerical flux {self.riemann!r}. Use 'rusanov' or 'godunov'."
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_` to the flux at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.flux, inputs)
