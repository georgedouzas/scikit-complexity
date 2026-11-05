"""Define the Swift-Hohenberg free energy, which selects patterns of one wavelength."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from .....core.interactions import GradientInteraction
from .....core.laws.utils import laplacian
from .....specs import ParameterSpec, RoleSpec


class SwiftHohenbergEnergyInteraction(GradientInteraction):
    """Gradient flow `u_t = r u - (1 + laplacian)^2 u + g u^2 - u^3` of a field in the Swift-Hohenberg free energy.

    The free energy is `F = integral -r u^2 / 2 + ((1 + laplacian) u)^2 / 2 - g u^3 / 3 + u^4 / 4`. Above onset,
    `r > 0`, the modes of wavenumber near 1 grow into rolls, or into hexagons for a large enough quadratic term `g`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field.

        state:
            The name of the state of the field.

        r:
            The distance above onset.

        g:
            The strength of the quadratic term, which breaks the symmetry `u -> -u`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.models.physics.common.fields import SwiftHohenbergEnergyInteraction
        >>> energy = SwiftHohenbergEnergyInteraction('energy', target='cells').prepare()
        >>> namespace = SimpleNamespace(target=SimpleNamespace(u=var('u')), r=var('r'), g=0)
        >>> [law.value for law in energy.run(namespace).laws_]
        [-u^3 + r*u - u - 2*laplacian(u) - laplacian(laplacian(u))]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'u',
        r: Annotated[float, ParameterSpec()] = 0.3,
        g: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, target=target, states=state, variational=True)
        self.state = state
        self.r = r
        self.g = g

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the density `-r u^2 / 2 + (u + laplacian(u))^2 / 2 - g u^3 / 3 + u^4 / 4`.

        Args:
            namespace:
                The namespace of the symbols the density can use.

        Returns:
            density:
                The density of the free energy.
        """
        u = getattr(namespace.target, self.state)
        return -namespace.r * u**2 / 2 + (u + laplacian(u)) ** 2 / 2 - namespace.g * u**3 / 3 + u**4 / 4
