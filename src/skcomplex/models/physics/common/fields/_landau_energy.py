"""Define the Landau free energy of a scalar order parameter, a double well with a cost of gradients."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.interactions import GradientInteraction
from .....core.laws.utils import laplacian
from .....specs import ParameterSpec, RoleSpec


class LandauEnergyInteraction(GradientInteraction):
    """Gradient flow of a field `phi` in the Landau free energy of a double well with a cost of gradients.

    The free energy is `F = integral -phi^2 / 2 + phi^4 / 4 + kappa |grad phi|^2 / 2`. Its functional derivative is
    the chemical potential `mu = -phi + phi^3 - kappa laplacian(phi)`. A non-conserved field relaxes as `phi_t = -mu`,
    the Allen-Cahn equation of model A, and a conserved field as `phi_t = laplacian(mu)`, the Cahn-Hilliard equation of
    model B, whose total stays constant.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field.

        state:
            The name of the state of the field that holds the order parameter.

        kappa:
            The cost of gradients, the square of the width of the interfaces.

        conserved:
            Whether the order parameter is conserved.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.models.physics.common.fields import LandauEnergyInteraction
        >>> energy = LandauEnergyInteraction('energy', target='mixture').check().prepare()
        >>> namespace = SimpleNamespace(target=SimpleNamespace(phi=var('phi')), kappa=var('kappa'))
        >>> [law.value for law in energy.run(namespace).laws_]
        [-phi^3 + kappa*laplacian(phi) + phi]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'phi',
        kappa: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        conserved: bool = False,
    ) -> None:
        super().__init__(label, target=target, states=state, variational=True, conserved=conserved)
        self.state = state
        self.kappa = kappa

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the density `-phi^2 / 2 + phi^4 / 4 - kappa phi laplacian(phi) / 2`, the gradient term by parts.

        Args:
            namespace:
                The namespace of the symbols the density can use.

        Returns:
            density:
                The density of the free energy.
        """
        phi = getattr(namespace.target, self.state)
        return -(phi**2) / 2 + phi**4 / 4 - namespace.kappa * phi * laplacian(phi) / 2
