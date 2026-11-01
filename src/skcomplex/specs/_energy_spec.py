"""Define the energy spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class EnergySpec(BaseSpec):
    """Spec of the local energy of a lattice site, every term of the total energy that involves the value of the site.

    The energy change of a move is the difference of the local energy at the proposed and the current value. Neighbour
    values are read with `neighbors(..., reduce='each')` in neighbourhood order, or summed with
    `neighbors(..., reduce='sum')`.

    Args:
        expression:
            The local energy, a symbolic expression of the states of the site, its neighbours, and parameters.

        over:
            The neighbourhood that `neighbors` reads in the expression.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.neighborhoods import VonNeumannNeighborhood
        >>> from skcomplex.specs import EnergySpec
        >>> spin, J = var('spin J')
        >>> EnergySpec(-J * spin, VonNeumannNeighborhood()).check().expression
        -J*spin
    """

    expression: object
    over: object

    def _check(self) -> None:
        """Accept the spec, which has nothing to check."""
