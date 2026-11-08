"""Define the objective spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ObjectiveSpec(BaseSpec):
    """Spec of the local objective of a lattice site.

    It is every term of the total objective that involves the site's value. The change of a move is the difference of
    the local objective at the proposed and the current value. Neighbour values are read with `neighbors(...,
    reduce='each')` in neighbourhood order, or summed with `neighbors(..., reduce='sum')`.

    Args:
        expression:
            The local objective, a symbolic expression of the states of the site, its neighbours, and parameters.

        over:
            The neighbourhood that `neighbors` reads in the expression.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.spaces import VonNeumannNeighborhood
        >>> from skcomplex.core.specs import ObjectiveSpec
        >>> spin, J = var('spin J')
        >>> ObjectiveSpec(-J * spin, VonNeumannNeighborhood()).expression
        -J*spin
    """

    expression: object
    over: object

    def _check(self) -> None:
        """Accept the spec.

        It has nothing to check.
        """
