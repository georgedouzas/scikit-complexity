"""Define the objective spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec


@dataclass(frozen=True)
class ObjectiveSpec(BaseSpec):
    """A spec of the cost of one site of a lattice, which a Monte Carlo move changes.

    It holds every term of the total cost that reads the value of the site, such as `ObjectiveSpec(-J * s.spin *
    neighbors(s.spin), around)` for a magnet, where neighbours that point the same way lower the energy. The change of
    a move is the cost at the proposed value less the cost at the current value. The expression reads the neighbours
    with `neighbors(..., reduce='each')` in the order of the neighbourhood, or their sum with `neighbors(...,
    reduce='sum')`. It gives the energy of a spin in a magnet, the discomfort of a resident among unlike neighbours,
    and the cost of a letter in a sequence of DNA.

    Read more in the user guide, under Laws.

    Args:
        expression:
            The local objective, a symbolic expression of the states of the site, its neighbours, and parameters.

        over:
            The neighbourhood that `neighbors` reads in the expression.

    See Also:
        MoveSpec: The change of a site that a sweep proposes.
        ObjectiveLaw: Gives the total cost that the compiler splits into sites.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.spaces import RadiusNeighborhood
        >>> from skcomplex.core.components.laws import ObjectiveSpec
        >>> spin, J = var('spin J')
        >>> ObjectiveSpec(-J * spin, RadiusNeighborhood(1, metric='manhattan')).expression
        -J*spin
    """

    expression: object
    over: object

    def _check(self) -> None:
        """Accept the spec, which has nothing to check."""
