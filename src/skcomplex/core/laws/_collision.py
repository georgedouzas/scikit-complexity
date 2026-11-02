"""Define collision laws, the binary collision integrals of densities over continuous classes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from sage.all__sagemath_symbolics import SR, function

from ...specs import RequirementSpec
from ..spaces import ContinuousTime, PhaseSpace
from ..types import Target
from ._rate import RateLaw

COLLISION = function('collision', nargs=1)


class CollisionLaw(RateLaw):
    """Rate of a density by the binary collision integral over the two interval class axes of a phase space.

    Args:
        target:
            The symbol of the density.

        cross_section:
            The function `B(speed, cosine)` of arrays of the relative speed and the cosine of the deflection angle.

        factor:
            An expression that scales the collision frequency, such as a function of a density moment, or `None`.

        value:
            The rate, or `None` for the factor times the collision integral of the target.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import CollisionLaw
        >>> f, n = var('f'), var('n')
        >>> CollisionLaw(f, lambda speed, cosine: 1.0, n).check().prepare().value
        n*collision(f)
    """

    def __init__(
        self,
        target: Target,
        cross_section: Callable[..., object],
        factor: object = None,
        *,
        value: object = None,
        label: str = '',
    ) -> None:
        rate = SR(1 if factor is None else factor) * COLLISION(target) if value is None else value
        super().__init__(target, rate, label=label)
        self.cross_section = cross_section
        self.factor = factor

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time on a phase space."""
        return RequirementSpec(time=(ContinuousTime,), space=(PhaseSpace,))
