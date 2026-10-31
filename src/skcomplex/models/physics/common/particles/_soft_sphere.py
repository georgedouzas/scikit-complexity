"""Define the soft repulsion between overlapping spheres."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.interactions import PairForceInteraction
from .....specs import ParameterSpec


class SoftSphereInteraction(PairForceInteraction):
    """Soft repulsion between overlapping spheres, `k * (cutoff - r) ** 2 / 2` closer than the cutoff.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The contact distance, the sum of the two radii.

        position:
            The name of the vector state that holds the positions of the members.

        velocity:
            The name of the vector state that holds the velocities of the members.

        mass:
            The name of the state or parameter that holds the masses of the members.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        k:
            The stiffness.

    Examples:
        >>> from skcomplex.models.physics.common.particles import SoftSphereInteraction
        >>> SoftSphereInteraction('contact', first='gas', second='gas', cutoff=0.1, k=50.0).k
        50.0
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        velocity: str = 'v',
        mass: str = 'm',
        period: float | None = None,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            velocity=velocity,
            mass=mass,
            period=period,
            energy=self._get_soft_sphere_energy,
        )
        self.k = k

    @staticmethod
    def _get_soft_sphere_energy(s: SimpleNamespace, r: Expression) -> Expression:
        """Return `k * (cutoff - r) ** 2 / 2`."""
        return s.k * (s.cutoff - r) ** 2 / 2
