"""Define the harmonic potential that pulls a particle towards a center."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....specs import ParameterSpec, RoleSpec
from ._potential_energy import PotentialEnergyInteraction


class HarmonicPotentialInteraction(PotentialEnergyInteraction):
    """Harmonic potential that pulls a particle towards a center.

    The energy is `k * |x - center|**2 / 2`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The stiffness.

        center:
            The center. A number applies to every component.

        states:
            The name of the vector state that holds the position of the particle.

        velocity:
            The name of the vector state that holds the velocity of the particle.

        divisor:
            The name of the state or parameter that holds the mass of the particle.

    Examples:
        >>> from skcomplex.models.physics.common.particles import HarmonicPotentialInteraction
        >>> HarmonicPotentialInteraction('trap', target='ball', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        center: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        states: str = 'x',
        velocity: str = 'v',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=target, states=states, velocity=velocity, divisor=divisor)
        self.k = k
        self.center = center

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return `k * |x - center|**2 / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        displacement = getattr(s.target, self._get_names()[0]) - s.center
        return s.k * displacement.dot_product(displacement) / 2
