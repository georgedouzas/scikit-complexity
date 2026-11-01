"""Random optical speckle field of Gaussian hotspots, which drags particles up its gradient at `u grad(I)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, exp, oo, vector
from sage.sets.real_set import RealSet

from ....core.interactions import GradientInteraction
from ....specs import ParameterSpec, RoleSpec


class SpeckleFieldInteraction(GradientInteraction):
    """Random optical speckle field of Gaussian hotspots, which drags particles up its gradient at `u grad(I)`.

    The potential is `-u I(x)`, with the intensity `I(x) = sum_i I_i exp(-|x - c_i|^2 / d_i^2)` of hotspots whose
    centers, intensities relative to `strength`, and decay lengths are fixed when the field is created.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        strength:
            The product `u I0` of the polarizability constant and the reference intensity.

        centers:
            The centers of the hotspots.

        intensities:
            The intensities of the hotspots in units of the reference intensity.

        lengths:
            The decay lengths of the hotspots.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import SpeckleFieldInteraction
        >>> SpeckleFieldInteraction(
        ...     'speckle', target='bead', centers=((0.0, 0.0),), intensities=(1.0,), lengths=(1.0,)
        ... ).strength
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        centers: tuple[tuple[float, float], ...] = (),
        intensities: tuple[float, ...] = (),
        lengths: tuple[float, ...] = (),
    ) -> None:
        super().__init__(label, target=target, states='x')
        self.strength = strength
        self.centers = centers
        self.intensities = intensities
        self.lengths = lengths

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the potential `-u I(x)` of the speckle field.

        Args:
            namespace:
                The namespace of the symbols the potential can use.

        Returns:
            potential:
                The potential.
        """
        position = namespace.target.x
        intensity = SR(0)
        for center, height, length in zip(self.centers, self.intensities, self.lengths, strict=True):
            offset = position - vector(SR, center)
            intensity += height * exp(-offset.dot_product(offset) / length**2)
        return -namespace.strength * intensity
