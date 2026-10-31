"""Define constant force interactions, which push a target with a fixed force."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt

from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RateLaw
from ._interaction import Interaction


class ConstantForceInteraction(Interaction):
    """Constant force `F` on a target, which changes its velocity by `F / m`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group the force acts on.

        F:
            The force vector. A number applies to every component.

        velocity:
            The name of the vector state that holds the velocity of the target.

        mass:
            The name of the state or parameter that holds the mass of the target.

    Examples:
        >>> from skcomplex.core.interactions import ConstantForceInteraction
        >>> ConstantForceInteraction('gravity', target='ball', F=[0, -9.81]).F
        [0, -9.81]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        F: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        velocity: str = 'v',
        mass: str = 'm',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.F = F
        self.velocity = velocity
        self.mass = mass

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the velocity change `F / m`.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the velocity of the target.
        """
        target = namespace.target
        return [RateLaw(getattr(target, self.velocity), namespace.F / getattr(target, self.mass))]
