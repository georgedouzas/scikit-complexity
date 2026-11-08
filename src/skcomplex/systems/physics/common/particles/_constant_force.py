"""Define constant force interactions.

They push a target with a fixed force.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt

from .....core.components.interactions import RateInteraction
from .....core.specs import ParameterSpec, RoleSpec


class ConstantForceInteraction(RateInteraction):
    """Constant force `F` on a target.

    The force is the second derivative `F / m` of its position of order 2.

    It is the rate `F` with the mass as divisor, and the LAMMPS backend runs it as `fix addforce`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group the force acts on.

        F:
            The force vector. A number applies to every component.

        state:
            The name of the vector state of order 2 that holds the position of the target.

        divisor:
            The name of the state or parameter that holds the mass of the target.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import ConstantForceInteraction
        >>> ConstantForceInteraction('gravity', target='ball', F=[0, -9.81]).F
        [0, -9.81]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        F: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        state: str = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=target, state=state, divisor=divisor)
        self.F = F

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the force `F`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The force, before the division by the mass.
        """
        return namespace.F
