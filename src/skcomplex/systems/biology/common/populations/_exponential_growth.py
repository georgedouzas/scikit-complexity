"""Define exponential growth, a population that grows at a constant per capita rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from .....core.specs import ParameterSpec, RoleSpec
from ._growth import GrowthInteraction


class ExponentialGrowthInteraction(GrowthInteraction):
    """Growth at a constant per capita rate, `dn/dt = r n`, the exponential form of a growth without harvest.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose population grows.

        r:
            The per capita growth rate, negative for decay.

        state:
            The name of the state of the target that holds the population.

    Examples:
        >>> from skcomplex.core.components.elements import CompartmentElement
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('rabbits', n=1.0)],
        ...     interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.5)],
        ... )
        >>> trajectory = Simulation(system, dt=0.01).prepare().run(until=2.0).trajectory_
        >>> round(float(trajectory.values('rabbits.n')[-1]), 3)
        2.718
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec()] = 1.0,
        state: str = 'n',
    ) -> None:
        super().__init__(label, target=target, r=r, form='exponential', state=state)

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the growth `r n`, with no harvest.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        return self._get_growth(namespace)
