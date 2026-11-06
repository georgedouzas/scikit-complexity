"""Define power-law growth, a population whose growth scales as a power of its size."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.specs import ParameterSpec, RoleSpec
from ._growth import GrowthInteraction


class PowerLawGrowthInteraction(GrowthInteraction):
    """Growth that scales as a power of the size, `dn/dt = r n ** p`, without harvest.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose population grows.

        r:
            The growth rate.

        p:
            The exponent, `2/3` for a growing rim and 1 for exponential growth.

        state:
            The name of the state of the target that holds the population.

    Examples:
        >>> from skcomplex.core.components.elements import CompartmentElement
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.common.populations import PowerLawGrowthInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('tumour', n=1.0)],
        ...     interactions=[PowerLawGrowthInteraction('growth', target='tumour', r=1.0, p=0.5)],
        ... )
        >>> trajectory = Simulation(system, dt=0.01).prepare().run(until=2.0).trajectory_
        >>> round(float(trajectory.values('tumour.n')[-1]), 3)
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.32,
        p: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.58,
        state: str = 'n',
    ) -> None:
        super().__init__(label, target=target, r=r, p=p, form='power', state=state)

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the power-law growth, with no harvest.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        return self._get_growth(namespace)
