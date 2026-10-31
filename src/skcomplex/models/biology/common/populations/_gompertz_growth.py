"""Define Gompertz growth, a population whose per capita rate falls with the logarithm of its size."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....specs import ParameterSpec, RoleSpec
from ._growth import GrowthInteraction


class GompertzGrowthInteraction(GrowthInteraction):
    """Gompertz growth towards a capacity, `dn/dt = r n log(K / n(t - tau))`, without harvest.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose population grows.

        r:
            The growth rate.

        K:
            The capacity.

        tau:
            The delay of the limiting factor, 0 for none.

        state:
            The name of the state of the target that holds the population.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.common.populations import GompertzGrowthInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('tumour', n=1.0)],
        ...     interactions=[GompertzGrowthInteraction('growth', target='tumour', r=1.0, K=20.0)],
        ... )
        >>> trajectory = Simulation(system, dt=0.1).check().prepare().run(until=30.0).trajectory_
        >>> round(float(trajectory.values('tumour.n')[-1]), 3)
        20.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0719,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1964.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        state: str = 'n',
    ) -> None:
        super().__init__(label, target=target, r=r, K=K, tau=tau, form='gompertz', state=state)

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the Gompertz growth, with no harvest.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        return self._get_growth(namespace)
