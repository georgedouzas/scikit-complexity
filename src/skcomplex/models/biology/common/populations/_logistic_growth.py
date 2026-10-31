"""Define logistic growth, a population that grows towards a carrying capacity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....specs import ParameterSpec, RoleSpec
from ._growth import GrowthInteraction


class LogisticGrowthInteraction(GrowthInteraction):
    """Growth limited by a carrying capacity, `dn/dt = r n (1 - (n(t - tau) / K) ** p)`, without harvest.

    The exponent `p` gives the generalized logistic of Richards, with `p = 1` for Verhulst's, and a delay `tau` above 0
    gives Hutchinson's delayed logistic equation.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose population grows.

        r:
            The per capita growth rate at low density.

        K:
            The carrying capacity.

        p:
            The exponent of the limiting factor.

        tau:
            The delay of the limiting factor, 0 for none.

        state:
            The name of the state of the target that holds the population.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.common.populations import LogisticGrowthInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('rabbits', n=1.0)],
        ...     interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.0, K=50.0)],
        ... )
        >>> trajectory = Simulation(system, dt=0.1).check().prepare().run(until=30.0).trajectory_
        >>> round(float(trajectory.values('rabbits.n')[-1]), 3)
        50.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        state: str = 'n',
    ) -> None:
        super().__init__(label, target=target, r=r, K=K, p=p, tau=tau, form='logistic', state=state)

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the logistic growth, with no harvest.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        return self._get_growth(namespace)
