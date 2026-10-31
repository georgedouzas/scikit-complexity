"""Define mortality, the loss of a population at a constant per capita rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.interactions import RateInteraction
from .....specs import ParameterSpec, RoleSpec


class MortalityInteraction(RateInteraction):
    """Death or loss at a constant per capita rate, `dn/dt = -d n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose population dies.

        d:
            The per capita death rate.

        state:
            The name of the state of the target that holds the population.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.common.populations import MortalityInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('foxes', n=8.0)],
        ...     interactions=[MortalityInteraction('death', target='foxes', d=0.5)],
        ... )
        >>> trajectory = Simulation(system, dt=0.01).check().prepare().run(until=2.0).trajectory_
        >>> round(float(trajectory.values('foxes.n')[-1]), 3)
        2.943
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        state: str = 'n',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.d = d

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the loss `-d n`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        loss: Expression = -namespace.d * getattr(namespace.target, self.state)
        return loss
