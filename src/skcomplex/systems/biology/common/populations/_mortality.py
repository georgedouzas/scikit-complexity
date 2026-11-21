"""Define mortality, the loss of a population at a constant per capita rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.types import Role


class MortalityInteraction(Interaction):
    """Death or loss at a constant per capita rate, `dx/dt = -d x`.

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
        >>> from skcomplex.systems.biology.common.kinetics import CompartmentElement
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.populations import MortalityInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('foxes', x=8.0)],
        ...     interactions=[MortalityInteraction('death', target='foxes', d=0.5)],
        ... )
        >>> trajectory = system.prepare().simulate(until=2.0, dt=0.01)
        >>> round(float(trajectory.get_values('foxes.x')[-1]), 3)
        2.943
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        d: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        state: str = 'x',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.d = d
        self.state = state

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of the population.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        self.laws_ = [RateLaw(getattr(context.target, self.state), self._get_rate(context))]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the loss `-d x`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        loss: Expression = -namespace.d * getattr(namespace.target, self.state)
        return loss
