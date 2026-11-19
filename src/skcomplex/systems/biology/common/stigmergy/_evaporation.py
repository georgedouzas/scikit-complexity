"""Define evaporation of a mark left in the environment.

The mark is lost at a per unit rate that its holder sets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, RateLaw
from .....core.components.spaces import BaseSpace
from .....core.errors import ValidationError
from .....core.types import Role


class EvaporationInteraction(Interaction):
    """Evaporation of a mark, `dm/dt = -e m`.

    The rate `e` is a parameter of the element that holds the mark.

    Animals leave marks such as pheromone or emotional traces in the cells of a field, and every mark evaporates at
    the rate its cell holds, so the rate can vary from cell to cell.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or field that holds the mark.

        state:
            The name of the state of the target that holds the mark.

        evaporation:
            The name of the parameter of the target that holds the rate of evaporation.

    Raises:
        ValidationError:
            The state or the name of the rate of evaporation is not a name.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.stigmergy import EvaporationInteraction
        >>> evaporation = EvaporationInteraction('evaporation', target='scent', state='level', evaporation='e')
        >>> system = System(elements=[Element('scent', values={'level': 8.0, 'e': 0.5})], interactions=[evaporation])
        >>> trajectory = system.prepare().simulate(until=2.0, dt=0.01)
        >>> round(float(trajectory.get_values('scent.level')[-1]), 3)
        2.943
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'level',
        evaporation: str = 'evaporation',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.evaporation = evaporation

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Raise an error for a name of the rate of evaporation that is no parameter.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The state or the name of the rate of evaporation is not a name.
        """
        super()._prepare(context)
        if not isinstance(self.evaporation, str) or not self.evaporation.isidentifier():
            error_msg = f'The evaporation of {self.label!r} is {self.evaporation!r}. Name a parameter of the target.'
            raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [RateLaw(getattr(context.target, self.state), self._get_rate(context))]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the loss `-e m` of the mark.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the mark.
        """
        loss: Expression = -getattr(namespace.target, self.evaporation) * getattr(namespace.target, self.state)
        return loss
