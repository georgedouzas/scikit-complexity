"""Define evaporation of a mark left in the environment.

The mark is lost at a per unit rate that its holder sets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from .....core.components.interactions import RateInteraction
from .....core.errors import ValidationError
from .....core.specs import RoleSpec


class EvaporationInteraction(RateInteraction):
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
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.stigmergy import EvaporationInteraction
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class ScentElement(Element):
        ...     def __init__(
        ...         self, label, level: Annotated[float, StateSpec()] = 8.0, e: Annotated[float, ParameterSpec()] = 0.5
        ...     ):
        ...         super().__init__(label)
        ...         self.level = level
        ...         self.e = e
        >>> evaporation = EvaporationInteraction('evaporation', target='scent', state='level', evaporation='e')
        >>> system = System(elements=[ScentElement('scent')], interactions=[evaporation])
        >>> trajectory = simulate(system, until=2.0, dt=0.01)
        >>> round(float(trajectory.values('scent.level')[-1]), 3)
        2.943
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'level',
        evaporation: str = 'evaporation',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.evaporation = evaporation

    def _check(self) -> None:
        """Raise an error for a name of the rate of evaporation that is no parameter.

        Raises:
            ValidationError:
                The state or the name of the rate of evaporation is not a name.
        """
        super()._check()
        if not isinstance(self.evaporation, str) or not self.evaporation.isidentifier():
            error_msg = f'The evaporation of {self.label!r} is {self.evaporation!r}. Name a parameter of the target.'
            raise ValidationError(error_msg)

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
