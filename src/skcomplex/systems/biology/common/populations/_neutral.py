"""Define the births and deaths of neutral theory, shared by every species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import JumpLaw
from .....core.components.laws.utils import where
from .....core.types import Role


class NeutralBirthInteraction(Interaction):
    """Births of single individuals at the rate `b0 + b1 x`, shared by every species.

    The constant part `b0` is immigration or speciation, and `b1` the birth rate of an individual.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the species or of the group of species.

        b0:
            The rate of immigration or speciation.

        b1:
            The birth rate of an individual.

    Examples:
        >>> from skcomplex.systems.biology.common.populations import NeutralBirthInteraction
        >>> NeutralBirthInteraction('birth', target='trees').b1
        0.9
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        b0: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        b1: Annotated[float, RealSet.closed_open(0, oo)] = 0.9,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.b0 = b0
        self.b1 = b1

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the births of single individuals.

        Args:
            context:
                The namespace of the species, the time `t`, and the parameters.
        """
        super()._prepare(context)
        x = context.target.x
        self.laws_ = [JumpLaw(context.b0 + context.b1 * x, {x: 1})]


class NeutralDeathInteraction(Interaction):
    """Deaths of single individuals at the rate `d0 + d1 x`, shared by every species.

    The constant part `d0` is emigration or extinction while the species has members, and `d1` the death rate of an
    individual.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the species or of the group of species.

        d0:
            The rate of emigration or extinction while the species has members.

        d1:
            The death rate of an individual.

    Examples:
        >>> from skcomplex.systems.biology.common.populations import NeutralDeathInteraction
        >>> NeutralDeathInteraction('death', target='trees').d1
        1.0
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        d0: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        d1: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.d0 = d0
        self.d1 = d1

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the deaths of single individuals.

        Args:
            context:
                The namespace of the species, the time `t`, and the parameters.
        """
        super()._prepare(context)
        x = context.target.x
        self.laws_ = [JumpLaw(where(x > 0, context.d0, 0) + context.d1 * x, {x: -1})]
