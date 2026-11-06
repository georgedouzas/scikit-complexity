"""One generation of the Moran process: as many birth and death events as there are individuals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components.interactions import EvolutionInteraction
from ....core.components.types import SystemViewProtocol
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array


class MoranStepInteraction(EvolutionInteraction):
    """One generation of the Moran process: as many birth and death events as there are individuals.

    At every event an individual reproduces with probability proportional to its fitness `(1 + advantage)^trait`,
    its offspring mutates to the trait one above or one below with probability `mutation / 2` each, and it replaces an
    individual chosen uniformly at random, so the size of the population stays constant. It is the Moran replacement
    of `EvolutionInteraction` with proportional selection and steps of the trait.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of individuals.

        advantage:
            The selective advantage of one unit of the trait.

        mutation:
            The probability that an offspring mutates.

        phase:
            The phase of the step the generation runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.evolution import MoranStepInteraction
        >>> MoranStepInteraction('moran', target='population', advantage=0.1).replacement
        'moran'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        advantage: Annotated[float, ParameterSpec()] = 0.0,
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            strategy='trait',
            selection='proportional',
            replacement='moran',
            mutation=mutation,
            alphabet=None,
            change='step',
            phase=phase,
        )
        self.advantage = advantage

    def _evaluate(self, strategies: Array, system: SystemViewProtocol) -> Array:
        """Return the fitness `(1 + advantage)^trait` of every trait.

        Args:
            strategies:
                The traits.

            system:
                The system view of the interaction.

        Returns:
            fitness:
                The fitness of every trait.
        """
        return (1 + system.parameter('advantage')) ** np.ravel(strategies)
