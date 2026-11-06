"""Attempt to cut the cost of a technology by redesigning a random component and the components that depend on it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ....core.types import Array
from ._part_view import PartView


class RedesignInteraction(Interaction):
    """Attempt to cut the cost of a technology by redesigning a random component and the components that depend on it.

    The redesigned components get new costs drawn uniformly from `[0, 1]`, and the redesign is kept only if the total
    cost falls. `dependencies[i][j]` is 1 when a change of component `j` requires a change of component `i`.

    Args:
        label:
            The label of the interaction.

        components:
            The path of the group of components.

        dependencies:
            The design structure matrix.

    Examples:
        >>> from skcomplex.systems.economics.innovation import RedesignInteraction
        >>> RedesignInteraction('redesign', components='components', dependencies=[[1.0]]).dependencies
        [[1.0]]
    """

    def __init__(
        self,
        label: str,
        *,
        components: Annotated[str, RoleSpec()],
        dependencies: Sequence[Sequence[float]] = (),
    ) -> None:
        super().__init__(label)
        self.components = components
        self.dependencies = dependencies

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the redesign rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.redesign)]

    def redesign(self, system: SystemView) -> None:
        """Propose new costs for a random component and its dependents and keep them if the total cost falls.

        Args:
            system:
                The system.
        """
        components = sorted(system.agents('components', PartView), key=lambda component: component.id)
        cost = np.array([component.cost for component in components])
        proposal, changed = RedesignInteraction._components(
            cost,
            np.asarray(self.dependencies, dtype=float),
            system.rng,
        )
        if proposal.sum() >= cost.sum():
            return
        for index in changed:
            components[index].cost = float(proposal[index])

    @staticmethod
    def _components(cost: Array, dependencies: Array, rng: np.random.Generator) -> tuple[Array, npt.NDArray[np.intp]]:
        """Return the proposed costs of a redesign of a random component and of the components that depend on it."""
        changed = np.flatnonzero(dependencies[:, int(rng.integers(len(cost)))])
        proposal = cost.copy()
        proposal[changed] = rng.random(len(changed))
        return proposal, changed
