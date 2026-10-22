"""Segregation models: residential sorting of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import Element, LatticeSpace, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView


class Resident(Element):
    """Resident of the Schelling model, who moves to an empty cell while too few neighbours share its kind.

    Args:
        label:
            The label of the agent or group.

        kind:
            The kind of the resident, 0 or 1.

        threshold:
            The smallest share of neighbours of the same kind that keeps a resident in place.

    Examples:
        >>> from skcomplex.models.social.segregation import Resident
        >>> Resident.group('residents', 10, kind=1.0)
        Group('residents', element=Resident, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'kind', 'color_range': (0.0, 1.0)})
    kind: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    threshold: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the relocation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.relocate)]

    def relocate(self, agent: AgentView, model: ModelView) -> None:
        """Move to a random empty cell while the share of similar neighbours is below the threshold.

        Args:
            agent:
                The resident.

            model:
                The model.
        """
        neighbors = agent.neighbors()
        if not neighbors:
            return
        similar = sum(neighbor.kind == agent.kind for neighbor in neighbors) / len(neighbors)
        if similar < agent.threshold:
            agent.move_to(model.random_empty_cell())


@register_model(
    field='social/segregation',
    tags=('segregation', 'agents', 'grid'),
    references=('Schelling 1971',),
    defaults={'until': 50},
)
def schelling(
    width: int = 20,
    height: int = 20,
    *,
    density: float = 0.8,
    minority: float = 0.3,
    threshold: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a Schelling segregation model on a grid with at most one resident per cell.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        density:
            The share of cells with a resident.

        minority:
            The share of residents of kind 1.

        threshold:
            The smallest share of similar neighbours that keeps a resident in place.

        random_state:
            The seed or generator of the kinds.

    Returns:
        system:
            The model, with the group `residents`.

    Examples:
        >>> from skcomplex.models.social.segregation import schelling
        >>> schelling(10, 10, density=0.5).elements[0].size
        50
    """
    rng = np.random.default_rng(random_state)
    size = int(width * height * density)
    kinds = (rng.random(size) < minority).astype(float)
    residents = Resident.group('residents', size, kind=kinds, threshold=threshold)
    return System(components=[residents], space=LatticeSpace((width, height), boundary='fixed', capacity=1))
