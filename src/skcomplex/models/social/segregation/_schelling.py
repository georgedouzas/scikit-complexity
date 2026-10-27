"""Segregation models: residential sorting of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import LatticeSpace
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import AgentView, ModelView


class ResidentElement(Element):
    """Resident of the Schelling model, who moves to an empty cell while too few neighbours share its kind.

    Args:
        label:
            The label of the agent or group.

        kind:
            The kind of the resident, 0 or 1.

        threshold:
            The smallest share of neighbours of the same kind that keeps a resident in place.

    Examples:
        >>> from skcomplex.models.social.segregation import ResidentElement
        >>> group = GroupElement('residents', element=ResidentElement, size=10, values={'kind': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('residents', 'ResidentElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'kind', 'color_range': (0.0, 1.0)}),
        *,
        kind: Annotated[float, StateSpec(domain=NN)] = 0.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kind = kind
        self.threshold = threshold

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relocation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.relocate)]

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
    residents = GroupElement(
        'residents',
        element=ResidentElement,
        size=size,
        values={'kind': kinds, 'threshold': threshold},
    )
    return System(elements=[residents], space=LatticeSpace((width, height), boundary='fixed', capacity=1))
