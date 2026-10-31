"""Segregation models: residential sorting of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MovementInteraction
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._resident_view import ResidentView


class ResidentElement(Element):
    """Resident of the Schelling model, who moves to an empty cell while too few neighbours share its kind.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

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

    @classmethod
    def _get_view(cls) -> type[ResidentView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return ResidentView


class RelocationInteraction(MovementInteraction):
    """Relocation of the Schelling model, where every resident with too few similar neighbours moves to an empty cell.

    Every step every resident, in random order, moves to a random empty cell when the share of its neighbours of its
    own kind is below its threshold.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of residents.

        probability:
            The probability that an unhappy resident moves in a step.

        phase:
            The phase of the step the relocation runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.segregation import RelocationInteraction
        >>> RelocationInteraction('relocation', target='residents').target
        'residents'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, probability=probability, phase=phase)

    def _get_destination(self, agent: AgentViewProtocol, model: ModelViewProtocol, occupied: set[object]) -> object:
        """Return a random empty cell for a resident with too few similar neighbours, or `None` to stay."""
        if not isinstance(agent, ResidentView) or not isinstance(model, ModelView):
            return None
        neighbors = [item for item in agent.neighbors() if isinstance(item, ResidentView)]
        if not neighbors:
            return None
        similar = sum(neighbor.kind == agent.kind for neighbor in neighbors) / len(neighbors)
        return model.random_empty_cell() if similar < agent.threshold else None


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
        >>> schelling(10, 10, density=0.5).elements[0].check().prepare().size_
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
    return System(
        elements=[residents],
        interactions=[RelocationInteraction('relocation', target='residents')],
        space=LatticeSpace((width, height), boundary='fixed', capacity=1),
    )
