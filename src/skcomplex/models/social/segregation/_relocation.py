"""Relocation of the Schelling model, where every resident with too few similar neighbours moves to an empty cell."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.real_set import RealSet

from ....core.interactions import MovementInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, RoleSpec
from ._resident_view import ResidentView


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
