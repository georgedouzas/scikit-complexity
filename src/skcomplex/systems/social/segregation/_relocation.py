"""Relocation of the Schelling system, where every resident with too few similar neighbours moves to an empty cell."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components.interactions import MovementInteraction
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ._resident_view import ResidentView


class RelocationInteraction(MovementInteraction):
    """Relocation of the Schelling system, where every resident with too few similar neighbours moves to an empty cell.

    Every step every resident, in random order, moves to a random empty cell when the share of its neighbours of its
    own kind is below its threshold. With `choices`, the Big Sort of Bishop's hypothesis, every resident moves with
    the probability `probability` whatever its neighbours, to the cell with the highest share of neighbours of its own
    kind among `choices` random empty cells, the first of them on ties.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of residents.

        probability:
            The probability that an unhappy resident moves in a step, or that any resident moves under the Big Sort.

        choices:
            The number of random empty cells a moving resident compares under the Big Sort, or `None` for Schelling's
            rule.

        phase:
            The phase of the step the relocation runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.segregation import RelocationInteraction
        >>> RelocationInteraction('relocation', target='residents').target
        'residents'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        choices: int | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, probability=probability, phase=phase)
        self.choices = choices

    def _get_destination(self, agent: AgentViewProtocol, system: SystemViewProtocol, occupied: set[object]) -> object:
        """Return a random empty cell for a resident with too few similar neighbours, or `None` to stay."""
        if not isinstance(agent, ResidentView) or not isinstance(system, SystemView):
            return None
        if self.choices is not None:
            return self._get_sorted_destination(agent, system)
        neighbors = [item for item in agent.neighbors() if isinstance(item, ResidentView)]
        if not neighbors:
            return None
        similar = sum(neighbor.kind == agent.kind for neighbor in neighbors) / len(neighbors)
        return system.random_empty_cell() if similar < agent.threshold else None

    def _get_sorted_destination(self, resident: ResidentView, system: SystemView) -> object:
        """Return the empty cell with the highest share of similar neighbours among random empty cells."""
        storage = system.storage
        if storage.get('time') != system.time:
            storage['time'] = system.time
            storage['kinds'] = {item.location_key: item.kind for item in system.agents(self.target, ResidentView)}
        kinds = storage['kinds']
        if not isinstance(kinds, dict):
            return None
        width, height = system.dimensions or (0, 0)
        best, best_share = None, -1.0
        for _ in range(int(self.choices or 0)):
            cell = system.random_empty_cell()
            column, row = (int(item) for item in np.asarray(cell).ravel())
            around = [
                kinds[column + dx, row + dy]
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if (dx or dy)
                and 0 <= column + dx < width
                and 0 <= row + dy < height
                and (column + dx, row + dy) in kinds
            ]
            share = sum(kind == resident.kind for kind in around) / len(around) if around else 0.0
            if share > best_share:
                best, best_share = cell, share
        if best is not None:
            kinds.pop(resident.location_key, None)
            kinds[tuple(int(item) for item in np.asarray(best).ravel())] = resident.kind
        return best
