"""Weekly move of every cow, in random order, to the nearest free patch with enough grass, or death."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import MovementInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import ParameterSpec, RoleSpec
from ._pasture_view import PastureView


class HerdingInteraction(MovementInteraction):
    """Weekly move of every cow, in random order, to the nearest free patch with enough grass, or death.

    It is a movement of the cows whose step leads every cow to the patch nearest to it that no other cow occupies and
    whose grass reaches `threshold`, and removes a cow that finds none. Every cow moves every week, the probability
    of movement fixed to 1 without a parameter.

    Args:
        label:
            The label of the interaction.

        cows:
            The path of the group of cows.

        pasture:
            The path of the group of patches.

        threshold:
            The least grass a patch needs for a cow to move there.

    Examples:
        >>> from skcomplex.models.biology.ecology import HerdingInteraction
        >>> HerdingInteraction('herding', cows='cows', pasture='pasture').threshold
        0.9
    """

    def __init__(
        self,
        label: str,
        *,
        cows: Annotated[str, RoleSpec()],
        pasture: Annotated[str, RoleSpec()],
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
    ) -> None:
        super().__init__(label, target=cows, step=self._herd)
        self.cows = cows
        self.pasture = pasture
        self.threshold = threshold

    def _get_role(self, role: str) -> str:
        """Return the cows as the target of the movement."""
        return 'cows' if role == 'target' else role

    @staticmethod
    def _cell(agent: AgentViewProtocol) -> tuple[int, ...]:
        """Return the cell of an agent as a tuple of integers."""
        location = agent.location
        if not isinstance(location, tuple):
            error_msg = (
                f'The agent is located at a {type(location).__name__}, not at the coordinates of a lattice cell.'
            )
            raise TypeError(error_msg)
        return tuple(int(item) for item in location)

    def _herd(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> tuple[int, ...] | None:
        """Return the cell of the nearest free patch with enough grass, or remove the cow that finds none."""
        patches = [cast(PastureView, patch) for patch in model.agents('pasture')]
        cells = [self._cell(patch) for patch in patches]
        index = {cell: number for number, cell in enumerate(cells)}
        occupied = np.zeros(len(cells), dtype=bool)
        occupied[[index[self._cell(cow)] for cow in model.agents('cows') if cow.alive and cow.id != agent.id]] = True
        free = (np.asarray([patch.grass for patch in patches]) >= model.parameter('threshold')) & ~occupied
        if not free.any():
            agent.remove()
            return None
        distances = np.hypot(*(np.asarray(cells, dtype=float) - np.asarray(self._cell(agent), dtype=float)).T)
        return cells[int(np.argmin(np.where(free, distances, np.inf)))]
