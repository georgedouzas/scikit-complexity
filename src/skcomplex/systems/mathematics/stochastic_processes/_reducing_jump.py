"""Define reducing jumps, the moves of a ball to a random lower state with a restart from the top state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.interactions import MapInteraction
from ....core.components.types import SystemViewProtocol
from ....core.specs import RoleSpec
from ._reducing_ball_view import ReducingBallView


class ReducingJumpInteraction(MapInteraction):
    """Jump of the sample-space-reducing process, to a random lower state, to any state, or back to the top state.

    A ball at state 1 restarts from the top state `n_states`. Otherwise it jumps with probability `lam` uniformly to a
    state below its own, and else uniformly to any state, so the states are visited with probabilities proportional to
    `i^(-lam)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of balls.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ReducingJumpInteraction
        >>> ReducingJumpInteraction('jump', target='balls').state
        'level'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, state='level', function=self._get_level, form='agent')

    def _get_level(self, agent: ReducingBallView, system: SystemViewProtocol) -> float:
        """Return the next state of a ball."""
        top = round(agent.n_states)
        if agent.level <= 1:
            return float(top)
        if system.rng.random() < agent.lam:
            return float(system.rng.integers(1, round(agent.level)))
        return float(system.rng.integers(1, top + 1))
