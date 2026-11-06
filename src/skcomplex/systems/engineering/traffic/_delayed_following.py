"""Acceleration of a follower car by the speed difference to its leader, both seen a reaction time late."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import delayed
from ....core.specs import ParameterSpec, RoleSpec


class DelayedFollowingInteraction(RateInteraction):
    """Acceleration of a follower car by the speed difference to its leader, both seen a reaction time late.

    The speed of the follower follows the Gazis-Herman-Rothery law
    `v_f'(t) = λ (v_l - v_f)(t - T) / (x_l - x_f)(t - T)^m` of the sensitivity `λ`, the reaction time `T`, and the
    power `m` of the gap.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the follower car.

        leader:
            The path of the car ahead of the follower.

        reaction:
            The sensitivity `λ`.

        T:
            The reaction time.

        power:
            The power `m` of the gap.

    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        leader: Annotated[str, RoleSpec()],
        reaction: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        T: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        power: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, target=target, state='v')
        self.leader = leader
        self.reaction = reaction
        self.T = T
        self.power = power

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the delayed response of the follower to its leader."""
        s, leader, follower = namespace, namespace.leader, namespace.target
        gap = delayed(leader.x, s.T) - delayed(follower.x, s.T)
        return s.reaction * (delayed(leader.v, s.T) - delayed(follower.v, s.T)) / gap**s.power
