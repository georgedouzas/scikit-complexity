"""Relaxation of the speed `x'` of every car of a ring road.

The speed relaxes towards the optimal velocity of its headway.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, tanh
from sage.sets.real_set import RealSet

from .....core.components.laws.utils import mod, shift_member
from .....core.specs import Role
from ._speed_adaptation import SpeedAdaptationInteraction


class OptimalVelocityInteraction(SpeedAdaptationInteraction):
    """Relaxation of the speed `x'` of every car of a ring road.

    The speed follows `x'' = a (V(h) - x')` with the optimal velocity `V(h) = V0 (tanh(h - hc) + tanh(hc))` of the
    headway `h` to the next car, a relaxation over the time `1 / a`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of cars, in the order they drive.

        a:
            The sensitivity, the inverse of the relaxation time.

        V0:
            The scale of the optimal velocity.

        hc:
            The safe headway, where the optimal velocity turns.

        road_length:
            The length of the ring road.

    Examples:
        >>> from skcomplex.systems.engineering.common.traffic import OptimalVelocityInteraction
        >>> OptimalVelocityInteraction('follow', target='cars').state
        "x'"
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        a: Annotated[float, RealSet.open(0, oo)] = 1.0,
        V0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        hc: Annotated[float, RealSet.open(0, oo)] = 2.0,
        road_length: Annotated[float, RealSet.open(0, oo)] = 40.0,
    ) -> None:
        super().__init__(label, target=target, state="x'")
        self.a = a
        self.V0 = V0
        self.hc = hc
        self.road_length = road_length

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the optimal velocity of the headway of every car."""
        headway = OptimalVelocityInteraction._get_headway(namespace.target.x, namespace.road_length)
        return namespace.V0 * (tanh(headway - namespace.hc) + tanh(namespace.hc))

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time `1 / a`."""
        return 1 / namespace.a

    @staticmethod
    def _get_headway(x: Expression, road_length: Expression) -> Expression:
        """Return the distance from every car of a ring road to the car ahead.

        The car ahead is the next member of the group.
        """
        return mod(shift_member(x, 1) - x, road_length)
