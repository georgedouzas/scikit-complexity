"""Relaxation of the speed `v` of every car of a ring road.

The speed relaxes towards the optimal velocity of its headway.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, tanh
from sage.sets.real_set import RealSet

from ....core.components.laws.utils import mod, shift_member
from ....core.specs import ParameterSpec, RoleSpec
from .traffic import SpeedAdaptationInteraction


class OptimalVelocityInteraction(SpeedAdaptationInteraction):
    """Relaxation of the speed `v` of every car of a ring road.

    The speed follows `v' = a (V(h) - v)` with the optimal velocity `V(h) = V0 (tanh(h - hc) + tanh(hc))` of the
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

    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        V0: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        hc: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        road_length: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 40.0,
    ) -> None:
        super().__init__(label, target=target, state='v')
        self.a = a
        self.V0 = V0
        self.hc = hc
        self.road_length = road_length

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the optimal velocity of the headway of every car."""
        headway = OptimalVelocityInteraction._headway(namespace.target.x, namespace.road_length)
        return namespace.V0 * (tanh(headway - namespace.hc) + tanh(namespace.hc))

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time `1 / a`."""
        return 1 / namespace.a

    @staticmethod
    def _headway(x: Expression, road_length: Expression) -> Expression:
        """Return the distance from every car of a ring road to the car ahead.

        The car ahead is the next member of the group.
        """
        return mod(shift_member(x, 1) - x, road_length)
