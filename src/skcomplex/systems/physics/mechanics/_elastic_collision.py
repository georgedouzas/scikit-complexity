"""Elastic collision of two equal masses in a row, which exchange their velocities when they touch."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import (
    Interaction,
)
from ....core.components.laws import BaseLaw, CrossingLaw
from ....core.components.laws.utils import derivative
from ....core.specs import ParameterSpec, RoleSpec


class ElasticCollisionInteraction(Interaction):
    """Elastic collision of two equal masses in a row, which exchange their velocities when they touch.

    The collision is the event at which the gap `x_right - x_left - diameter` between the balls falls to zero, and
    equal masses swap their velocities in an elastic collision.

    Args:
        label:
            The label of the interaction.

        left:
            The path of the left ball, with a position `x` of order 2.

        right:
            The path of the right ball, with a position `x` of order 2.

        diameter:
            The diameter of the balls.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ElasticCollisionInteraction
        >>> ElasticCollisionInteraction('collision', left='a', right='b', diameter=0.2).diameter
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        left: Annotated[str, RoleSpec()],
        right: Annotated[str, RoleSpec()],
        diameter: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.left = left
        self.right = right
        self.diameter = diameter

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the crossing law of the contact, which swaps the velocities of the balls.

        Args:
            s:
                The namespace of the symbols the laws can use, with `s.left` and `s.right`.

        Returns:
            laws:
                The crossing law.
        """
        return [
            CrossingLaw(
                s.right.x - s.left.x - s.diameter,
                {
                    derivative(s.left.x, 1): derivative(s.right.x, 1),
                    derivative(s.right.x, 1): derivative(s.left.x, 1),
                },
                direction=-1,
            ),
        ]
