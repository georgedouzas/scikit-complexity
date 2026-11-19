"""Rössler flow `x' = -y - z`, `y' = x + a y`, `z' = b + z (x - c)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Role


class RosslerInteraction(Interaction):
    """The laws of the Rössler flow `x' = -y - z`, `y' = x + a y`, `z' = b + z (x - c)`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose coordinates `x`, `y`, and `z` the flow moves.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The threshold of the third coordinate's growth.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import RosslerInteraction
        >>> RosslerInteraction('flow', target='rossler', c=4.0).c
        4.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'rossler',
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.c = c

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the three coordinates as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, z = context.target.x, context.target.y, context.target.z
            self.laws_: list[BaseComponent] = [
                RateLaw(x, -y - z),
                RateLaw(y, x + context.a * y),
                RateLaw(z, context.b + z * (x - context.c)),
            ]
