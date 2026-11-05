"""Bosonic mode of the Hilbert space it takes, with `annihilate`, `create`, and `number`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import CC

from .....core.components.elements import Element
from .....core.specs import HilbertSpec, StateSpec, VisualSpec


class ModeElement(Element):
    """Bosonic mode of the Hilbert space it takes, with `annihilate`, `create`, and `number`.

    Args:
        label:
            The label of the mode.

        psi:
            The state in the Hilbert space: the number of quanta, or a vector.

        hilbert:
            The Hilbert space, `mode_space(quanta)`.

        visual:
            How the mode is drawn.

    Examples:
        >>> from skcomplex.systems.physics.common.quantum import ModeElement, mode_space
        >>> ModeElement('cavity', hilbert=mode_space(4)).hilbert.dimension
        5
    """

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 0,
        *,
        hilbert: HilbertSpec | None = None,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert
