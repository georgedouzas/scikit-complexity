"""Bosonic mode.

The module holds its Hilbert space truncated at a number of quanta, and the element of a mode that takes it.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import qutip
from sage.all__sagemath_symbolics import CC

from .....core.components.elements import Element
from .....core.errors import ValidationError
from .....core.specs import HilbertSpec, StateSpec, VisualSpec


def mode_space(quanta: int = 4) -> HilbertSpec:
    """Return the Hilbert space of a bosonic mode truncated at a number of quanta.

    Its operators are `annihilate`, `create`, and `number`, and its basis state `n` holds `n` quanta.

    Args:
        quanta:
            The largest number of quanta, so the dimension is `quanta + 1`.

    Returns:
        space:
            The Hilbert space.

    Raises:
        ValidationError:
            The number of quanta is not a positive integer.

    Examples:
        >>> from skcomplex.systems.physics.common.quantum import mode_space
        >>> mode_space(4).dimension
        5
    """
    if isinstance(quanta, bool) or not isinstance(quanta, int | np.integer) or quanta < 1:
        error_msg = f"A mode has quanta = {quanta!r}, which is not a positive integer."
        raise ValidationError(error_msg)
    size = int(quanta) + 1
    operators = {
        "annihilate": (qutip.destroy(size).full(), False),
        "create": (qutip.create(size).full(), False),
        "number": (qutip.num(size).full(), True),
    }
    return HilbertSpec(size, operators, display="number")


class ModeElement(Element):
    """Bosonic mode of the Hilbert space it takes.

    The mode has `annihilate`, `create`, and `number`.

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
        visual: VisualSpec | None = VisualSpec("expectations"),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert
