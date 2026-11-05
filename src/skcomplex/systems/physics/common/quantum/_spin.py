"""Spin of the Hilbert space it takes, with the spin components `x`, `y`, `z` and the ladder operators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import qutip
from sage.all__sagemath_symbolics import CC

from .....core.components.elements import Element
from .....core.errors import ValidationError
from .....core.specs import HilbertSpec, StateSpec, VisualSpec

MIN_SPIN_DIMENSION = 2


def spin_space(j: float = 0.5) -> HilbertSpec:
    """Return the Hilbert space of a spin of quantum number `j`, with the basis from `m = j` down to `m = -j`.

    Its operators are the spin components `x`, `y`, `z` and the ladder operators `plus` and `minus`.

    Args:
        j:
            The spin quantum number, a positive multiple of one half.

    Returns:
        space:
            The Hilbert space, of dimension `2 j + 1`.

    Raises:
        ValidationError:
            The quantum number is not a positive multiple of one half.

    Examples:
        >>> from skcomplex.systems.physics.common.quantum import spin_space
        >>> spin_space(1).dimension
        3
    """
    size = 2 * j + 1
    if not float(size).is_integer() or size < MIN_SPIN_DIMENSION:
        error_msg = f'A spin has j = {j!r}, which is not a positive multiple of one half.'
        raise ValidationError(error_msg)
    operators = {
        'x': (qutip.jmat(j, 'x').full(), True),
        'y': (qutip.jmat(j, 'y').full(), True),
        'z': (qutip.jmat(j, 'z').full(), True),
        'plus': (qutip.jmat(j, '+').full(), False),
        'minus': (qutip.jmat(j, '-').full(), False),
    }
    return HilbertSpec(int(size), operators, basis={'up': 0, 'down': int(size) - 1})


SPIN_HALF = spin_space()


class SpinElement(Element):
    """Spin of the Hilbert space it takes, with the spin components `x`, `y`, `z` and the ladder operators.

    The basis is ordered from `m = j` to `m = -j`, so `'up'` is index 0 and `'down'` is the last index. Its laws read
    the operators of its state `psi`, such as `local_operator(namespace.psi, 'z')`.

    Args:
        label:
            The label of the spin.

        psi:
            The state in the Hilbert space: `'up'`, `'down'`, a basis index, or a vector.

        hilbert:
            The Hilbert space, `spin_space(j)`, by default that of a spin one half.

        visual:
            How the spin is drawn.

    Examples:
        >>> from skcomplex.systems.physics.common.quantum import SpinElement, spin_space
        >>> SpinElement('s', hilbert=spin_space(1)).hilbert.dimension
        3
    """

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        *,
        hilbert: HilbertSpec = SPIN_HALF,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert
