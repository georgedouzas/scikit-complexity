"""Two-level system with the Pauli matrices `x`, `y`, `z` and the ladder operators `plus` and `minus`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import qutip
from sage.all__sagemath_symbolics import CC

from .....core.elements import Element
from .....specs import HilbertSpec, StateSpec, VisualSpec

QUBIT = HilbertSpec(
    2,
    {
        'x': (qutip.sigmax().full(), True),
        'y': (qutip.sigmay().full(), True),
        'z': (qutip.sigmaz().full(), True),
        'plus': (qutip.sigmap().full(), False),
        'minus': (qutip.sigmam().full(), False),
    },
    basis={'up': 0, 'excited': 0, 'down': 1, 'ground': 1},
)


class QubitElement(Element):
    """Two-level system with the Pauli matrices `x`, `y`, `z` and the ladder operators `plus` and `minus`.

    The state `'up'`, also `'excited'`, is index 0 with `z = 1`, and `'down'`, also `'ground'`, is index 1.

    Args:
        label:
            The label of the qubit.

        psi:
            The state in the Hilbert space: `'up'`, `'down'`, `'excited'`, `'ground'`, a basis index, or a vector.

        hilbert:
            The Hilbert space of the qubit.

        visual:
            How the qubit is drawn.

    Examples:
        >>> from skcomplex.models.physics.common.quantum import QubitElement
        >>> sorted(QubitElement('q').hilbert.operators)
        ['minus', 'plus', 'x', 'y', 'z']
    """

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        *,
        hilbert: HilbertSpec = QUBIT,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert
