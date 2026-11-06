"""Surface of a nucleus in the quartic quadrupole potential, with the coordinates `q1, q2` and momenta `p1, p2`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class QuadrupoleElement(Element):
    """Surface of a nucleus in the quartic quadrupole potential, with the coordinates `q1, q2` and momenta `p1, p2`.

    The quadrupole deformation of the nuclear surface moves under the Hamiltonian
    `H = (q² + p²) / 2 + B q1 (3 q2² - q1²) / √2 + q⁴ / 8`, with `q² = q1² + q2²`, which a Hamiltonian interaction
    turns into the equations of motion. The cubic term of strength `B` breaks the rotational symmetry and makes the
    motion chaotic at intermediate energies, while the quartic term keeps it bounded at every energy.

    Args:
        label:
            The label of the nucleus.

        visual:
            How the nucleus is drawn.

        q1:
            The initial first coordinate.

        q2:
            The initial second coordinate.

        p1:
            The initial first momentum.

        p2:
            The initial second momentum.

        B:
            The strength of the cubic term.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import QuadrupoleElement
        >>> QuadrupoleElement('nucleus').B
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('bodies', {'positions': ('q1', 'q2'), 'trail': None}),
        *,
        q1: Annotated[float, StateSpec()] = 0.0,
        q2: Annotated[float, StateSpec()] = 1.0,
        p1: Annotated[float, StateSpec()] = 0.5,
        p2: Annotated[float, StateSpec()] = 0.0,
        B: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.q1 = q1
        self.q2 = q2
        self.p1 = p1
        self.p2 = p2
        self.B = B
