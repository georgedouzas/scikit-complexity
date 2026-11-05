"""Binary cell that holds the probability that the output of its local rule is flipped."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.specs import ParameterSpec, VisualSpec
from ._binary_cell import BinaryCellElement


class NoisyElementaryCellElement(BinaryCellElement):
    """Binary cell that holds the probability that the output of its local rule is flipped.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        alive:
            The initial value, 0 or 1.

        noise:
            The probability that the output of the rule is flipped.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import NoisyElementaryCellElement
        >>> NoisyElementaryCellElement('cell', noise=0.1).noise
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: float = 0,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual, alive=alive)
        self.noise = noise
