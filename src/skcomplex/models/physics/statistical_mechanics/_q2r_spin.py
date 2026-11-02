"""Site of the Q2R automaton, a spin and a phase that says whether the site updates in the current step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class Q2RSpinElement(Element):
    """Site of the Q2R automaton, a spin and a phase that says whether the site updates in the current step.

    The local rule of Q2R flips the spins of the updating sites whose neighbours sum to zero, which leaves the Ising
    energy unchanged, and swaps the phase of every site, so the two sublattices update in turn and the dynamics is
    reversible.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        spin:
            The initial spin, 1 or -1.

        phase:
            The initial phase, 1 for updating and 0 for quiescent.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import Q2RSpinElement
        >>> Q2RSpinElement('site').phase
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        phase: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.spin = spin
        self.phase = phase
