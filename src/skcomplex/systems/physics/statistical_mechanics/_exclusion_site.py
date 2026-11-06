"""Site of the totally asymmetric simple exclusion process, empty at 0 or holding one particle at 1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ExclusionSiteElement(Element):
    """Site of the totally asymmetric simple exclusion process, empty at 0 or holding one particle at 1.

    It holds the probabilities of a hop, of an entry at the left end of an open line, and of an exit at its right end,
    which the transfers of the process read.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        occupied:
            The initial occupation.

        p:
            The probability of a hop.

        alpha:
            The probability of an entry at the left end of an open line.

        beta:
            The probability of an exit at the right end of an open line.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ExclusionSiteElement
        >>> ExclusionSiteElement('site').p
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        occupied: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.occupied = occupied
        self.p = p
        self.alpha = alpha
        self.beta = beta
