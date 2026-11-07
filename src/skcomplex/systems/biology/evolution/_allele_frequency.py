"""Frequency of an allele in a large population under selection and genetic drift, Kimura's diffusion."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import clip
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class AlleleFrequencyElement(Element):
    """Frequency of an allele in a large population under selection and genetic drift, Kimura's diffusion.

    The frequency follows the Itô equation `dx = N s x (1 - x) dt + sqrt(2 x (1 - x)) dW`, the continuum limit of the
    Moran process with the scaled selection `N s`, in units of `N` generations. The frequencies 0 and 1, loss and
    fixation, are absorbing, and a boundary keeps the frequency between them.

    Args:
        label:
            The label of the allele or of the group of replicate populations.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial frequency.

        Ns:
            The scaled selection, the population size times the excess fitness.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AlleleFrequencyElement
        >>> AlleleFrequencyElement('allele', Ns=2.0).Ns
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.5,
        Ns: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.Ns = Ns

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the selection, the drift, and the absorbing boundaries of the frequency.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        variance = clip(s.x * (1 - s.x), low=0)
        return [
            RateLaw(s.x, s.Ns * variance),
            NoiseLaw(s.x, sqrt(2 * variance)),
            BoundaryLaw(s.x, clip(s.x, low=0, high=1)),
        ]
