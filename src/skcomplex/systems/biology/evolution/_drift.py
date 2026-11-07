"""Evolution systems: allele frequencies under selection and genetic drift in the diffusion limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import clip
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import EulerMaruyamaSolver
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

    law_classes = (
        BoundaryLaw,
        NoiseLaw,
        RateLaw,
    )

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


class KimuraDiffusionSystem(BaseSystem):
    """Replicate populations whose allele frequencies diffuse by drift until the allele is lost or fixed.

    Each replicate follows `dx = N s x (1 - x) dt + sqrt(2 x (1 - x)) dW`, whose density obeys Kimura's diffusion
    equation `dp/dt = -N s d/dx [x (1 - x) p] + d^2/dx^2 [x (1 - x) p]`. Every replicate ends at loss or fixation, and
    an allele that starts at the frequency `x` fixes with the probability `(1 - exp(-N s x)) / (1 - exp(-N s))`, or `x`
    for a neutral allele. The system has the frequencies `alleles.x` of the replicates.

    Args:
        x:
            The initial frequency of the allele.

        Ns:
            The scaled selection, the population size times the excess fitness.

        replicates:
            The number of independent replicate populations.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import KimuraDiffusionSystem
        >>> KimuraDiffusionSystem(replicates=10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'biology/evolution'
    topics = ('population genetics', 'genetic drift', 'fixation', 'sde', 'diffusion')
    references = ('Kimura 1955', 'Bertin 2021')
    until = 10.0
    dt = 0.05
    solver = EulerMaruyamaSolver(dt=0.001)

    def __init__(
        self,
        x: float = 0.1,
        *,
        Ns: float = 2.0,
        replicates: int = 500,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.Ns = Ns
        self.replicates = replicates

    def _build_system(self) -> System:
        """Return the parts of the system."""
        alleles = GroupElement(
            'alleles',
            element=AlleleFrequencyElement,
            size=self.replicates,
            values={'x': self.x, 'Ns': self.Ns},
        )
        return System(self.label, elements=[alleles])
