"""Evolution systems: allele frequencies under selection and genetic drift.

The frequencies are in the diffusion limit.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import clip
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class AlleleFrequencyElement(Element):
    """Frequency `x` of an allele in a population, with the scaled selection `Ns` of the allele.

    Args:
        label:
            The label of the allele or of the group of replicate populations.

        x:
            The initial frequency.

        Ns:
            The scaled selection, the population size times the excess fitness.

        size:
            The number of replicate populations, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AlleleFrequencyElement
        >>> AlleleFrequencyElement('allele', Ns=2.0).Ns
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed(0, 1)] = 0.5,
        Ns: float = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.Ns = Ns


class KimuraDriftInteraction(Interaction):
    """Selection and genetic drift of an allele frequency, Kimura's diffusion.

    The population is large. The frequency follows the Itô equation `dx = N s x (1 - x) dt + sqrt(2 x (1 - x)) dW`,
    the continuum limit of the Moran process with the scaled selection `N s` of the target, in units of `N`
    generations. The frequencies 0 and 1, loss and fixation, are absorbing, and a boundary keeps the frequency between
    them.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the alleles, whose values are `x` and `Ns`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import KimuraDriftInteraction
        >>> KimuraDriftInteraction('drift', target='alleles').target
        'alleles'
    """

    law_classes = (
        UpdateLaw,
        NoiseLaw,
        RateLaw,
    )

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the selection, the drift, and the absorbing boundaries of the frequency.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, Ns = context.target.x, context.target.Ns
            variance = clip(x * (1 - x), low=0)
            self.laws_ = [
                RateLaw(x, Ns * variance),
                NoiseLaw(x, sqrt(2 * variance)),
                UpdateLaw(x, clip(x, low=0, high=1)),
            ]


class KimuraDiffusionSystem(BaseSystem):
    """Replicate populations whose allele frequencies diffuse by drift.

    The frequencies diffuse until the allele is lost or fixed.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import KimuraDiffusionSystem
        >>> KimuraDiffusionSystem(replicates=10).prepare().system_.elements[0].size
        10
    """

    field = 'biology/evolution'
    topics = ('population genetics', 'genetic drift', 'fixation', 'sde', 'diffusion')
    references = ('Kimura 1955', 'Bertin 2021')
    until = 10.0
    dt = 0.05

    def __init__(
        self,
        x: float = 0.1,
        *,
        Ns: float = 2.0,
        replicates: int = 500,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.Ns = Ns
        self.replicates = replicates

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of replicate populations.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        alleles = AlleleFrequencyElement('alleles', x=self.x, Ns=self.Ns, size=self.replicates)
        self.system_ = System(
            self.label,
            elements=[alleles],
            interactions=[KimuraDriftInteraction('drift', target='alleles')],
            solver=EulerMaruyamaSolver(dt=0.001),
        )
