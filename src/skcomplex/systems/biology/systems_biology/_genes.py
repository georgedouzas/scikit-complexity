"""Systems biology systems: gene expression and transcriptional repression."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, ReactionInteraction
from ....core.components.systems import BaseSystem, System
from ....core.specs import StateSpec, VisualSpec
from ..common.kinetics import RepressionInteraction


class GeneElement(Element):
    """Gene with a count `m` of messenger RNA and a count `p` of protein.

    A gene has no laws of its own. Reactions transcribe, translate, and degrade its molecules.

    Args:
        label:
            The label of the gene.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        m:
            The initial number of mRNA molecules.

        p:
            The initial number of protein molecules.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import GeneElement
        >>> GeneElement('lacI', p=20.0).p
        20.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        m: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        p: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.m = m
        self.p = p


def _express(gene: str, b: float, beta: float, delta: float, mode: str) -> list[BaseInteraction]:
    """Return the decay of the mRNA, the translation, and the decay of the protein.

    They are those of a gene.
    """
    return [
        ReactionInteraction(f'mrna_decay_{gene}', reactants={(gene, 'm'): 1}, k=delta, mode=mode),
        ReactionInteraction(
            f'translation_{gene}',
            products={(gene, 'p'): 1},
            modifiers={(gene, 'm'): 1},
            k=b,
            mode=mode,
        ),
        ReactionInteraction(f'protein_decay_{gene}', reactants={(gene, 'p'): 1}, k=beta, mode=mode),
    ]


def _repress(label: str, repressor: str, target: str, parameters: dict[str, float], mode: str) -> RepressionInteraction:
    """Return the transcription of the mRNA of a target gene.

    The protein of a repressor gene represses it.
    """
    return RepressionInteraction(
        label,
        products={(target, 'm'): 1},
        modifiers={(repressor, 'p'): 1},
        alpha=parameters['alpha'],
        alpha0=parameters['alpha0'],
        K=parameters['K'],
        n=parameters['n'],
        mode=mode,
    )


def _circuit(
    proteins: Sequence[float],
    edges: Sequence[tuple[int, int]],
    expression: dict[str, float],
    repression: dict[str, float],
    mode: str,
    *,
    label: str,
) -> System:
    """Return labelled genes with initial protein counts.

    The first gene of each edge represses the second.
    """
    genes = [GeneElement(f'gene_{index}', p=float(p)) for index, p in enumerate(proteins)]
    repressions: list[BaseInteraction] = [
        _repress(f'repression_{source}_{target}', f'gene_{source}', f'gene_{target}', repression, mode)
        for source, target in edges
    ]
    expressions = [interaction for gene in genes for interaction in _express(gene.label, mode=mode, **expression)]
    return System(label, elements=genes, interactions=[*repressions, *expressions])


class RepressilatorSystem(BaseSystem):
    """The repressilator of Elowitz and Leibler.

    Three genes in a ring each repress the next.

    Gene `i` is repressed by gene `i - 1`, so the ring has an odd number of repressions and oscillates when
    transcription is strong and repression steep. Counts are molecule numbers, so the same rates run as differential
    equations or, with `stochastic=True`, as a jump process of single transcriptions, translations, and decays. The
    system has the counts `gene_<i>.m` and `gene_<i>.p`.

    Args:
        proteins:
            The initial protein counts of the three genes, with no mRNA.

        alpha:
            The transcription rate without repressor.

        alpha0:
            The leaky transcription rate.

        K:
            The repressor count at half repression.

        n:
            The Hill coefficient of repression.

        b:
            The translation rate per mRNA molecule.

        beta:
            The decay rate of the proteins.

        delta:
            The decay rate of mRNA.

        stochastic:
            Whether the system is a jump process.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import RepressilatorSystem
        >>> [interaction.modifiers for interaction in RepressilatorSystem().prepare().system_.interactions[:3]]
        [{('gene_2', 'p'): 1}, {('gene_0', 'p'): 1}, {('gene_1', 'p'): 1}]
    """

    field = 'biology/systems_biology'
    topics = ('gene regulation', 'oscillations', 'synthetic biology', 'jump process')
    references = ('Elowitz and Leibler 2000', 'arXiv:2111.04508')
    until = 200.0
    dt = 0.5

    def __init__(
        self,
        proteins: Sequence[float] = (20.0, 0.0, 0.0),
        *,
        alpha: float = 40.0,
        alpha0: float = 0.04,
        K: float = 20.0,
        n: float = 2.0,
        b: float = 2.0,
        beta: float = 0.2,
        delta: float = 1.0,
        stochastic: bool = False,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.proteins = proteins
        self.alpha = alpha
        self.alpha0 = alpha0
        self.K = K
        self.n = n
        self.b = b
        self.beta = beta
        self.delta = delta
        self.stochastic = stochastic

    def _build_system(self) -> System:
        """Return the parts of the repressilator of Elowitz and Leibler."""
        mode = 'jump' if self.stochastic else 'rate'
        expression = {'b': self.b, 'beta': self.beta, 'delta': self.delta}
        repression = {'alpha': self.alpha, 'alpha0': self.alpha0, 'K': self.K, 'n': self.n}
        return _circuit(self.proteins, [(2, 0), (0, 1), (1, 2)], expression, repression, mode, label=self.label)


class ToggleSwitchSystem(BaseSystem):
    """The genetic toggle switch of Gardner, Cantor, and Collins.

    Two genes repress each other.

    With steep and strong repression the switch is bistable: one gene is expressed and silences the other, and the
    initial protein counts decide which. The stochastic mode lets noise flip the switch. The system has the counts
    `gene_0.m`, `gene_0.p`, `gene_1.m`, and `gene_1.p`.

    Args:
        proteins:
            The initial protein counts of the two genes, with no mRNA.

        alpha:
            The transcription rate without repressor.

        alpha0:
            The leaky transcription rate.

        K:
            The repressor count at half repression.

        n:
            The Hill coefficient of repression.

        b:
            The translation rate per mRNA molecule.

        beta:
            The decay rate of the proteins.

        delta:
            The decay rate of mRNA.

        stochastic:
            Whether the system is a jump process.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import ToggleSwitchSystem
        >>> [element.p for element in ToggleSwitchSystem((0.0, 50.0)).prepare().system_.elements]
        [0.0, 50.0]
    """

    field = 'biology/systems_biology'
    topics = ('gene regulation', 'bistability', 'synthetic biology', 'jump process')
    references = ('Gardner, Cantor, and Collins 2000', 'arXiv:2111.04508')
    until = 100.0
    dt = 0.5

    def __init__(
        self,
        proteins: Sequence[float] = (60.0, 0.0),
        *,
        alpha: float = 20.0,
        alpha0: float = 0.0,
        K: float = 20.0,
        n: float = 2.0,
        b: float = 2.0,
        beta: float = 0.2,
        delta: float = 1.0,
        stochastic: bool = False,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.proteins = proteins
        self.alpha = alpha
        self.alpha0 = alpha0
        self.K = K
        self.n = n
        self.b = b
        self.beta = beta
        self.delta = delta
        self.stochastic = stochastic

    def _build_system(self) -> System:
        """Return the parts of the genetic toggle switch of Gardner."""
        mode = 'jump' if self.stochastic else 'rate'
        expression = {'b': self.b, 'beta': self.beta, 'delta': self.delta}
        repression = {'alpha': self.alpha, 'alpha0': self.alpha0, 'K': self.K, 'n': self.n}
        return _circuit(self.proteins, [(1, 0), (0, 1)], expression, repression, mode, label=self.label)


class NegativeAutoregulationSystem(BaseSystem):
    """A gene that represses its own transcription and makes its protein in bursts.

    Every mRNA molecule yields on average `b / delta` proteins before it decays, so with fast mRNA decay the protein
    arrives in geometric bursts and its count fluctuates far more than a Poisson count. Negative feedback lowers the
    mean and the noise. A very large `K` removes the feedback. The system has the counts `gene.m` and `gene.p`.

    Args:
        p:
            The initial protein count, with no mRNA.

        alpha:
            The transcription rate without repressor.

        K:
            The protein count at half repression.

        n:
            The Hill coefficient of repression.

        b:
            The translation rate per mRNA molecule.

        beta:
            The decay rate of the protein.

        delta:
            The decay rate of mRNA.

        stochastic:
            Whether the system is a jump process.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import NegativeAutoregulationSystem
        >>> NegativeAutoregulationSystem().prepare().system_.interactions[0].modifiers
        {('gene', 'p'): 1}
    """

    field = 'biology/systems_biology'
    topics = ('gene regulation', 'noise', 'bursts', 'jump process')
    references = ('Thattai and van Oudenaarden 2001', 'Becskei and Serrano 2000', 'arXiv:1608.06582')
    until = 200.0
    dt = 1.0

    def __init__(
        self,
        p: float = 0.0,
        *,
        alpha: float = 2.0,
        K: float = 50.0,
        n: float = 1.0,
        b: float = 10.0,
        beta: float = 0.1,
        delta: float = 1.0,
        stochastic: bool = True,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.p = p
        self.alpha = alpha
        self.K = K
        self.n = n
        self.b = b
        self.beta = beta
        self.delta = delta
        self.stochastic = stochastic

    def _build_system(self) -> System:
        """Return the parts of a gene that represses its own transcription.

        The gene makes its protein in bursts.
        """
        mode = 'jump' if self.stochastic else 'rate'
        feedback = _repress(
            'autoregulation',
            'gene',
            'gene',
            {'alpha': self.alpha, 'alpha0': 0.0, 'K': self.K, 'n': self.n},
            mode,
        )
        expression = _express('gene', self.b, self.beta, self.delta, mode)
        return System(self.label, elements=[GeneElement('gene', p=self.p)], interactions=[feedback, *expression])
