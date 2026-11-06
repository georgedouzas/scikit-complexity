"""Systems biology systems: gene expression and transcriptional repression."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, ReactionInteraction
from ....core.components.system import System
from ....core.specs import StateSpec, VisualSpec
from ..._catalog import register_system
from ._repression import RepressionInteraction


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
    """Return the decay of the mRNA, the translation, and the decay of the protein of a gene."""
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
    """Return the transcription of the mRNA of a target gene repressed by the protein of a repressor gene."""
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
) -> System:
    """Return genes with initial protein counts, where the first gene of each edge represses the second."""
    genes = [GeneElement(f'gene_{index}', p=float(p)) for index, p in enumerate(proteins)]
    repressions: list[BaseInteraction] = [
        _repress(f'repression_{source}_{target}', f'gene_{source}', f'gene_{target}', repression, mode)
        for source, target in edges
    ]
    expressions = [interaction for gene in genes for interaction in _express(gene.label, mode=mode, **expression)]
    return System(elements=genes, interactions=[*repressions, *expressions])


@register_system(
    field='biology/systems_biology',
    tags=('gene regulation', 'oscillations', 'synthetic biology', 'jump process'),
    references=('Elowitz and Leibler 2000', 'arXiv:2111.04508'),
    defaults={'until': 200.0, 'dt': 0.5},
)
def repressilator(
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
) -> System:
    """Return the repressilator of Elowitz and Leibler, three genes in a ring that each repress the next.

    Gene `i` is repressed by gene `i - 1`, so the ring has an odd number of repressions and oscillates when
    transcription is strong and repression steep. Counts are molecule numbers, so the same rates run as differential
    equations or, with `stochastic=True`, as a jump process of single transcriptions, translations, and decays.

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

    Returns:
        system:
            The system, with the counts `gene_<i>.m` and `gene_<i>.p`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import repressilator
        >>> [interaction.modifiers for interaction in repressilator().interactions[:3]]
        [{('gene_2', 'p'): 1}, {('gene_0', 'p'): 1}, {('gene_1', 'p'): 1}]
    """
    mode = 'jump' if stochastic else 'rate'
    expression = {'b': b, 'beta': beta, 'delta': delta}
    repression = {'alpha': alpha, 'alpha0': alpha0, 'K': K, 'n': n}
    return _circuit(proteins, [(2, 0), (0, 1), (1, 2)], expression, repression, mode)


@register_system(
    field='biology/systems_biology',
    tags=('gene regulation', 'bistability', 'synthetic biology', 'jump process'),
    references=('Gardner, Cantor, and Collins 2000', 'arXiv:2111.04508'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def toggle_switch(
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
) -> System:
    """Return the genetic toggle switch of Gardner, Cantor, and Collins, two genes that repress each other.

    With steep and strong repression the switch is bistable: one gene is expressed and silences the other, and the
    initial protein counts decide which. The stochastic mode lets noise flip the switch.

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

    Returns:
        system:
            The system, with the counts `gene_0.m`, `gene_0.p`, `gene_1.m`, and `gene_1.p`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import toggle_switch
        >>> [element.p for element in toggle_switch((0.0, 50.0)).elements]
        [0.0, 50.0]
    """
    mode = 'jump' if stochastic else 'rate'
    expression = {'b': b, 'beta': beta, 'delta': delta}
    repression = {'alpha': alpha, 'alpha0': alpha0, 'K': K, 'n': n}
    return _circuit(proteins, [(1, 0), (0, 1)], expression, repression, mode)


@register_system(
    field='biology/systems_biology',
    tags=('gene regulation', 'noise', 'bursts', 'jump process'),
    references=('Thattai and van Oudenaarden 2001', 'Becskei and Serrano 2000', 'arXiv:1608.06582'),
    defaults={'until': 200.0, 'dt': 1.0},
)
def negative_autoregulation(
    p: float = 0.0,
    *,
    alpha: float = 2.0,
    K: float = 50.0,
    n: float = 1.0,
    b: float = 10.0,
    beta: float = 0.1,
    delta: float = 1.0,
    stochastic: bool = True,
) -> System:
    """Return a gene that represses its own transcription and makes its protein in bursts.

    Every mRNA molecule yields on average `b / delta` proteins before it decays, so with fast mRNA decay the protein
    arrives in geometric bursts and its count fluctuates far more than a Poisson count. Negative feedback lowers the
    mean and the noise. A very large `K` removes the feedback.

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

    Returns:
        system:
            The system, with the counts `gene.m` and `gene.p`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import negative_autoregulation
        >>> negative_autoregulation().interactions[0].modifiers
        {('gene', 'p'): 1}
    """
    mode = 'jump' if stochastic else 'rate'
    feedback = _repress('autoregulation', 'gene', 'gene', {'alpha': alpha, 'alpha0': 0.0, 'K': K, 'n': n}, mode)
    expression = _express('gene', b, beta, delta, mode)
    return System(elements=[GeneElement('gene', p=p)], interactions=[feedback, *expression])
