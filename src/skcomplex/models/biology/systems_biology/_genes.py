"""Systems biology models: gene expression and transcriptional repression."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial

from ....core.base import Domain, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Jump, Law, Rate, Symbols
from ....core.runtime import register_model
from ....core.system import System


class Gene(Element):
    """Gene whose messenger RNA decays, is translated into protein, and whose protein decays.

    The mRNA count `m` decays at rate `delta * m`, translation adds protein at rate `b * m`, and the protein count `p`
    decays at rate `beta * p`. In stochastic mode each of these is a random event that changes a count by one.

    Args:
        label:
            The label of the gene.

        m:
            The initial number of mRNA molecules.

        p:
            The initial number of protein molecules.

        delta:
            The decay rate of mRNA.

        b:
            The translation rate per mRNA molecule.

        beta:
            The decay rate of the protein.

        stochastic:
            Whether expression is a sequence of random events.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import Gene
        >>> Gene('lacI', p=20.0).p
        20.0
    """

    m: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    p: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    delta: float = parameter(default=1.0, domain=Domain.POSITIVE)
    b: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.2, domain=Domain.POSITIVE)
    stochastic: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the decay of mRNA, the translation, and the decay of the protein.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        if self.stochastic:
            return [Jump(s.delta * s.m, {s.m: -1}), Jump(s.b * s.m, {s.p: 1}), Jump(s.beta * s.p, {s.p: -1})]
        return [Rate(s.m, -s.delta * s.m), Rate(s.p, s.b * s.m - s.beta * s.p)]


class Repression(Interaction):
    """Transcription of a target gene repressed by the protein of a repressor gene, a Hill function of its count.

    The target gains mRNA at rate `alpha0 + alpha / (1 + (p / K)^n)`, where `p` is the repressor protein. In stochastic
    mode each transcription is a random event that adds one mRNA molecule. A gene that represses itself binds both
    roles to the same gene.

    Args:
        label:
            The label of the interaction.

        repressor:
            The path of the repressing gene.

        target:
            The path of the repressed gene.

        alpha:
            The transcription rate without repressor.

        alpha0:
            The leaky transcription rate at saturating repressor.

        K:
            The repressor count at half repression.

        n:
            The Hill coefficient of repression.

        stochastic:
            Whether transcriptions are random events.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import Repression
        >>> Repression('repression', repressor='tetR', target='cI', n=2.0).n
        2.0
    """

    repressor: str = role()
    target: str = role()
    alpha: float = parameter(default=40.0, domain=Domain.NONNEGATIVE)
    alpha0: float = parameter(default=0.04, domain=Domain.NONNEGATIVE)
    K: float = parameter(default=20.0, domain=Domain.POSITIVE)
    n: float = parameter(default=2.0, domain=Domain.POSITIVE)
    stochastic: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the repressed transcription `alpha0 + alpha / (1 + (p / K)^n)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        rate = s.alpha0 + s.alpha / (1 + (s.repressor.p / s.K) ** s.n)
        if self.stochastic:
            return [Jump(rate, {s.target.m: 1})]
        return [Rate(s.target.m, rate)]


def _circuit(
    proteins: Sequence[float],
    edges: Sequence[tuple[int, int]],
    gene: Gene,
    repression: Callable[..., Repression],
) -> System:
    """Return copies of a gene with initial protein counts, where the first gene of each edge represses the second."""
    genes = [replace(gene, label=f'gene_{index}', p=float(p)) for index, p in enumerate(proteins)]
    repressions = [
        repression(f'repression_{source}_{target}', repressor=f'gene_{source}', target=f'gene_{target}')
        for source, target in edges
    ]
    return System(components=genes, interactions=repressions)


@register_model(
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
            Whether the model is a jump process.

    Returns:
        system:
            The model, with the counts `gene_<i>.m` and `gene_<i>.p`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import repressilator
        >>> [interaction.repressor for interaction in repressilator().interactions]
        ['gene_2', 'gene_0', 'gene_1']
    """
    gene = Gene('gene', b=b, beta=beta, delta=delta, stochastic=stochastic)
    repression = partial(Repression, alpha=alpha, alpha0=alpha0, K=K, n=n, stochastic=stochastic)
    return _circuit(proteins, [(2, 0), (0, 1), (1, 2)], gene, repression)


@register_model(
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
            Whether the model is a jump process.

    Returns:
        system:
            The model, with the counts `gene_0.m`, `gene_0.p`, `gene_1.m`, and `gene_1.p`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import toggle_switch
        >>> [element.p for element in toggle_switch((0.0, 50.0)).elements]
        [0.0, 50.0]
    """
    gene = Gene('gene', b=b, beta=beta, delta=delta, stochastic=stochastic)
    repression = partial(Repression, alpha=alpha, alpha0=alpha0, K=K, n=n, stochastic=stochastic)
    return _circuit(proteins, [(1, 0), (0, 1)], gene, repression)


@register_model(
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
            Whether the model is a jump process.

    Returns:
        system:
            The model, with the counts `gene.m` and `gene.p`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import negative_autoregulation
        >>> negative_autoregulation().interactions[0].repressor
        'gene'
    """
    gene = Gene('gene', p=p, b=b, beta=beta, delta=delta, stochastic=stochastic)
    feedback = Repression(
        'autoregulation',
        repressor='gene',
        target='gene',
        alpha=alpha,
        alpha0=0.0,
        K=K,
        n=n,
        stochastic=stochastic,
    )
    return System(components=[gene], interactions=[feedback])
