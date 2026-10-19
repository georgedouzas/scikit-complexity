"""Evolution models: quasispecies under mutation and selection, and hypercycles of catalytic molecules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence

import numpy as np

from ....core.base import Domain, Expression, ValidationError, parameter, role, state
from ....core.model import Element, Interaction, Law, Rate, Symbols, System, Update, total
from ....core.runtime import register_model


def _mean_fitness(master: Symbols) -> Expression:
    """Return the mean fitness `x0 + (1 - sigma) (1 - x0)` of a population around a master sequence."""
    return master.x + (1 - master.sigma) * (1 - master.x)


class GenotypeClass(Element):
    """Share of a population whose genomes lie a given number of point mutations away from the master sequence.

    Args:
        label:
            The label of the class.

        x:
            The initial share.

    Examples:
        >>> from skcomplex.models.biology.evolution import GenotypeClass
        >>> GenotypeClass('k1', x=0.1).x
        0.1
    """

    x: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class MasterSequence(GenotypeClass):
    """Share of the population with the fittest genome, in a sharp-peak landscape where every other genome is less fit.

    The master sequence has fitness 1 and every other genome `1 - sigma`. A genome mutates at one locus with
    probability `u` per generation.

    Args:
        label:
            The label of the class.

        x:
            The initial share.

        u:
            The mutation rate per genome and generation.

        sigma:
            The selective disadvantage of the other genomes.

    Examples:
        >>> from skcomplex.models.biology.evolution import MasterSequence
        >>> MasterSequence('master', u=0.2).u
        0.2
    """

    u: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    sigma: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the replication of the master sequence without mutation, `x0' = x0 (1 - u) / <W>`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Update(s.x, s.x * (1 - s.u) / _mean_fitness(s))]


class Mutation(Interaction):
    """Replication of a class of mutants, which gains the mutants of the class before it and loses its own.

    With the mean fitness `<W>`, the share `k` steps from the master sequence follows
    `x_k' = (u W_{k-1} x_{k-1} + (1 - u) (1 - sigma) x_k) / <W>`, where `W_0 = 1` and `W_k = 1 - sigma` otherwise.
    The last class of a truncated chain keeps its own mutants.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the class.

        source:
            The path of the class one mutation closer to the master sequence.

        master:
            The path of the master sequence.

        last:
            Whether the class is the last of the chain.

    Examples:
        >>> from skcomplex.models.biology.evolution import Mutation
        >>> Mutation('mutation_k1', target='k1', source='master', master='master').last
        False
    """

    target: str = role()
    source: str = role()
    master: str = role()
    last: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the next share of the class.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        u, sigma = s.master.u, s.master.sigma
        source_fitness = 1 if self.source == self.master else 1 - sigma
        kept = 1 if self.last else 1 - u
        gained = u * source_fitness * s.source.x + kept * (1 - sigma) * s.target.x
        return [Update(s.target.x, gained / _mean_fitness(s.master))]


Molecule = Element.define(
    'Molecule',
    states={'x': state(default=0.5, domain=Domain.NONNEGATIVE)},
    parameters={'growth': 0.0},
)


class Catalysis(Interaction):
    """Catalysed replication of the molecules of a group, with a common outflow that keeps the total at 1.

    Molecule `i` replicates at the rate `lambda_i + omega sum_j T_ij x_j`, with the 0 or 1 topology `T` of the
    catalytic links read as the weights of a total over the molecules, and the outflow `phi` is the total replication:
    `dx_i/dt = x_i (lambda_i + omega sum_j T_ij x_j - phi)`.

    Args:
        label:
            The label of the interaction.

        molecules:
            The path of the group of molecules.

        omega:
            The catalytic rate of a link.

        topology:
            The catalytic links, 1 where molecule `j` catalyses molecule `i`.

    Examples:
        >>> from skcomplex.models.biology.evolution import Catalysis
        >>> Catalysis('catalysis', molecules='soup', topology=((0, 1), (1, 0))).topology
        ((0, 1), (1, 0))
    """

    molecules: str = role()
    omega: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    topology: Sequence[Sequence[float]] = ((0.0, 1.0), (1.0, 0.0))

    def laws(self, s: Symbols) -> list[Law]:
        """Return the catalysed replication and the common outflow.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        molecule = s.molecules
        rate = molecule.growth + s.omega * total(molecule.x, weights=np.asarray(self.topology, dtype=float))
        replication = molecule.x * rate
        return [Rate(molecule.x, replication - molecule.x * total(replication))]


@register_model(
    field='biology/evolution',
    tags=('quasispecies', 'error threshold', 'discrete map', 'mutation', 'selection'),
    references=('Eigen 1971', 'Gros 2015'),
    defaults={'until': 100, 'dt': 1},
)
def eigen_quasispecies(classes: int = 12, *, u: float = 0.3, sigma: float = 0.5) -> System:
    """Return the shares of a population at every number of mutations from the master sequence, the linear chain model.

    Below the error threshold, `u < sigma`, a quasispecies forms around the master sequence, whose share settles at
    `x0 = 1 - u / sigma`. Above it the master sequence is lost, the error catastrophe. Back mutations and multiple
    mutations are neglected, so the chain holds only forward mutations, and the last class collects the far mutants.

    Args:
        classes:
            The number of classes of mutants after the master sequence.

        u:
            The mutation rate per genome and generation.

        sigma:
            The selective disadvantage of the mutants.

    Returns:
        system:
            The model, with the shares `master.x` and `k1.x` to `k<classes>.x`.

    Examples:
        >>> from skcomplex.models.biology.evolution import eigen_quasispecies
        >>> [element.label for element in eigen_quasispecies(2).elements]
        ['master', 'k1', 'k2']
    """
    labels = ['master', *(f'k{k}' for k in range(1, classes + 1))]
    elements = [MasterSequence('master', x=1.0, u=u, sigma=sigma), *(GenotypeClass(label) for label in labels[1:])]
    interactions = [
        Mutation(f'mutation_{target}', target=target, source=source, master='master', last=target == labels[-1])
        for source, target in itertools.pairwise(labels)
    ]
    return System(elements=elements, interactions=interactions)


@register_model(
    field='biology/evolution',
    tags=('hypercycle', 'autocatalysis', 'prebiotic', 'ode'),
    references=('Eigen and Schuster 1977', 'Gros 2015'),
    defaults={'until': 50.0, 'dt': 0.5},
)
def hypercycles(
    molecules: int = 4,
    *,
    alpha: float = 0.1,
    omega: float = 1.0,
    topology: str = 'cycle',
) -> System:
    """Return a network of self-replicating molecules whose catalysis keeps a set of them alive, a hypercycle.

    The growth rates rise as `lambda_i = alpha i`. In a `'cycle'`, every molecule catalyses the next one, and the
    hypercycle keeps all of them, at a fixed point up to four molecules and on a limit cycle from five. In a
    `'complete'` network every molecule catalyses every other, and about `sqrt(2 omega / alpha)` of the fastest
    replicators survive while the others die out.

    Args:
        molecules:
            The number of molecules.

        alpha:
            The step of the autocatalytic growth rates.

        omega:
            The catalytic rate of a link.

        topology:
            `'cycle'` or `'complete'`.

    Returns:
        system:
            The model, with the concentrations `soup.x` of the molecules, members of the group `soup`.

    Raises:
        ValidationError:
            The topology is unknown.

    Examples:
        >>> from skcomplex.models.biology.evolution import hypercycles
        >>> hypercycles(3).interactions[0].topology
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    """
    if topology == 'cycle':
        links = np.roll(np.eye(molecules), 1, axis=0)
    elif topology == 'complete':
        links = 1 - np.eye(molecules)
    else:
        error_msg = f'Unknown topology {topology!r}. Use cycle or complete.'
        raise ValidationError(error_msg)
    soup = Molecule.group(
        'soup',
        molecules,
        x=[1 / molecules] * molecules,
        growth=[float(alpha * index) for index in range(1, molecules + 1)],
    )
    catalysis = Catalysis(
        'catalysis',
        molecules='soup',
        omega=omega,
        topology=tuple(tuple(float(value) for value in row) for row in links),
    )
    return System(elements=[soup], interactions=[catalysis])
