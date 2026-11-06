"""Evolution systems: quasispecies under mutation and selection, and hypercycles of catalytic molecules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from functools import partial
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.system import System
from ....core.errors import ValidationError
from ..._catalog import register_system
from ._catalysis import CatalysisInteraction
from ._genotype_class import GenotypeClassElement
from ._master_sequence import MasterSequenceElement
from ._molecule import MoleculeElement


def _mean_fitness(master: SimpleNamespace) -> Expression:
    """Return the mean fitness `x0 + (1 - sigma) (1 - x0)` of a population around a master sequence."""
    return master.x + (1 - master.sigma) * (1 - master.x)


def _replicate_master(s: SimpleNamespace) -> Expression:
    """Return the next share of the master sequence, replicated without mutation, `x0 (1 - u) / <W>`."""
    master = s.target
    return master.x * (1 - master.u) / _mean_fitness(master)


def _replicate_mutants(s: SimpleNamespace, *, first: bool, last: bool) -> Expression:
    """Return the next share of a class of mutants, `(u W_{k-1} x_{k-1} + kept (1 - sigma) x_k) / <W>`.

    The sources are the class one mutation closer to the master sequence and the master sequence. The fitness of the
    source is 1 for the first class, whose source is the master sequence, and the last class keeps its own mutants.
    """
    source, master = s.sources
    u, sigma = master.u, master.sigma
    source_fitness = 1 if first else 1 - sigma
    kept = 1 if last else 1 - u
    gained = u * source_fitness * source.x + kept * (1 - sigma) * s.target.x
    return gained / _mean_fitness(master)


@register_system(
    field='biology/evolution',
    tags=('quasispecies', 'error threshold', 'discrete map', 'mutation', 'selection'),
    references=('Eigen 1971', 'Gros 2015'),
    defaults={'until': 100, 'dt': 1},
)
def eigen_quasispecies(classes: int = 12, *, u: float = 0.3, sigma: float = 0.5) -> System:
    """Return the shares of a population at every number of mutations from the master sequence, the linear chain system.

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
            The system, with the shares `master.x` and `k1.x` to `k<classes>.x`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import eigen_quasispecies
        >>> [element.label for element in eigen_quasispecies(2).elements]
        ['master', 'k1', 'k2']
    """
    labels = ['master', *(f'k{k}' for k in range(1, classes + 1))]
    elements = [
        MasterSequenceElement('master', x=1.0, u=u, sigma=sigma),
        *(GenotypeClassElement(label) for label in labels[1:]),
    ]
    selection = MapInteraction('selection', target='master', state='x', function=_replicate_master)
    mutations = [
        MapInteraction(
            f'mutation_{target}',
            target=target,
            state='x',
            function=partial(_replicate_mutants, first=source == 'master', last=target == labels[-1]),
            sources=(source, 'master'),
        )
        for source, target in itertools.pairwise(labels)
    ]
    interactions = [selection, *mutations]
    return System(elements=elements, interactions=interactions)


@register_system(
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

    The growth rates rise as `lambda_i = alpha i`. In a `'cycle'`, every molecule catalyses the next one. Up to four
    molecules the hypercycle keeps all of them at a fixed point, and from five some members die out and the others
    settle at a fixed point. In a `'complete'` network every molecule catalyses every other, and about
    `sqrt(2 omega / alpha)` of the fastest replicators survive while the others die out.

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
            The system, with the concentrations `soup.x` of the molecules, members of the group `soup`.

    Raises:
        ValidationError:
            The topology is unknown.

    Examples:
        >>> from skcomplex.systems.biology.evolution import hypercycles
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
    soup = GroupElement(
        'soup',
        element=MoleculeElement,
        size=molecules,
        values={
            'x': [1 / molecules] * molecules,
            'growth': [float(alpha * index) for index in range(1, molecules + 1)],
        },
    )
    catalysis = CatalysisInteraction(
        'catalysis',
        target='soup',
        omega=omega,
        topology=tuple(tuple(float(value) for value in row) for row in links),
    )
    return System(elements=[soup], interactions=[catalysis])
