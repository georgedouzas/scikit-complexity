"""Evolution systems: monomorphic populations that walk a fitness landscape by the origin and fixation of mutants."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import LatticeElement
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.specs import ObservableSpec
from ....core.types import Array
from ..._catalog import register_system
from ._locus import LocusElement
from ._substitution import SubstitutionInteraction


def _field_share(member: SimpleNamespace) -> Expression:
    """Return the share `h_i s_i` of one locus in the fitness of the genome."""
    return member.h * member.s


def _coupling_share(alleles: Array, couplings: Array) -> float:
    """Return the share `sum_ij J_ij s_i s_j / 2` of the couplings between loci in the fitness of the genome."""
    return float(alleles @ couplings @ alleles / 2)


def _genome_fitness(s: SimpleNamespace) -> Expression:
    """Return the fitness of the genome, the base fitness of the interaction `substitution` and the shares of loci."""
    return s.substitution.F0 + s.field_share + s.coupling_share


def _walk_observables(system: System) -> list[ObservableSpec]:
    """Return the shares of the fields and the couplings in the fitness, and the fitness of the genome."""
    couplings = system.interactions[0].couplings
    loci = system.elements[0].prepare().size_
    matrix = np.zeros((loci, loci)) if couplings is None else np.asarray(couplings)
    return [
        ObservableSpec('field_share', _field_share, group='genome', reduce='sum'),
        ObservableSpec(
            'coupling_share',
            group='genome',
            variable='s',
            reduce=partial(_coupling_share, couplings=matrix),
        ),
        ObservableSpec('fitness', _genome_fitness),
    ]


@register_system(
    field='biology/evolution',
    tags=('population genetics', 'fitness landscape', 'jump process', 'fixation', 'strong selection weak mutation'),
    references=('Bertin 2021', 'Sella and Hirsh 2005', 'Drossel 2001'),
    defaults={'until': 640.0, 'dt': 10.0},
    observables=_walk_observables,
)
def origin_fixation_walk(
    loci: int = 96,
    *,
    N: float = 100.0,
    mu: float = 0.01,
    field: float = 0.01,
    coupling: float = 0.0,
    F0: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a monomorphic population that climbs a fitness landscape as mutants arise and fix one at a time.

    Under strong selection and weak mutation the population carries one genome of `loci` alleles `-1` or `1`, and
    every substitution of a mutant moves it to a neighbouring genome. The walk climbs the landscape and settles on the
    stationary distribution `Q ~ f^(N - 1)`, a Boltzmann distribution of the energy `-ln f` at the temperature
    `1 / (N - 1)`. Random fields alone, `coupling = 0`, give a landscape with one peak. Couplings `J_ij` of variance
    `coupling^2 / loci` give the rugged landscape of a spin glass, where a small population escapes local peaks.

    Args:
        loci:
            The number of loci of the genome.

        N:
            The size of the population.

        mu:
            The rate at which one locus of an individual mutates.

        field:
            The standard deviation of the random fields `h_i`.

        coupling:
            The scale of the random couplings `J_ij`, or 0 for none.

        F0:
            The fitness of a genome without fields and couplings, above the sum of the `|h_i|` and `|J_ij|` so that
            every fitness is positive.

        random_state:
            The seed or generator of the fields and couplings.

    Returns:
        system:
            The system, with the alleles `genome.s` on a one-dimensional lattice of loci, starting from the genome of
            all `-1`, and the fitness `fitness` of the genome.

    Examples:
        >>> from skcomplex.systems.biology.evolution import origin_fixation_walk
        >>> origin_fixation_walk(8).elements[0].prepare().size_
        8
    """
    rng = np.random.default_rng(random_state)
    fields = rng.normal(0.0, field, loci)
    upper = np.triu(rng.normal(0.0, coupling / np.sqrt(loci), (loci, loci)), 1)
    matrix = upper + upper.T
    couplings = matrix if coupling else None
    genome = LatticeElement(
        'genome',
        element=LocusElement,
        space=LatticeSpace(loci),
        values={'s': -np.ones(loci), 'h': fields},
    )
    return System(
        elements=[genome],
        interactions=[SubstitutionInteraction('substitution', target='genome', N=N, mu=mu, F0=F0, couplings=couplings)],
    )
