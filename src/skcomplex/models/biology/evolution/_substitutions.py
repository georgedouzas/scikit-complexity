"""Evolution models: monomorphic populations that walk a fitness landscape by the origin and fixation of mutants."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import ZZ, Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, JumpLaw
from ....core.laws.utils import total
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model


def _fitness_of(genome: SimpleNamespace, F0: Expression, couplings: Array) -> Expression:
    """Return the fitness `F0 + sum_i h_i s_i + sum_ij J_ij s_i s_j / 2` of the genome of a group of loci."""
    return F0 + total(genome.h * genome.s) + total(genome.s * total(genome.s, weights=couplings)) / 2


def _field_share(member: SimpleNamespace) -> Expression:
    """Return the share `h_i s_i` of one locus in the fitness of the genome."""
    return member.h * member.s


def _coupling_share(alleles: Array, couplings: Array) -> float:
    """Return the share `sum_ij J_ij s_i s_j / 2` of the couplings between loci in the fitness of the genome."""
    return float(alleles @ couplings @ alleles / 2)


def _genome_fitness(s: SimpleNamespace) -> Expression:
    """Return the fitness of the genome, the base fitness of the interaction `substitution` and the shares of loci."""
    return s.substitution.F0 + s.field_share + s.coupling_share


class LocusElement(Element):
    """Locus of the genome shared by a monomorphic population, with an allele of `-1` or `1`.

    Args:
        label:
            The label of the locus or the group of loci.

        s:
            The allele, `-1` or `1`.

        h:
            The random field of the locus, the fitness gained by the allele `1` over the allele `-1`, halved.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.evolution import LocusElement
        >>> GroupElement('genome', element=LocusElement, size=10, values={'h': 0.05}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('image', {'low': -1.0, 'high': 1.0, 'history': 64}),
        *,
        s: Annotated[float, StateSpec(domain=ZZ)] = -1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.h = h


class SubstitutionInteraction(Interaction):
    """Origin and fixation of mutants that flip one locus of the genome of a monomorphic population.

    Every locus of each of the `N` individuals mutates at the rate `mu`, so mutants of a given locus arise at the rate
    `N mu`. A mutant from the fitness `f` to `f'` fixes with the Moran probability
    `(1 - f / f') / (1 - (f / f')^N)`, so the population substitutes the allele of the locus at the product of the two.
    The fitness is `F0 + sum_i h_i s_i + sum_ij J_ij s_i s_j / 2`, and a flip of locus `i` changes it by
    `-2 s_i (h_i + sum_j J_ij s_j)`.

    Args:
        label:
            The label of the interaction.

        couplings:
            The symmetric matrix `J` of couplings between loci, with a zero diagonal, or `None` for no couplings.

        **values:
            The values of the declared variables, by name:

            genome:
                The path of the group of loci.

            N:
                The size of the population.

            mu:
                The rate at which one locus of an individual mutates.

            F0:
                The fitness of a genome without fields and couplings, above the sum of the `|h_i|` and `|J_ij|` so that
                every fitness is positive.

    Examples:
        >>> from skcomplex.models.biology.evolution import SubstitutionInteraction
        >>> SubstitutionInteraction('substitution', genome='genome', N=50.0).N
        50.0
    """

    def __init__(
        self,
        label: str,
        *,
        genome: Annotated[str, RoleSpec()],
        N: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        F0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        couplings: object = None,
    ) -> None:
        super().__init__(label)
        self.genome = genome
        self.N = N
        self.mu = mu
        self.F0 = F0
        self.couplings = couplings

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the substitutions of the allele of every locus, from `-1` to `1` and back.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        genome = s.genome
        local = genome.h
        fitness = s.F0 + total(genome.h * genome.s)
        if self.couplings is not None:
            couplings = np.asarray(self.couplings, dtype=float)
            local = genome.h + total(genome.s, weights=couplings)
            fitness = _fitness_of(genome, s.F0, couplings)
        ratio = fitness / (fitness - 2 * genome.s * local)
        rate = s.N * s.mu * abs((1 - ratio) / (1 - ratio**s.N))
        return [
            JumpLaw(rate * (1 - genome.s) / 2, {genome.s: 2}),
            JumpLaw(rate * (1 + genome.s) / 2, {genome.s: -2}),
        ]


def _walk_observables(system: System) -> list[ObservableSpec]:
    """Return the shares of the fields and the couplings in the fitness, and the fitness of the genome."""
    couplings = system.interactions[0].couplings
    loci = system.elements[0].check().prepare().size_
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


@register_model(
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
            The model, with the alleles `genome.s` on a one-dimensional lattice of loci, starting from the genome of all
            `-1`, and the fitness `fitness` of the genome.

    Examples:
        >>> from skcomplex.models.biology.evolution import origin_fixation_walk
        >>> origin_fixation_walk(8).elements[0].check().prepare().size_
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
        interactions=[SubstitutionInteraction('substitution', genome='genome', N=N, mu=mu, F0=F0, couplings=couplings)],
    )
